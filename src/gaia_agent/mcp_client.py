from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from gaia_agent.config import RunConfig
from gaia_agent.providers.base import ToolSpec

_PUBLIC_TOOLS = frozenset(
    {"web_search", "read_webpage", "execute_python", "read_attachment", "submit_answer"}
)


class ToolExecutionError(RuntimeError):
    pass


class MCPToolClient:
    def __init__(
        self,
        *,
        task_root: Path,
        config: RunConfig | None = None,
        allowed: set[str] | frozenset[str] | None = None,
        attachments: Iterable[Path] = (),
    ) -> None:
        self._task_root = task_root.resolve()
        self._config = config or RunConfig.from_env()
        self._allowed = frozenset(allowed) if allowed is not None else _PUBLIC_TOOLS
        unknown = self._allowed - _PUBLIC_TOOLS
        if unknown:
            raise ValueError(f"unknown MCP tools in allowlist: {sorted(unknown)}")
        self._attachments = self._validate_attachments(attachments)
        self._client_cm: AbstractAsyncContextManager[Client] | None = None
        self._client: Client | None = None
        self.tool_specs: tuple[ToolSpec, ...] = ()

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def __aenter__(self) -> Self:
        if self._client is not None:
            raise RuntimeError("MCP client is already connected")
        trusted_source_root = Path(__file__).resolve().parents[1]
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-P", "-m", "gaia_agent.mcp_server"],
            env=self._minimal_server_env(),
            cwd=trusted_source_root,
        )
        self._client_cm = Client(stdio_client(parameters))
        try:
            self._client = await self._client_cm.__aenter__()
            listed = await self._client.list_tools()
        except BaseException:
            self._client = None
            if self._client_cm is not None:
                await self._client_cm.__aexit__(None, None, None)
                self._client_cm = None
            raise
        self.tool_specs = tuple(
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                parameters=_strict_function_schema(tool.input_schema),
            )
            for tool in listed.tools
            if tool.name in self._allowed
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        client_cm = self._client_cm
        self._client = None
        self._client_cm = None
        self.tool_specs = ()
        if client_cm is not None:
            await client_cm.__aexit__(None, None, None)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise ToolExecutionError("MCP client is not connected")
        if name not in self._allowed:
            raise ToolExecutionError(f"tool is not allowed for this agent: {name}")
        try:
            result = await self._client.call_tool(
                name,
                arguments,
                read_timeout_seconds=self._config.timeout_seconds,
            )
        except TimeoutError:
            raise ToolExecutionError(f"tool call timed out: {name}") from None
        # This is the process boundary: every SDK/transport failure must become a
        # redacted ToolExecutor error instead of leaking implementation exceptions.
        except Exception as error:  # noqa: BLE001
            raise ToolExecutionError(self._redact(str(error))) from None
        if result.is_error:
            message = "\n".join(
                content.text for content in result.content if getattr(content, "type", None) == "text"
            )
            raise ToolExecutionError(self._redact(message or f"tool failed: {name}"))
        output = result.structured_content
        if not isinstance(output, dict):
            raise ToolExecutionError(f"tool returned a non-object result: {name}")
        return self._redact_value(output)

    def _validate_attachments(self, attachments: Iterable[Path]) -> tuple[Path, ...]:
        validated: list[Path] = []
        for path in attachments:
            candidate = path if path.is_absolute() else self._task_root / path
            resolved = candidate.resolve()
            try:
                resolved.relative_to(self._task_root)
            except ValueError:
                raise ValueError(f"attachment is outside task root: {path}") from None
            if not resolved.is_file():
                raise ValueError(f"attachment does not exist or is not a file: {path}")
            validated.append(resolved)
        return tuple(validated)

    def _minimal_server_env(self) -> dict[str, str]:
        env = {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "PYTHONSAFEPATH": "1",
            "GAIA_TASK_ROOT": str(self._task_root),
            "GAIA_ATTACHMENTS": json.dumps([str(path) for path in self._attachments]),
            "GAIA_MODEL": self._config.model,
            "GAIA_REASONING_EFFORT": self._config.reasoning_effort,
            "GAIA_TIMEOUT_SECONDS": str(self._config.timeout_seconds),
            "OPENAI_API_KEY": self._config.openai_api_key.get_secret_value(),
            "HF_TOKEN": self._config.hf_token.get_secret_value(),
        }
        if os.environ.get("GAIA_MCP_FAKE_BACKEND") == "1":
            env["GAIA_MCP_FAKE_BACKEND"] = "1"
        return env

    def _redact(self, message: str) -> str:
        for secret in (
            self._config.openai_api_key.get_secret_value(),
            self._config.hf_token.get_secret_value(),
        ):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, str):
            return self._redact(value)
        return value


def _strict_function_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict = {
        key: (
            _strict_function_schema(item)
            if isinstance(item, dict)
            else [
                _strict_function_schema(entry) if isinstance(entry, dict) else entry
                for entry in item
            ]
            if isinstance(item, list)
            else item
        )
        for key, item in schema.items()
    }
    properties = strict.get("properties")
    if isinstance(properties, dict):
        strict["additionalProperties"] = False
        strict["required"] = list(properties)
    return strict
