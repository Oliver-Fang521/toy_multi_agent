from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import sys
from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from gaia_agent.config import RunConfig
from gaia_agent.models import HOSTED_USAGE_METADATA_KEY, Usage
from gaia_agent.observability import Observability
from gaia_agent.providers.base import ToolCallError, ToolSpec

_PUBLIC_TOOLS = frozenset(
    {
        "web_search",
        "read_webpage",
        "execute_python",
        "read_attachment",
        "transcribe_audio",
        "submit_answer",
    }
)
_MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
_ATTACHMENT_READ_CHUNK_BYTES = 1024 * 1024


class ToolExecutionError(ToolCallError):
    pass


class AttachmentPolicyError(ValueError):
    pass


class MCPToolClient:
    def __init__(
        self,
        *,
        task_root: Path,
        config: RunConfig | None = None,
        allowed: set[str] | frozenset[str] | None = None,
        attachments: Iterable[Path] = (),
        observability: Observability | None = None,
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
        self._usage = Usage()
        self._usage_lock = asyncio.Lock()
        self._call_count = 0
        self._observability = observability

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def usage(self) -> Usage:
        return self._usage

    async def __aenter__(self) -> Self:
        if self._client is not None:
            raise RuntimeError("MCP client is already connected")
        self._usage = Usage()
        self._call_count = 0
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
        self._call_count += 1
        trace = self._observability.current_trace() if self._observability else None
        span = (
            trace.span(
                "mcp-tool",
                {"tool": name, "call_number": self._call_count},
                input={
                    "argument_count": len(arguments),
                    "argument_keys": sorted(arguments),
                },
            )
            if trace is not None
            else None
        )
        try:
            result = await self._call_impl(name, arguments)
        except Exception as error:
            if span is not None:
                span.end(
                    status="error",
                    metadata={"error_type": type(error).__name__},
                    output={"result_key_count": 0, "result_keys": []},
                )
            raise
        if span is not None:
            span.end(
                status="ok",
                metadata={"result_key_count": len(result)},
                output={
                    "result_key_count": len(result),
                    "result_keys": sorted(result),
                },
            )
        return result

    async def _call_impl(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
            raise _tool_execution_error(
                name, category="timeout", retryable=True, provider=None, attempt_count=1
            ) from None
        # This is the process boundary: every SDK/transport failure must become a
        # redacted ToolExecutor error instead of leaking implementation exceptions.
        except Exception as error:  # noqa: BLE001
            raise ToolExecutionError(self._redact(str(error))) from None
        if result.is_error:
            message = "\n".join(
                content.text for content in result.content if getattr(content, "type", None) == "text"
            )
            safe = _safe_failure_payload(message)
            if safe is not None:
                raise _tool_execution_error(name, **safe)
            if name == "read_webpage" and (classified := _webpage_failure_payload(message)):
                raise _tool_execution_error(name, **classified)
            raise ToolExecutionError(self._redact(message or f"tool failed: {name}"))
        output = result.structured_content
        if not isinstance(output, dict):
            raise ToolExecutionError(f"tool returned a non-object result: {name}")
        output = dict(output)
        raw_usage = output.pop(HOSTED_USAGE_METADATA_KEY, None)
        if raw_usage is not None:
            usage = _validated_usage(raw_usage)
            async with self._usage_lock:
                self._usage = _add_usage(self._usage, usage)
        return self._redact_value(output)

    async def read_attachment_bytes(self, path: Path | str) -> bytes:
        """Read one registered attachment under a bounded, task-scoped policy."""

        return await asyncio.to_thread(self._read_attachment_bytes, path)

    def _read_attachment_bytes(self, path: Path | str) -> bytes:
        candidate = Path(path)
        candidate = candidate if candidate.is_absolute() else self._task_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise AttachmentPolicyError("attachment does not exist or is unavailable") from None
        try:
            resolved.relative_to(self._task_root)
        except ValueError:
            raise AttachmentPolicyError("attachment is outside task root") from None
        if resolved not in self._attachments:
            raise AttachmentPolicyError("attachment is not registered for this task")
        try:
            before = resolved.stat()
        except OSError:
            raise AttachmentPolicyError("attachment does not exist or is unavailable") from None
        _validate_attachment_stat(before)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(resolved, flags)
        except OSError:
            raise AttachmentPolicyError("attachment changed or is unavailable") from None
        body_failed = False
        try:
            try:
                opened = os.fstat(descriptor)
                _validate_attachment_stat(opened)
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise AttachmentPolicyError("attachment changed during validation")
                chunks: list[bytes] = []
                total = 0
                while total <= _MAX_ATTACHMENT_BYTES:
                    chunk = os.read(
                        descriptor,
                        min(
                            _ATTACHMENT_READ_CHUNK_BYTES,
                            _MAX_ATTACHMENT_BYTES + 1 - total,
                        ),
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                if total > _MAX_ATTACHMENT_BYTES:
                    raise AttachmentPolicyError("attachment exceeds size limit")
                after = os.fstat(descriptor)
                if (
                    after.st_size != opened.st_size
                    or after.st_mtime_ns != opened.st_mtime_ns
                    or total != after.st_size
                ):
                    raise AttachmentPolicyError("attachment changed during read")
                return b"".join(chunks)
            except AttachmentPolicyError:
                body_failed = True
                raise
            except OSError:
                body_failed = True
                raise AttachmentPolicyError("attachment I/O failed") from None
            except BaseException:
                body_failed = True
                raise
        finally:
            try:
                os.close(descriptor)
            except OSError:
                if not body_failed:
                    raise AttachmentPolicyError("attachment I/O failed") from None

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
        if os.environ.get("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
        for name in ("SERPER_API_KEY", "SERPAPI_API_KEY"):
            if value := os.environ.get(name):
                env[name] = value
        return env

    def _redact(self, message: str) -> str:
        for secret in (
            self._config.openai_api_key.get_secret_value(),
            self._config.hf_token.get_secret_value(),
            os.environ.get("SERPER_API_KEY", ""),
            os.environ.get("SERPAPI_API_KEY", ""),
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


def _safe_failure_payload(message: str) -> dict[str, Any] | None:
    try:
        value = json.loads(message[message.find("{") :])
    except (TypeError, ValueError):
        return None
    required = {"error", "category", "retryable", "provider", "attempt_count"}
    if not isinstance(value, dict) or not required.issubset(value) or value.get("error") != "tool_failed":
        return None
    category = value["category"]
    provider = value["provider"]
    attempt_count = value["attempt_count"]
    http_status = value.get("http_status")
    if (
        not isinstance(category, str)
        or not isinstance(value["retryable"], bool)
        or provider is not None and not isinstance(provider, str)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
        or http_status is not None and not isinstance(http_status, int)
    ):
        return None
    return {
        "category": category,
        "retryable": value["retryable"],
        "provider": provider,
        "attempt_count": attempt_count,
        "http_status": http_status,
    }


def _tool_execution_error(
    name: str,
    *,
    category: str,
    retryable: bool,
    provider: str | None,
    attempt_count: int,
    http_status: int | None = None,
) -> ToolExecutionError:
    error = ToolExecutionError(f"tool failed: {name}")
    error.category = category
    error.retryable = retryable
    error.provider = provider
    error.attempt_count = attempt_count
    error.http_status = http_status
    return error


def _webpage_failure_payload(message: str) -> dict[str, Any] | None:
    status = re.search(r"\bHTTP\s+(\d{3})\b", message, flags=re.IGNORECASE)
    if status is not None:
        http_status = int(status.group(1))
        return {
            "category": "http_status",
            "retryable": http_status == 408 or http_status == 429 or http_status >= 500,
            "provider": None,
            "attempt_count": 1,
            "http_status": http_status,
        }
    normalized = message.lower()
    if "timed out" in normalized or "timeout" in normalized:
        return {
            "category": "timeout",
            "retryable": True,
            "provider": None,
            "attempt_count": 1,
            "http_status": None,
        }
    if any(term in normalized for term in ("public host", "credentials", "http(s)")):
        return {
            "category": "policy_rejection",
            "retryable": False,
            "provider": None,
            "attempt_count": 1,
            "http_status": None,
        }
    if "empty" in normalized:
        return {
            "category": "empty_content",
            "retryable": False,
            "provider": None,
            "attempt_count": 1,
            "http_status": None,
        }
    if any(term in normalized for term in ("decode", "parse", "content type")):
        return {
            "category": "parse_failure",
            "retryable": False,
            "provider": None,
            "attempt_count": 1,
            "http_status": None,
        }
    return None


def _validated_usage(value: Any) -> Usage:
    expected = set(Usage.model_fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise ToolExecutionError("invalid hosted usage metadata")
    try:
        usage = Usage.model_validate(value, strict=True)
    except (TypeError, ValueError):
        raise ToolExecutionError("invalid hosted usage metadata") from None
    if any(getattr(usage, field) < 0 for field in expected):
        raise ToolExecutionError("invalid hosted usage metadata")
    return usage


def _add_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )


def _validate_attachment_stat(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise AttachmentPolicyError("attachment is not a regular file")
    if value.st_size > _MAX_ATTACHMENT_BYTES:
        raise AttachmentPolicyError("attachment exceeds size limit")
