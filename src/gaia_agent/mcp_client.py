from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import sys
from collections.abc import Iterable, Mapping
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
_PUBLIC_TOOL_TRACE_ARGUMENT_KEYS = {
    "web_search": frozenset({"query"}),
    "read_webpage": frozenset({"url"}),
    "execute_python": frozenset({"instruction", "code", "paths", "attachment_ids", "file_ids"}),
    "read_attachment": frozenset({"path"}),
    "transcribe_audio": frozenset({"path"}),
    "submit_answer": frozenset({"answer", "evidence_ids"}),
}
_PUBLIC_TOOL_TRACE_RESULT_KEYS = {
    "web_search": frozenset({"answer", "sources", "output", "truncated"}),
    "read_webpage": frozenset(
        {
            "url",
            "status_code",
            "content_type",
            "text",
            "truncated",
            "answer",
            "sources",
            "fallback",
            "output",
        }
    ),
    "execute_python": frozenset({"output_text", "outputs", "output", "truncated"}),
    "read_attachment": frozenset(
        {
            "path",
            "kind",
            "text",
            "rows",
            "data",
            "page_count",
            "sheets",
            "media_type",
            "size_bytes",
            "output",
            "truncated",
        }
    ),
    "transcribe_audio": frozenset({"text", "output", "truncated"}),
    "submit_answer": frozenset({"answer", "evidence_ids", "output", "truncated"}),
}
_MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
_ATTACHMENT_READ_CHUNK_BYTES = 1024 * 1024
_SAFE_FAILURE_CATEGORIES = frozenset(
    {
        "tool_error",
        "search_unavailable",
        "timeout",
        "transport",
        "http_status",
        "parse_failure",
        "empty_content",
        "policy_rejection",
        "provider_error",
    }
)
_SAFE_SEARCH_PROVIDERS = frozenset({"openai", "serper", "serpapi"})
_MAX_TRACE_DECISION_COUNT = 1_000_000
_TRACE_META_KEY = "gaia.trace"


class ToolExecutionError(ToolCallError):
    pass


class AttachmentPolicyError(ValueError):
    pass


class _ObservedToolResult(dict[str, Any]):
    def __init__(self, payload: dict[str, Any], trace_metadata: dict[str, Any]) -> None:
        super().__init__(payload)
        self.trace_metadata = trace_metadata


def _public_tool_trace_keys(
    name: str,
    value: dict[str, Any],
    *,
    allowed_by_tool: dict[str, frozenset[str]],
) -> list[str]:
    """Return only the public contract keys registered for this tool."""
    allowed = allowed_by_tool.get(name, frozenset())
    return sorted(key for key in value if key in allowed)


class MCPToolClient:
    def __init__(
        self,
        *,
        task_root: Path,
        config: RunConfig | None = None,
        allowed: set[str] | frozenset[str] | None = None,
        attachments: Iterable[Path] = (),
        observability: Observability | None = None,
        runtime_env: Mapping[str, str] | None = None,
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
        self._runtime_env = dict(runtime_env) if runtime_env is not None else None

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
        argument_keys = _public_tool_trace_keys(
            name,
            arguments,
            allowed_by_tool=_PUBLIC_TOOL_TRACE_ARGUMENT_KEYS,
        )
        trace = self._observability.current_trace() if self._observability else None
        span = (
            trace.span(
                "mcp-tool",
                {"tool": name, "call_number": self._call_count},
                input={
                    "argument_count": len(argument_keys),
                    "argument_keys": argument_keys,
                },
            )
            if trace is not None
            else None
        )
        span_status = "cancelled"
        span_metadata: dict[str, Any] = {}
        span_output: dict[str, object] = {"result_key_count": 0, "result_keys": []}
        try:
            result = await self._call_impl(name, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            span_status = "error"
            span_metadata = {"error_type": type(error).__name__}
            span_metadata.update(_safe_tool_failure_trace(error))
            raise
        else:
            result_keys = _public_tool_trace_keys(
                name,
                result,
                allowed_by_tool=_PUBLIC_TOOL_TRACE_RESULT_KEYS,
            )
            span_status = "ok"
            span_metadata = {"result_key_count": len(result_keys)}
            span_metadata.update(getattr(result, "trace_metadata", {}))
            if result.get("fallback") == "search":
                span_metadata["fallback_used"] = True
            span_output = {
                "result_key_count": len(result_keys),
                "result_keys": result_keys,
            }
            return result
        finally:
            if span is not None:
                span.end(status=span_status, metadata=span_metadata, output=span_output)

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
        trace_metadata = _safe_mcp_trace_metadata(getattr(result, "meta", None))
        if result.is_error:
            message = "\n".join(
                content.text
                for content in result.content
                if getattr(content, "type", None) == "text"
            )
            safe = _safe_failure_payload(message)
            if safe is not None:
                safe.update(
                    {
                        key: value
                        for key, value in trace_metadata.items()
                        if key
                        in {
                            "circuit_open_count",
                            "fallback_attempted",
                            "fallback_used",
                        }
                    }
                )
                raise _tool_execution_error(name, **safe)
            if _is_tool_failure_envelope(message):
                raise ToolExecutionError(f"tool failed: {name}")
            if name == "read_webpage" and (classified := _webpage_failure_payload(message)):
                classified.update(
                    {
                        key: value
                        for key, value in trace_metadata.items()
                        if key in {"circuit_open_count", "fallback_attempted"}
                    }
                )
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
        return _ObservedToolResult(self._redact_value(output), trace_metadata)

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
            "GAIA_SPECIALIST_TIMEOUT_SECONDS": str(self._config.specialist_timeout_seconds),
            "GAIA_PROVIDER_MAX_ATTEMPTS": str(self._config.provider_max_attempts),
            "GAIA_PROVIDER_BACKOFF_BASE": str(self._config.provider_backoff_base),
            "GAIA_SEARCH_CIRCUIT_COOLDOWN": str(self._config.search_circuit_cooldown),
            "GAIA_SEARCH_TIMEOUT_SECONDS": str(self._config.search_timeout_seconds),
            "OPENAI_API_KEY": self._config.openai_api_key.get_secret_value(),
            "HF_TOKEN": self._config.hf_token.get_secret_value(),
        }
        if self._runtime_env is not None:
            env.pop("OPENAI_API_KEY", None)
            env.pop("HF_TOKEN", None)
            for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SERPER_API_KEY", "SERPAPI_API_KEY"):
                if value := self._runtime_env.get(name):
                    env[name] = value
            return env
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
            (self._runtime_env if self._runtime_env is not None else os.environ).get(
                "SERPER_API_KEY", ""
            ),
            (self._runtime_env if self._runtime_env is not None else os.environ).get(
                "SERPAPI_API_KEY", ""
            ),
            (self._runtime_env or {}).get("OPENAI_API_KEY", ""),
        ):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                self._redact(key)
                if self._runtime_env is not None and isinstance(key, str)
                else key: self._redact_value(item)
                for key, item in value.items()
            }
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
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or value.get("error") != "tool_failed"
    ):
        return None
    category = value["category"]
    provider = value["provider"]
    attempt_count = value["attempt_count"]
    http_status = value.get("http_status")
    if (
        not isinstance(category, str)
        or category not in _SAFE_FAILURE_CATEGORIES
        or not isinstance(value["retryable"], bool)
        or provider is not None
        and (not isinstance(provider, str) or provider not in _SAFE_SEARCH_PROVIDERS)
        or not _safe_count(attempt_count, minimum=1)
        or http_status is not None
        and (
            not isinstance(http_status, int)
            or isinstance(http_status, bool)
            or not 100 <= http_status <= 599
        )
    ):
        return None
    safe = {
        "category": category,
        "retryable": value["retryable"],
        "provider": provider,
        "attempt_count": attempt_count,
        "http_status": http_status,
    }
    circuit_open_count = value.get("circuit_open_count")
    if _safe_count(circuit_open_count, minimum=0):
        safe["circuit_open_count"] = circuit_open_count
    fallback_attempted = value.get("fallback_attempted")
    if isinstance(fallback_attempted, bool):
        safe["fallback_attempted"] = fallback_attempted
    fallback_used = value.get("fallback_used")
    if isinstance(fallback_used, bool):
        safe["fallback_used"] = fallback_used
    return safe


def _is_tool_failure_envelope(message: str) -> bool:
    try:
        value = json.loads(message[message.find("{") :])
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict) and value.get("error") == "tool_failed"


def _tool_execution_error(
    name: str,
    *,
    category: str,
    retryable: bool,
    provider: str | None,
    attempt_count: int,
    http_status: int | None = None,
    circuit_open_count: int | None = None,
    fallback_attempted: bool | None = None,
    fallback_used: bool | None = None,
) -> ToolExecutionError:
    error = ToolExecutionError(f"tool failed: {name}")
    error.category = category
    error.retryable = retryable
    error.provider = provider
    error.attempt_count = attempt_count
    error.http_status = http_status
    if circuit_open_count is not None:
        error.circuit_open_count = circuit_open_count
    if fallback_attempted is not None:
        error.fallback_attempted = fallback_attempted
    if fallback_used is not None:
        error.fallback_used = fallback_used
    return error


def _safe_count(value: Any, *, minimum: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= _MAX_TRACE_DECISION_COUNT
    )


def _safe_tool_failure_trace(error: Exception) -> dict[str, Any]:
    """Copy only validated control-flow facts from a typed tool failure."""

    safe: dict[str, Any] = {}
    provider = getattr(error, "provider", None)
    if isinstance(provider, str) and provider in _SAFE_SEARCH_PROVIDERS:
        safe["provider"] = provider
    category = getattr(error, "category", None)
    if isinstance(category, str) and category in _SAFE_FAILURE_CATEGORIES:
        safe["category"] = category
    retryable = getattr(error, "retryable", None)
    if isinstance(retryable, bool):
        safe["retryable"] = retryable
    attempt_count = getattr(error, "attempt_count", None)
    if _safe_count(attempt_count, minimum=1):
        safe["attempt_count"] = attempt_count
    http_status = getattr(error, "http_status", None)
    if (
        isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 100 <= http_status <= 599
    ):
        safe["http_status"] = http_status
    circuit_open_count = getattr(error, "circuit_open_count", None)
    if _safe_count(circuit_open_count, minimum=0):
        safe["circuit_open_count"] = circuit_open_count
    fallback_attempted = getattr(error, "fallback_attempted", None)
    if isinstance(fallback_attempted, bool):
        safe["fallback_attempted"] = fallback_attempted
    fallback_used = getattr(error, "fallback_used", None)
    if isinstance(fallback_used, bool):
        safe["fallback_used"] = fallback_used
    return safe


def _safe_mcp_trace_metadata(meta: Any) -> dict[str, Any]:
    if not isinstance(meta, dict) or not isinstance(meta.get(_TRACE_META_KEY), dict):
        return {}
    value = meta[_TRACE_META_KEY]
    safe: dict[str, Any] = {}
    provider = value.get("provider")
    if isinstance(provider, str) and provider in _SAFE_SEARCH_PROVIDERS:
        safe["provider"] = provider
    category = value.get("category")
    if isinstance(category, str) and category in _SAFE_FAILURE_CATEGORIES:
        safe["category"] = category
    retryable = value.get("retryable")
    if isinstance(retryable, bool):
        safe["retryable"] = retryable
    for key, minimum in (("attempt_count", 1), ("circuit_open_count", 0)):
        item = value.get(key)
        if _safe_count(item, minimum=minimum):
            safe[key] = item
    http_status = value.get("http_status")
    if (
        isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 100 <= http_status <= 599
    ):
        safe["http_status"] = http_status
    for key in ("fallback_attempted", "fallback_used"):
        item = value.get(key)
        if isinstance(item, bool):
            safe[key] = item
    return safe


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
