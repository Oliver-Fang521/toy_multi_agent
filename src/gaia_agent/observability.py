"""Best-effort, privacy-preserving Langfuse observability for GAIA runs.

The evaluation path deliberately depends on this module only through a small adapter.  The
adapter is a no-op when Langfuse is disabled or credentials are incomplete, and every SDK call
is isolated so a telemetry outage cannot alter a prediction or its error category.
"""

from __future__ import annotations

import logging
import math
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from gaia_agent.models import Usage

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://cloud.langfuse.com"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_SAFE_STATUS_VALUES = frozenset({"ok", "error", "failed", "timeout", "cancelled"})

# This is intentionally an allowlist.  In particular, prompt, question, answer, attachment,
# argument, output, and evidence-content fields are never accepted even if a caller supplies them.
_SAFE_METADATA_KEYS = frozenset(
    {
        "task_id",
        "level",
        "model",
        "reasoning_effort",
        "run_index",
        "evaluation_mode",
        "project_name",
        "stage",
        "specialist",
        "step_number",
        "step_count",
        "evidence_count",
        "has_candidate",
        "verdict",
        "checked_count",
        "tool",
        "call_number",
        "category",
        "parsed",
        "refused",
        "status",
        "error_type",
        "http_status",
        "result_key_count",
        "argument_key_count",
        "tool_call_count",
        "provider_attempt_count",
        "provider_retry_count",
        "retry_category",
    }
)
_SAFE_LIST_KEYS = frozenset()
_MAX_STRING_LENGTH = 128
_SAFE_IO_KEYS = frozenset(
    {
        "argument_count",
        "argument_keys",
        "result_key_count",
        "result_keys",
    }
)
_SAFE_SUMMARY_KEYS = frozenset(
    {
        "route",
        "specialist",
        "specialists",
        "step_number",
        "step_count",
        "tool_names",
        "evidence_count",
        "evidence_kinds",
        "verdict",
        "checked_count",
        "decision",
        "repair_attempted",
        "fallback_used",
        "candidate_present",
        "parsed",
        "refused",
        "tool_call_count",
        "provider_attempt_count",
        "provider_retry_count",
        "retry_category",
    }
)
_CURRENT_TRACE: ContextVar[TaskTrace | None] = ContextVar("gaia_langfuse_trace", default=None)


@dataclass(frozen=True)
class ObservabilityConfig:
    """Environment-backed Langfuse settings without exposing credential values."""

    enabled: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    base_url: str = _DEFAULT_BASE_URL
    project_name: str | None = None

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.public_key and self.secret_key)

    @classmethod
    def from_env(cls) -> Self:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or None
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or None
        raw_enabled = (os.environ.get("LANGFUSE_ENABLED") or "").strip().lower()
        if raw_enabled in _TRUE_VALUES:
            enabled = True
        elif raw_enabled in _FALSE_VALUES:
            enabled = False
        else:
            enabled = bool(public_key and secret_key)
        return cls(
            enabled=enabled,
            public_key=public_key,
            secret_key=secret_key,
            base_url=os.environ.get("LANGFUSE_BASE_URL") or _DEFAULT_BASE_URL,
            project_name=os.environ.get("LANGFUSE_PROJECT_NAME") or None,
        )


def _safe_scalar(key: str, value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:_MAX_STRING_LENGTH]
    if key in _SAFE_LIST_KEYS and isinstance(value, (list, tuple)):
        return [item[:_MAX_STRING_LENGTH] for item in value if isinstance(item, str)][:32]
    return None


def _sanitize_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    if not metadata:
        return {}
    safe: dict[str, object] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        if key not in _SAFE_METADATA_KEYS:
            continue
        sanitized = _safe_scalar(key, value)
        if sanitized is not None:
            safe[key] = sanitized
    return safe


def _usage_details(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": max(0, usage.input_tokens),
        "output_tokens": max(0, usage.output_tokens),
        "reasoning_tokens": max(0, usage.reasoning_tokens),
        "total_tokens": max(0, usage.total_tokens),
    }


def _sanitize_io(value: dict[str, object] | None) -> dict[str, object]:
    """Keep only structural input/output summaries safe for trace display."""
    if not value:
        return {}
    safe: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key not in _SAFE_IO_KEYS:
            continue
        if key.endswith("_count") and isinstance(raw_value, int) and not isinstance(raw_value, bool):
            safe[key] = max(0, raw_value)
        elif key.endswith("_keys") and isinstance(raw_value, (list, tuple)):
            safe[key] = [
                item[:_MAX_STRING_LENGTH]
                for item in raw_value
                if isinstance(item, str)
            ][:32]
    return safe


def _sanitize_summary(value: dict[str, object] | None) -> dict[str, object]:
    """Keep a bounded, non-content reasoning summary for trace display."""
    if not value:
        return {}
    safe: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key not in _SAFE_SUMMARY_KEYS:
            continue
        if isinstance(raw_value, bool):
            safe[key] = raw_value
        elif isinstance(raw_value, int) and not isinstance(raw_value, bool):
            safe[key] = max(0, raw_value) if key.endswith(("_count", "_number")) else raw_value
        elif isinstance(raw_value, str):
            safe[key] = raw_value[:_MAX_STRING_LENGTH]
        elif isinstance(raw_value, (list, tuple)):
            safe[key] = [
                item[:_MAX_STRING_LENGTH]
                for item in raw_value
                if isinstance(item, str)
            ][:32]
    return safe


class Span:
    """A safe wrapper around one Langfuse observation or a no-op observation."""

    def __init__(self, context_manager: Any | None = None) -> None:
        self._context_manager = context_manager
        self._observation: Any | None = None
        self._closed = False
        self._pending_summary: dict[str, object] | None = None
        if context_manager is not None:
            try:
                self._observation = context_manager.__enter__()
            except Exception:
                self._context_manager = None
                logger.debug("Langfuse observation could not start", exc_info=True)

    @property
    def enabled(self) -> bool:
        return self._observation is not None

    def update(
        self,
        *,
        metadata: dict[str, object] | None = None,
        usage: Usage | None = None,
        input: dict[str, object] | None = None,
        output: dict[str, object] | None = None,
        summary: dict[str, object] | None = None,
    ) -> None:
        if self._observation is None or self._closed:
            return
        kwargs: dict[str, object] = {}
        safe_metadata = _sanitize_metadata(metadata)
        if safe_metadata:
            kwargs["metadata"] = safe_metadata
        safe_input = _sanitize_io(input)
        if safe_input:
            kwargs["input"] = safe_input
        safe_output = _sanitize_io(output)
        safe_summary = _sanitize_summary(summary)
        if safe_summary:
            safe_output = {**safe_output, "reasoning_summary": safe_summary}
        if safe_output:
            kwargs["output"] = safe_output
        if usage is not None:
            kwargs["usage_details"] = _usage_details(usage)
        if not kwargs:
            return
        try:
            self._observation.update(**kwargs)
        except Exception:
            logger.debug("Langfuse observation update failed", exc_info=True)

    def set_summary(self, summary: dict[str, object] | None) -> None:
        """Publish a safe summary when this context-managed span exits."""
        self._pending_summary = dict(summary or {})

    def end(
        self,
        *,
        status: str,
        metadata: dict[str, object] | None = None,
        usage: Usage | None = None,
        input: dict[str, object] | None = None,
        output: dict[str, object] | None = None,
        summary: dict[str, object] | None = None,
    ) -> None:
        if self._closed:
            return
        final_metadata = dict(metadata or {})
        final_metadata["status"] = status if status in _SAFE_STATUS_VALUES else "error"
        self.update(
            metadata=final_metadata,
            usage=usage,
            input=input,
            output=output,
            summary=self._pending_summary if summary is None else summary,
        )
        self._pending_summary = None
        self._closed = True
        if self._observation is None:
            return
        try:
            if self._context_manager is not None:
                self._context_manager.__exit__(None, None, None)
            else:
                self._observation.end()
        except Exception:
            logger.debug("Langfuse observation close failed", exc_info=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.end(status="error" if exc_type else "ok", summary=self._pending_summary)


class TaskTrace:
    def __init__(self, client: Any | None, metadata: dict[str, object]) -> None:
        self._client = client
        self._root = Span() if client is None else self._start("span", "gaia-task", metadata)
        self._token: Token[TaskTrace | None] = _CURRENT_TRACE.set(self)
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._root.enabled

    def _start(
        self,
        as_type: str,
        name: str,
        metadata: dict[str, object],
        *,
        input: dict[str, object] | None = None,
        output: dict[str, object] | None = None,
    ) -> Span:
        if self._client is None:
            return Span()
        safe_metadata = _sanitize_metadata(metadata)
        try:
            kwargs: dict[str, object] = {"as_type": as_type, "name": name}
            if safe_metadata:
                kwargs["metadata"] = safe_metadata
            safe_input = _sanitize_io(input)
            if safe_input:
                kwargs["input"] = safe_input
            safe_output = _sanitize_io(output)
            if safe_output:
                kwargs["output"] = safe_output
            context_manager = self._client.start_as_current_observation(**kwargs)
            return Span(context_manager)
        except Exception:
            logger.debug("Langfuse observation creation failed", exc_info=True)
            return Span()

    def span(
        self,
        name: str,
        metadata: dict[str, object] | None = None,
        *,
        input: dict[str, object] | None = None,
        output: dict[str, object] | None = None,
    ) -> Span:
        if self._client is None or not self._root.enabled:
            return Span()
        return self._start("span", name, metadata or {}, input=input, output=output)

    def generation(
        self,
        name: str,
        metadata: dict[str, object] | None = None,
        *,
        input: dict[str, object] | None = None,
        output: dict[str, object] | None = None,
    ) -> Span:
        if self._client is None or not self._root.enabled:
            return Span()
        return self._start("generation", name, metadata or {}, input=input, output=output)

    def close(self, *, status: str = "ok", metadata: dict[str, object] | None = None) -> None:
        if self._closed:
            return
        self._root.end(status=status, metadata=metadata)
        self._closed = True
        _CURRENT_TRACE.reset(self._token)


class Observability:
    """Own a Langfuse client while keeping all SDK failures outside the evaluation path."""

    def __init__(self, config: ObservabilityConfig | None = None, *, client: Any | None = None) -> None:
        self.config = config or ObservabilityConfig.from_env()
        self.client: Any | None = None
        if client is not None:
            self.client = client if self.config.active else None
        elif self.config.active:
            try:
                from langfuse import Langfuse

                self.client = Langfuse(
                    public_key=self.config.public_key,
                    secret_key=self.config.secret_key,
                    base_url=self.config.base_url,
                )
            except Exception:
                logger.debug("Langfuse client could not be initialized", exc_info=True)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def start_task(
        self,
        task_id: str,
        *,
        level: int,
        model: str,
        reasoning_effort: str,
        run_index: int,
        evaluation_mode: str,
    ) -> TaskTrace:
        metadata: dict[str, object] = {
            "task_id": task_id,
            "level": level,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "run_index": run_index,
            "evaluation_mode": evaluation_mode,
        }
        if self.config.project_name:
            metadata["project_name"] = self.config.project_name
        return TaskTrace(self.client, metadata)

    @staticmethod
    def current_trace() -> TaskTrace | None:
        return _CURRENT_TRACE.get()

    def flush(self) -> None:
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)
