"""Best-effort, privacy-preserving Langfuse observability for GAIA runs.

The evaluation path deliberately depends on this module only through a small adapter.  The
adapter is a no-op when Langfuse is disabled or credentials are incomplete, and every SDK call
is isolated so a telemetry outage cannot alter a prediction or its error category.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from types import TracebackType
from typing import Any, Self

from gaia_agent.models import Usage

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://cloud.langfuse.com"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_SAFE_STATUS_VALUES = frozenset({"ok", "error", "failed", "timeout", "cancelled"})
_SAFE_CRITIC_VERDICTS = frozenset({"approve", "revise", "insufficient_evidence", "not_run"})
_SAFE_ERROR_CATEGORIES = frozenset(
    {
        "timeout",
        "malformed_response",
        "tool_budget_exhausted",
        "loop_detected",
        "critic_rejection",
        "attachment_error",
        "retry_exhausted",
        "no_consensus",
    }
)
_MAX_OUTCOME_COUNT = 1_000_000
_TASK_OUTCOME_KEYS = frozenset(
    {
        "answer_present",
        "evidence_count",
        "critic_verdict",
        "error_category",
        "correct",
        "provider_retry_count",
        "tool_success_count",
        "tool_failure_count",
        "checkpoint_restored",
    }
)

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
        "run_id",
        "attempt_number",
        *_TASK_OUTCOME_KEYS,
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
    input_cost_per_million: Decimal | None = None
    output_cost_per_million: Decimal | None = None

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
            input_cost_per_million=_parse_cost_rate(
                os.environ.get("LANGFUSE_INPUT_COST_PER_MILLION")
            ),
            output_cost_per_million=_parse_cost_rate(
                os.environ.get("LANGFUSE_OUTPUT_COST_PER_MILLION")
            ),
        )


def _parse_cost_rate(value: object) -> Decimal | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return rate if rate.is_finite() and rate >= 0 else None


def _bounded_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if not isinstance(value, int):
        return 0
    return min(_MAX_OUTCOME_COUNT, max(0, value))


@dataclass(frozen=True)
class TaskTraceOutcome:
    """Non-content terminal state recorded for one selected GAIA task."""

    answer_present: bool
    evidence_count: int
    critic_verdict: str
    error_category: str | None
    correct: bool
    provider_retry_count: int = 0
    tool_success_count: int = 0
    tool_failure_count: int = 0
    checkpoint_restored: bool = False

    def as_output(self) -> dict[str, object]:
        return {
            "answer_present": bool(self.answer_present),
            "evidence_count": _bounded_count(self.evidence_count),
            "critic_verdict": (
                self.critic_verdict if self.critic_verdict in _SAFE_CRITIC_VERDICTS else "not_run"
            ),
            "error_category": (
                self.error_category if self.error_category in _SAFE_ERROR_CATEGORIES else None
            ),
            "correct": bool(self.correct),
            "provider_retry_count": _bounded_count(self.provider_retry_count),
            "tool_success_count": _bounded_count(self.tool_success_count),
            "tool_failure_count": _bounded_count(self.tool_failure_count),
            "checkpoint_restored": bool(self.checkpoint_restored),
        }


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
        if key in _TASK_OUTCOME_KEYS:
            sanitized = _sanitize_task_outcome({key: value})[key]
            if sanitized is not None:
                safe[key] = sanitized
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
        if key in _TASK_OUTCOME_KEYS:
            safe[key] = _sanitize_task_outcome(value)[key]
            continue
        elif key not in _SAFE_IO_KEYS:
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


def _sanitize_task_outcome(value: dict[str, object] | object) -> dict[str, object]:
    if isinstance(value, dict):
        raw = value
    else:
        raw = {"answer_present": value}
    return TaskTraceOutcome(
        answer_present=bool(raw.get("answer_present", False)),
        evidence_count=raw.get("evidence_count", 0),
        critic_verdict=str(raw.get("critic_verdict", "not_run")),
        error_category=(
            str(raw["error_category"]) if raw.get("error_category") is not None else None
        ),
        correct=bool(raw.get("correct", False)),
        provider_retry_count=raw.get("provider_retry_count", 0),
        tool_success_count=raw.get("tool_success_count", 0),
        tool_failure_count=raw.get("tool_failure_count", 0),
        checkpoint_restored=bool(raw.get("checkpoint_restored", False)),
    ).as_output()


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

    def __init__(
        self,
        context_manager: Any | None = None,
        *,
        on_end: Callable[[str, dict[str, object], Usage | None, float], None] | None = None,
        cost_details: Callable[[Usage], dict[str, float] | None] | None = None,
    ) -> None:
        self._context_manager = context_manager
        self._observation: Any | None = None
        self._closed = False
        self._pending_summary: dict[str, object] | None = None
        self._on_end = on_end
        self._cost_details = cost_details
        self._started_at = time.monotonic()
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
            cost_details = self._cost_details(usage) if self._cost_details is not None else None
            if cost_details:
                kwargs["cost_details"] = cost_details
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
        if self._on_end is not None:
            self._on_end(final_metadata["status"], final_metadata, usage, time.monotonic() - self._started_at)
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
    def __init__(
        self,
        client: Any | None,
        metadata: dict[str, object],
        *,
        input_cost_per_million: Decimal | None = None,
        output_cost_per_million: Decimal | None = None,
    ) -> None:
        self._client = client
        self._input_cost_per_million = _parse_cost_rate(input_cost_per_million)
        self._output_cost_per_million = _parse_cost_rate(output_cost_per_million)
        self._provider_usage = Usage()
        self._provider_retry_count = 0
        self._tool_success_count = 0
        self._tool_failure_count = 0
        self._provider_latency_seconds = 0.0
        self._tool_latency_seconds = 0.0
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
            return Span(
                context_manager,
                on_end=(
                    None
                    if name == "gaia-task"
                    else lambda status, final_metadata, usage, duration: self._record_stage(
                        name, status, final_metadata, usage, duration
                    )
                ),
                cost_details=self._cost_details,
            )
        except Exception:
            logger.debug("Langfuse observation creation failed", exc_info=True)
            return Span()

    def _cost_details(self, usage: Usage) -> dict[str, float] | None:
        details: dict[str, float] = {}
        if self._input_cost_per_million is not None:
            details["input"] = float(
                Decimal(max(0, usage.input_tokens)) * self._input_cost_per_million / Decimal(1_000_000)
            )
        if self._output_cost_per_million is not None:
            details["output"] = float(
                Decimal(max(0, usage.output_tokens))
                * self._output_cost_per_million
                / Decimal(1_000_000)
            )
        return details or None

    def _record_stage(
        self,
        name: str,
        status: str,
        metadata: dict[str, object],
        usage: Usage | None,
        duration_seconds: float,
    ) -> None:
        if name == "provider-generation":
            if usage is not None:
                self._provider_usage = Usage(
                    input_tokens=self._provider_usage.input_tokens + max(0, usage.input_tokens),
                    output_tokens=self._provider_usage.output_tokens + max(0, usage.output_tokens),
                    reasoning_tokens=self._provider_usage.reasoning_tokens
                    + max(0, usage.reasoning_tokens),
                    total_tokens=self._provider_usage.total_tokens + max(0, usage.total_tokens),
                )
            self._provider_retry_count = _bounded_count(
                self._provider_retry_count + _bounded_count(metadata.get("provider_retry_count", 0))
            )
            self._provider_latency_seconds += max(0.0, duration_seconds)
        elif name == "mcp-tool":
            if status == "ok":
                self._tool_success_count = _bounded_count(self._tool_success_count + 1)
            else:
                self._tool_failure_count = _bounded_count(self._tool_failure_count + 1)
            self._tool_latency_seconds += max(0.0, duration_seconds)

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

    def finish(self, outcome: TaskTraceOutcome) -> None:
        """Finish the root with only the bounded terminal task outcome."""
        if self._closed:
            return
        completed = replace(
            outcome,
            provider_retry_count=_bounded_count(
                outcome.provider_retry_count + self._provider_retry_count
            ),
            tool_success_count=_bounded_count(outcome.tool_success_count + self._tool_success_count),
            tool_failure_count=_bounded_count(outcome.tool_failure_count + self._tool_failure_count),
        )
        output = completed.as_output()
        self._root.end(
            status="error" if output["error_category"] is not None else "ok",
            usage=self._provider_usage,
            output=output,
        )
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
        run_index: int | None = None,
        run_id: str | None = None,
        attempt_number: int | None = None,
        evaluation_mode: str,
    ) -> TaskTrace:
        metadata: dict[str, object] = {
            "task_id": task_id,
            "level": level,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "evaluation_mode": evaluation_mode,
        }
        if run_id is not None:
            metadata["run_id"] = run_id
        if attempt_number is not None:
            metadata["attempt_number"] = attempt_number
        if run_index is not None:
            metadata["run_index"] = run_index
        if self.config.project_name:
            metadata["project_name"] = self.config.project_name
        return TaskTrace(
            self.client,
            metadata,
            input_cost_per_million=self.config.input_cost_per_million,
            output_cost_per_million=self.config.output_cost_per_million,
        )

    @staticmethod
    def current_trace() -> TaskTrace | None:
        return _CURRENT_TRACE.get()

    def flush(self) -> bool:
        if self.client is None:
            return True
        try:
            self.client.flush()
        except Exception:  # noqa: BLE001 - telemetry flush cannot alter evaluation outcome
            logger.warning("Langfuse flush warning")
            return False
        return True
