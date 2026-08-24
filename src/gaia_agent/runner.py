"""Evaluation coordination, dependency wiring, and independent result verification."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol

from openai import AsyncOpenAI

from gaia_agent.agents import (
    ComputationAgent,
    CriticAgent,
    EvidenceLinkerAgent,
    FileAnalysisAgent,
    LookupAgent,
    PlannerAgent,
    VisualAgent,
)
from gaia_agent.config import RunConfig
from gaia_agent.dataset import load_gaia_validation, sample_stratified
from gaia_agent.mcp_client import MCPToolClient
from gaia_agent.models import (
    AgentPlan,
    AuditEvent,
    GaiaTask,
    PlanStep,
    SpecialistName,
    TaskPrediction,
    TerminalErrorCategory,
    Usage,
)
from gaia_agent.observability import Observability, ObservabilityConfig, TaskTraceOutcome
from gaia_agent.orchestrator import Orchestrator, choose_majority
from gaia_agent.persistence import RunManifest, RunStore
from gaia_agent.providers import (
    DeepSeekProvider,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    OpenAIProvider,
    ToolExecutor,
)
from gaia_agent.reliability import (
    InvocationDeadline,
    RetryPolicy,
    ensure_current_invocation_allowed,
    transient_provider_error_category,
)
from gaia_agent.reporting import (
    EvaluationSummary,
    LevelMetrics,
    summarize,
    verify_summary,
    write_reports,
)
from gaia_agent.research_notebook import ResearchNotebook
from gaia_agent.scoring import question_scorer

_PUBLIC_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
_VOTE_NAMESPACE_PATTERN = re.compile(r"vote-[0-9]{4,}\Z")
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "InternalServerError",
        "PoolTimeout",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "WriteError",
        "WriteTimeout",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "attachment",
        "attachment_content",
        "attachment_path",
        "authorization",
        "credential",
        "final answer",
        "final_answer",
        "hidden_reasoning",
        "instructions",
        "password",
        "prompt",
        "question",
        "reference_answer",
        "secret",
        "token",
    }
)
_MAX_AGENT_TOOL_CALLS = 6
_RECOVERY_BACKOFF_SECONDS = 0.05
_EARLY_TRANSIENT_RATIO = 0.5
_CANCELLATION_CLEANUP_SECONDS = 0.005
_MAX_DETACHED_ATTEMPTS = 64
_DETACHED_ATTEMPTS: set[asyncio.Task[Any]] = set()
_DETACHED_ATTEMPT_RESERVATIONS = 0
_DETACHED_ATTEMPT_OVERFLOW_COUNT = 0
_DETACHED_ATTEMPT_LOCK = Lock()
_PRIVATE_RUNTIME_AUDIT_EVENT_TYPES = frozenset(
    {
        "checkpoint_restored",
        "checkpoint_saved",
        "evidence_repair_failed",
        "evidence_repair_completed",
        "evidence_repair_started",
        "stage_retry_exhausted",
    }
)


def _private_run_root(output_dir: Path) -> Path:
    absolute = output_dir.absolute()
    return absolute.parent / f".{absolute.name}.private"


def _private_vote_root(output_dir: Path, vote_namespace: str) -> Path:
    if _VOTE_NAMESPACE_PATTERN.fullmatch(vote_namespace) is None:
        raise ValueError("vote namespace must be an internal stable identifier")
    return _private_run_root(output_dir) / vote_namespace


class TaskOrchestrator(Protocol):
    audit_events: Sequence[AuditEvent]

    async def run(self, task: GaiaTask) -> TaskPrediction: ...


DatasetLoader = Callable[[RunConfig], tuple[list[GaiaTask], str]]
OrchestratorFactory = (
    Callable[[GaiaTask, str], TaskOrchestrator]
    | Callable[[GaiaTask], TaskOrchestrator]
)
Sleep = Callable[[float], Awaitable[None]]
ProviderFactory = Callable[[RunConfig], "ProviderRuntime"]
RunOpened = Callable[[RunManifest], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class ProviderRuntime:
    """Injected provider composition and its explicit persisted run identity."""

    provider: ModelProvider
    provider_name: str
    evaluation_mode: Literal["production", "test"]
    fake_mcp_backend: bool = False


class _DeadlineGuardedProvider:
    """Reject provider operations that begin after the bound vote closes."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        self.supports_vision = bool(getattr(provider, "supports_vision", False))

    async def complete(
        self,
        request: ModelRequest,
        executor: ToolExecutor | None = None,
    ) -> ModelResponse:
        ensure_current_invocation_allowed()
        return await self._provider.complete(request, executor)


@dataclass(frozen=True)
class _TaskRun:
    prediction: TaskPrediction
    audit_events: tuple[AuditEvent, ...]
    plan: AgentPlan | None


class EvaluationRunner:
    """Run a selected GAIA sample with bounded attempts and atomic checkpoints."""

    def __init__(
        self,
        *,
        config: RunConfig,
        dataset_loader: DatasetLoader,
        orchestrator_factory: OrchestratorFactory,
        sleep: Sleep = asyncio.sleep,
        provider_name: str = "injected",
        evaluation_mode: Literal["production", "test"] = "test",
        observability: Observability | None = None,
        run_opened: RunOpened | None = None,
        clock: Clock = time.monotonic,
        offload_orchestrator_factory: bool = False,
    ) -> None:
        self._config = config
        self._dataset_loader = dataset_loader
        self._orchestrator_factory = orchestrator_factory
        self._factory_accepts_vote_namespace = _accepts_vote_namespace(
            orchestrator_factory
        )
        self._sleep = sleep
        self._provider_name = provider_name
        self._evaluation_mode = evaluation_mode
        self._observability = observability or Observability()
        self._run_opened = run_opened
        self._clock = clock
        self._offload_orchestrator_factory = offload_orchestrator_factory
        self._last_flush_ok: bool | None = None

    @property
    def langfuse_enabled(self) -> bool:
        return self._observability.enabled

    @property
    def last_flush_ok(self) -> bool | None:
        return self._last_flush_ok

    async def run(self) -> EvaluationSummary:
        try:
            return await self._run()
        finally:
            self._last_flush_ok = self._observability.flush()

    async def _run(self) -> EvaluationSummary:
        validate_config(self._config)
        tasks, revision = self._dataset_loader(self._config)
        selected = sample_stratified(tasks, self._config.sample_size, self._config.seed)
        store = _open_or_create_store(
            self._config,
            selected,
            revision,
            provider_name=self._provider_name,
            evaluation_mode=self._evaluation_mode,
        )
        if self._run_opened is not None:
            self._run_opened(store.manifest)

        completed_predictions = {prediction.task_id: prediction for prediction in store.load_predictions()}
        for task in selected:
            started = time.monotonic()
            persisted_prediction = completed_predictions.get(task.task_id)
            trace = self._observability.start_task(
                task.task_id,
                level=task.level,
                model=self._config.model,
                reasoning_effort=self._config.reasoning_effort,
                run_id=store.manifest.run_id,
                attempt_number=(
                    persisted_prediction.attempt_number if persisted_prediction is not None else 1
                ),
                evaluation_mode=self._evaluation_mode,
            )
            if persisted_prediction is not None:
                trace.finish(_task_trace_outcome(persisted_prediction, checkpoint_restored=True))
                continue
            task_run: _TaskRun | None = None
            prediction: TaskPrediction | None = None
            trace_failed = False
            try:
                task_run = await self._run_task(task)
                prediction = task_run.prediction
                try:
                    correct = prediction.error_category is None and question_scorer(
                        prediction.answer, task.final_answer
                    )
                except Exception:  # noqa: BLE001 - scoring is isolated per selected task
                    prediction = prediction.model_copy(
                        update={
                            "correct": False,
                            "error_category": TerminalErrorCategory.MALFORMED_RESPONSE,
                        }
                    )
                else:
                    prediction = prediction.model_copy(update={"correct": correct})
                prediction = prediction.model_copy(
                    update={"duration_seconds": time.monotonic() - started}
                )
                _persist_task_artifacts(store, task.task_id, task_run)
                # A plain append deliberately invokes Task 7's repair path for a
                # parseable incomplete or corrupt record on resume.
                store.append(prediction)
            except Exception:
                trace_failed = True
                raise
            finally:
                terminal = (
                    _failure_prediction(task, TerminalErrorCategory.RETRY_EXHAUSTED)
                    if trace_failed
                    else prediction
                ) or _failure_prediction(
                    task, TerminalErrorCategory.RETRY_EXHAUSTED
                )
                trace.finish(
                    _task_trace_outcome(
                        terminal,
                        checkpoint_restored=bool(
                            task_run
                            and any(
                                event.event_type == "tool_call_restored"
                                for event in task_run.audit_events
                            )
                        ),
                        audit_events=task_run.audit_events if task_run is not None else (),
                    )
                )

        predictions = store.load_predictions()
        expected_ids = tuple(task.task_id for task in selected)
        if tuple(prediction.task_id for prediction in predictions) != expected_ids:
            raise ValueError("evaluation did not produce one terminal prediction per selected task")
        summary = summarize(predictions)
        write_reports(store, summary)
        return summary

    async def _run_task(self, task: GaiaTask) -> _TaskRun:
        predictions: list[TaskPrediction] = []
        audit_events: list[AuditEvent] = []
        first_plan: AgentPlan | None = None
        for run_index in range(self._config.votes):
            task_trace = self._observability.current_trace()
            trace = task_trace.span("vote", {"run_index": run_index}) if task_trace else None
            vote: _TaskRun | None = None
            try:
                vote = await self._run_vote(
                    task,
                    vote_namespace=f"vote-{run_index + 1:04d}",
                )
            finally:
                status = "ok"
                close_metadata: dict[str, object] | None = None
                if vote is None or vote.prediction.error_category is not None:
                    status = "error"
                    if vote is not None and vote.prediction.error_category is not None:
                        close_metadata = {"category": vote.prediction.error_category.value}
                if trace is not None:
                    trace.end(status=status, metadata=close_metadata)
            assert vote is not None
            predictions.append(vote.prediction.model_copy(update={"run_index": run_index}))
            audit_events.extend(vote.audit_events)
            first_plan = first_plan or vote.plan
            selected = choose_majority(predictions, self._config.votes)
            if selected.error_category != TerminalErrorCategory.NO_CONSENSUS:
                break

        selected = choose_majority(predictions, self._config.votes)
        usage = _sum_usage(predictions)
        selected = selected.model_copy(
            update={
                "usage": usage,
                "attempt_number": max(item.attempt_number for item in predictions),
                "recovery_count": sum(item.recovery_count for item in predictions),
                "recovery_succeeded": any(item.recovery_succeeded for item in predictions),
                "provider_attempt_count": sum(
                    item.provider_attempt_count for item in predictions
                ),
                "retry_reason": next(
                    (item.retry_reason for item in predictions if item.retry_reason is not None),
                    None,
                ),
                "retry_category": next(
                    (item.retry_category for item in predictions if item.retry_category is not None),
                    None,
                ),
            }
        )
        return _TaskRun(
            prediction=selected,
            audit_events=tuple(audit_events),
            plan=first_plan,
        )

    async def _run_vote(
        self,
        task: GaiaTask,
        *,
        vote_namespace: str = "vote-0001",
    ) -> _TaskRun:
        vote_started = self._clock()
        vote_deadline = vote_started + self._config.timeout_seconds
        invocation_deadline = InvocationDeadline(vote_deadline, self._clock)
        try:
            return await self._run_vote_with_invocation_deadline(
                task,
                vote_namespace=vote_namespace,
                vote_started=vote_started,
                vote_deadline=vote_deadline,
                invocation_deadline=invocation_deadline,
            )
        finally:
            invocation_deadline.close()

    async def _run_vote_with_invocation_deadline(
        self,
        task: GaiaTask,
        *,
        vote_namespace: str,
        vote_started: float,
        vote_deadline: float,
        invocation_deadline: InvocationDeadline,
    ) -> _TaskRun:
        audit_events: list[AuditEvent] = []
        first_plan: AgentPlan | None = None
        prior_predictions: list[TaskPrediction] = []
        retry_reason: str | None = None
        retry_category: str | None = None
        provider_attempt_count = 0
        recovery_attempted = False
        checkpoint_restored_this_attempt = False
        authenticated_usage_this_attempt: Usage | None = None
        for attempt in range(2):
            remaining_seconds = vote_deadline - self._clock()
            if remaining_seconds <= 0:
                prediction = _failure_prediction(task, TerminalErrorCategory.TIMEOUT)
                actual_attempt_number = max(1, attempt)
                break
            orchestrator: TaskOrchestrator | None = None
            actual_attempt_number = attempt + 1
            try:
                orchestrator = await self._make_orchestrator_with_deadline(
                    task,
                    vote_namespace,
                    remaining_seconds,
                )
                _set_orchestrator_invocation_deadline(
                    orchestrator,
                    invocation_deadline,
                )
                remaining_seconds = vote_deadline - self._clock()
                if remaining_seconds <= 0:
                    raise TimeoutError
                prediction = await _run_orchestrator_with_deadline(
                    orchestrator,
                    task,
                    remaining_seconds,
                    invocation_deadline,
                )
                _validate_prediction_identity(prediction, task)
            except Exception as error:  # noqa: BLE001 - task isolation boundary
                events = tuple(getattr(orchestrator, "audit_events", ()))
                audit_events.extend(events)
                first_plan = first_plan or _plan_from_audit(events)
                is_transient = _is_transient_failure(error)
                category = (
                    TerminalErrorCategory.TIMEOUT
                    if isinstance(error, TimeoutError)
                    else TerminalErrorCategory.RETRY_EXHAUSTED
                    if is_transient
                    else _category_for_exception(error)
                )
                prediction = _failure_prediction(task, category)
                if is_transient:
                    prediction = prediction.model_copy(
                        update={
                            "retry_reason": (
                                "timeout" if isinstance(error, TimeoutError) else "task_transient"
                            ),
                            "retry_category": _transient_failure_category(error),
                        }
                    )
            else:
                events = tuple(getattr(orchestrator, "audit_events", ()))
                audit_events.extend(events)
                first_plan = first_plan or _plan_from_audit(events)
            checkpoint_restored_this_attempt = any(
                event.event_type == "checkpoint_restored" for event in events
            )
            authenticated_usage_this_attempt = _authenticated_checkpoint_usage(
                orchestrator
            )
            provider_attempt_count += prediction.provider_attempt_count
            if self._clock() >= vote_deadline and prediction.error_category is None:
                prediction = _failure_prediction(
                    task,
                    TerminalErrorCategory.TIMEOUT,
                ).model_copy(update={"usage": prediction.usage})
            is_transient = _is_transient_prediction(prediction)
            if retry_reason is None and is_transient:
                retry_reason = prediction.retry_reason or "timeout"
                retry_category = prediction.retry_category or (
                    "TimeoutError"
                    if prediction.error_category == TerminalErrorCategory.TIMEOUT
                    else "provider_transient"
                )
                if attempt == 0:
                    elapsed_seconds = self._clock() - vote_started
                    remaining_seconds = vote_deadline - self._clock()
                    stable_checkpoint_present = _has_stable_checkpoint(events)
                    specialist_timeout_is_terminal = (
                        _specialist_timeout_without_stable_candidate(events)
                    )
                    critic_failure_is_terminal = _critic_failure_is_terminal(
                        events
                    )
                    minimum_recovery_seconds = self._minimum_recovery_seconds()
                    is_early_transient = (
                        not stable_checkpoint_present
                        and elapsed_seconds
                        <= self._config.timeout_seconds * _EARLY_TRANSIENT_RATIO
                    )
                    can_resume_checkpoint = (
                        stable_checkpoint_present
                        and not specialist_timeout_is_terminal
                        and not critic_failure_is_terminal
                        and remaining_seconds > minimum_recovery_seconds
                    )
                    can_retry_fresh = (
                        is_early_transient
                        and remaining_seconds > minimum_recovery_seconds
                    )
                    if can_resume_checkpoint or can_retry_fresh:
                        await self._sleep(
                            min(
                                _RECOVERY_BACKOFF_SECONDS,
                                max(
                                    0.0,
                                    remaining_seconds - minimum_recovery_seconds,
                                ),
                            )
                        )
                        if (
                            vote_deadline - self._clock()
                            > minimum_recovery_seconds
                        ):
                            prior_predictions.append(prediction)
                            recovery_attempted = True
                            continue

            recovery_succeeded = recovery_attempted and prediction.error_category is None
            if recovery_attempted:
                audit_events.append(
                    _task_recovery_event(
                        sequence=len(audit_events) + 1,
                        attempt_number=actual_attempt_number,
                        retry_reason=retry_reason,
                        retry_category=retry_category or "provider_transient",
                        recovery_succeeded=recovery_succeeded,
                    )
                )
            prediction = prediction.model_copy(
                update={
                    "usage": (
                        _cumulative_recovery_usage(
                            prior_predictions,
                            prediction,
                            authenticated_usage_this_attempt,
                        )
                        if checkpoint_restored_this_attempt
                        else _sum_usage([*prior_predictions, prediction])
                    ),
                    "attempt_number": actual_attempt_number,
                    "recovery_count": 1 if recovery_attempted else 0,
                    "recovery_succeeded": recovery_succeeded,
                    "retry_reason": retry_reason,
                    "retry_category": retry_category,
                    "provider_attempt_count": provider_attempt_count,
                }
            )
            return _TaskRun(prediction, tuple(audit_events), first_plan)
        prediction = prediction.model_copy(
            update={
                "usage": (
                    _cumulative_recovery_usage(
                        prior_predictions,
                        prediction,
                        authenticated_usage_this_attempt,
                    )
                    if checkpoint_restored_this_attempt
                    else _sum_usage([*prior_predictions, prediction])
                ),
                "attempt_number": actual_attempt_number,
                "recovery_count": 1 if recovery_attempted else 0,
                "recovery_succeeded": False,
                "retry_reason": retry_reason,
                "retry_category": retry_category,
                "provider_attempt_count": provider_attempt_count,
            }
        )
        return _TaskRun(prediction, tuple(audit_events), first_plan)

    def _make_orchestrator(
        self,
        task: GaiaTask,
        vote_namespace: str,
    ) -> TaskOrchestrator:
        if self._factory_accepts_vote_namespace:
            namespaced_factory = self._orchestrator_factory
            return namespaced_factory(task, vote_namespace)  # type: ignore[call-arg]
        legacy_factory = self._orchestrator_factory
        return legacy_factory(task)  # type: ignore[call-arg]

    async def _make_orchestrator_with_deadline(
        self,
        task: GaiaTask,
        vote_namespace: str,
        remaining_seconds: float,
    ) -> TaskOrchestrator:
        if not self._offload_orchestrator_factory:
            return self._make_orchestrator(task, vote_namespace)
        if not _reserve_detached_attempt():
            raise TimeoutError
        try:
            creation = asyncio.create_task(
                asyncio.to_thread(self._make_orchestrator, task, vote_namespace)
            )
        except BaseException:
            _release_detached_attempt_reservation()
            raise
        try:
            done, _ = await asyncio.wait({creation}, timeout=remaining_seconds)
        except asyncio.CancelledError:
            _observe_detached_attempt(creation)
            raise
        if creation in done:
            _release_detached_attempt_reservation()
            try:
                return creation.result()
            except asyncio.CancelledError as error:
                raise TimeoutError from error
        _observe_detached_attempt(creation)
        raise TimeoutError

    def _minimum_recovery_seconds(self) -> float:
        return max(
            0.001,
            min(
                self._config.critic_timeout_seconds,
                self._config.timeout_seconds * 0.1,
            ),
        )


def _accepts_vote_namespace(factory: OrchestratorFactory) -> bool:
    """Keep legacy injected factories working while production receives vote scope."""
    try:
        signature(factory).bind(object(), "vote-0001")
    except (TypeError, ValueError):
        return False
    return True


def _set_orchestrator_invocation_deadline(
    orchestrator: TaskOrchestrator,
    deadline: InvocationDeadline,
) -> None:
    setter = getattr(orchestrator, "set_invocation_deadline", None)
    if callable(setter):
        setter(deadline)


def _authenticated_checkpoint_usage(
    orchestrator: TaskOrchestrator | None,
) -> Usage | None:
    if orchestrator is None:
        return None
    loader = getattr(orchestrator, "authenticated_checkpoint_usage", None)
    if not callable(loader):
        return None
    try:
        usage = loader()
    except (OSError, TypeError, ValueError):
        return None
    return usage if isinstance(usage, Usage) else None


async def _run_orchestrator_with_deadline(
    orchestrator: TaskOrchestrator,
    task: GaiaTask,
    remaining_seconds: float,
    invocation_deadline: InvocationDeadline,
) -> TaskPrediction:
    if not _reserve_detached_attempt():
        raise TimeoutError
    try:
        attempt = asyncio.create_task(orchestrator.run(task))
    except BaseException:
        _release_detached_attempt_reservation()
        raise
    try:
        done, _ = await asyncio.wait({attempt}, timeout=remaining_seconds)
    except asyncio.CancelledError:
        invocation_deadline.close()
        await _cancel_and_observe(attempt)
        raise
    if attempt in done:
        _release_detached_attempt_reservation()
        try:
            return attempt.result()
        except asyncio.CancelledError as error:
            raise TimeoutError from error

    invocation_deadline.close()
    await _cancel_and_observe(attempt)
    raise TimeoutError


async def _cancel_and_observe(attempt: asyncio.Task[Any]) -> None:
    _observe_detached_attempt(attempt)
    attempt.cancel()
    await asyncio.wait(
        {attempt},
        timeout=_CANCELLATION_CLEANUP_SECONDS,
    )


def _observe_detached_attempt(attempt: asyncio.Task[Any]) -> None:
    _track_detached_attempt(attempt)
    attempt.add_done_callback(_consume_detached_attempt)


def _consume_detached_attempt(attempt: asyncio.Task[Any]) -> None:
    with _DETACHED_ATTEMPT_LOCK:
        _DETACHED_ATTEMPTS.discard(attempt)
    if attempt.cancelled():
        return
    try:
        attempt.exception()
    except asyncio.CancelledError:
        pass


def _reserve_detached_attempt() -> bool:
    global _DETACHED_ATTEMPT_OVERFLOW_COUNT
    global _DETACHED_ATTEMPT_RESERVATIONS
    with _DETACHED_ATTEMPT_LOCK:
        if (
            len(_DETACHED_ATTEMPTS) + _DETACHED_ATTEMPT_RESERVATIONS
            >= _MAX_DETACHED_ATTEMPTS
        ):
            _DETACHED_ATTEMPT_OVERFLOW_COUNT += 1
            return False
        _DETACHED_ATTEMPT_RESERVATIONS += 1
        return True


def _release_detached_attempt_reservation() -> None:
    global _DETACHED_ATTEMPT_RESERVATIONS
    with _DETACHED_ATTEMPT_LOCK:
        _DETACHED_ATTEMPT_RESERVATIONS = max(
            0,
            _DETACHED_ATTEMPT_RESERVATIONS - 1,
        )


def _track_detached_attempt(attempt: asyncio.Task[Any]) -> None:
    global _DETACHED_ATTEMPT_RESERVATIONS
    with _DETACHED_ATTEMPT_LOCK:
        _DETACHED_ATTEMPT_RESERVATIONS = max(
            0,
            _DETACHED_ATTEMPT_RESERVATIONS - 1,
        )
        _DETACHED_ATTEMPTS.add(attempt)


def _detached_attempt_diagnostics() -> dict[str, int]:
    with _DETACHED_ATTEMPT_LOCK:
        return {
            "tracked": len(_DETACHED_ATTEMPTS),
            "reserved": _DETACHED_ATTEMPT_RESERVATIONS,
            "capacity": _MAX_DETACHED_ATTEMPTS,
            "overflow_count": _DETACHED_ATTEMPT_OVERFLOW_COUNT,
        }


def validate_config(config: RunConfig) -> None:
    """Validate cost and path-affecting values without exposing their contents."""

    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", config.openai_api_key.get_secret_value()),
            ("HF_TOKEN", config.hf_token.get_secret_value()),
        )
        if not value
    ]
    if missing:
        raise ValueError("missing required environment variables: " + ", ".join(missing))
    if not _PUBLIC_IDENTIFIER_PATTERN.fullmatch(config.model):
        raise ValueError("model must be a non-empty bounded public identifier")
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    output_dir = config.output_dir.absolute()
    if output_dir.exists() and (output_dir.is_symlink() or not output_dir.is_dir()):
        raise ValueError("output_dir must be a safe directory")


def build_runner(
    config: RunConfig,
    *,
    dataset_loader: DatasetLoader | None = None,
    provider_factory: ProviderFactory | None = None,
) -> EvaluationRunner:
    """Compose provider-neutral agents and orchestrators with the real stdio MCP client."""

    injected_dataset = dataset_loader is not None
    injected_provider = provider_factory is not None
    if injected_dataset != injected_provider:
        raise ValueError("dataset_loader and provider_factory must be injected together")

    validate_config(config)
    if not injected_dataset and (
        config.model != "gpt-5.6-luna" or config.reasoning_effort != "max"
    ):
        raise ValueError("production evaluation requires model=gpt-5.6-luna and reasoning_effort=max")
    observability_config = ObservabilityConfig.from_env()
    observability = Observability(
        replace(
            observability_config,
            input_cost_per_million=(
                config.langfuse_input_cost_per_million
                if config.langfuse_input_cost_per_million is not None
                else observability_config.input_cost_per_million
            ),
            output_cost_per_million=(
                config.langfuse_output_cost_per_million
                if config.langfuse_output_cost_per_million is not None
                else observability_config.output_cost_per_million
            ),
        )
    )
    if not injected_dataset:
        dataset_loader = lambda active: load_gaia_validation(
            active.hf_token.get_secret_value()
        )
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = AsyncOpenAI(
            api_key=config.openai_api_key.get_secret_value(),
            max_retries=0,
        )
        if base_url and "api.openai.com" not in base_url:
            provider: ModelProvider = DeepSeekProvider(
                client,
                model=config.model,
                observability=observability,
                retry_policy=RetryPolicy(
                    max_attempts=config.provider_max_attempts,
                    base_delay_seconds=config.provider_backoff_base,
                ),
            )
            provider_name = "deepseek"
        else:
            provider = OpenAIProvider(
                client,
                model=config.model,
                use_previous_response_id=base_url is None,
                observability=observability,
                retry_policy=RetryPolicy(
                    max_attempts=config.provider_max_attempts,
                    base_delay_seconds=config.provider_backoff_base,
                ),
            )
            provider_name = "openai"
        runtime = ProviderRuntime(
            provider=provider,
            provider_name=provider_name,
            evaluation_mode="production",
        )
    else:
        assert dataset_loader is not None
        assert provider_factory is not None
        runtime = provider_factory(config)
        if runtime.evaluation_mode != "test" or runtime.provider_name == "openai":
            raise ValueError("injected composition must declare a non-production test identity")
    if not _PUBLIC_IDENTIFIER_PATTERN.fullmatch(runtime.provider_name):
        raise ValueError("provider_name must be a bounded public identifier")
    if runtime.fake_mcp_backend and runtime.evaluation_mode != "test":
        raise ValueError("fake MCP backend requires an injected test runtime")
    provider = _DeadlineGuardedProvider(runtime.provider)
    agent_tool_calls = min(config.max_tool_calls, _MAX_AGENT_TOOL_CALLS)
    planner = PlannerAgent(
        provider,
        max_steps=config.max_steps,
        reasoning_effort=config.reasoning_effort,
    )
    specialists = {
        SpecialistName.LOOKUP: LookupAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=agent_tool_calls,
            max_context_bytes=config.stage_context_max_bytes,
        ),
        SpecialistName.COMPUTATION: ComputationAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=agent_tool_calls,
            max_context_bytes=config.stage_context_max_bytes,
        ),
        SpecialistName.FILE: FileAnalysisAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=agent_tool_calls,
            max_context_bytes=config.stage_context_max_bytes,
        ),
        SpecialistName.VISUAL: VisualAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=agent_tool_calls,
            max_context_bytes=config.stage_context_max_bytes,
        ),
    }
    critic = CriticAgent(
        provider,
        reasoning_effort=config.reasoning_effort,
        max_tool_calls=agent_tool_calls,
        max_context_bytes=config.stage_context_max_bytes,
        quality_policy=config.quality_policy,
    )
    evidence_linker = EvidenceLinkerAgent(
        provider,
        reasoning_effort=config.reasoning_effort,
        max_context_bytes=config.stage_context_max_bytes,
    )
    notebook_generation: str | None = None

    def bind_notebook_generation(manifest: RunManifest) -> None:
        nonlocal notebook_generation
        notebook_generation = manifest.started_at.astimezone(UTC).isoformat()

    def orchestrator_factory(
        task: GaiaTask,
        vote_namespace: str = "vote-0001",
    ) -> Orchestrator:
        notebook = (
            ResearchNotebook(
                _private_vote_root(config.output_dir, vote_namespace),
                task.task_id,
                max_entry_bytes=config.private_notebook_max_bytes,
                run_generation=notebook_generation,
            )
            if notebook_generation is not None
            else None
        )
        return Orchestrator(
            planner=planner,
            specialists=specialists,
            critic=critic,
            tool_client_factory=lambda active_task: _stdio_client(
                active_task,
                config,
                fake_backend=runtime.fake_mcp_backend,
                observability=observability,
            ),
            config=config,
            observability=observability,
            notebook=notebook,
            evidence_linker=evidence_linker,
        )

    return EvaluationRunner(
        config=config,
        dataset_loader=dataset_loader,
        orchestrator_factory=orchestrator_factory,
        provider_name=runtime.provider_name,
        evaluation_mode=runtime.evaluation_mode,
        observability=observability,
        run_opened=bind_notebook_generation,
        offload_orchestrator_factory=(runtime.evaluation_mode == "production"),
    )


def verify_run_directory(root: Path) -> EvaluationSummary:
    """Reload and independently verify every public run artifact."""

    root = root.absolute()
    required = tuple(root / name for name in ("manifest.json", "predictions.jsonl", "summary.json", "report.md"))
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ValueError("run is incomplete: required result artifacts are missing")

    manifest = _load_public_manifest(root / "manifest.json")
    if manifest.sample_size < 3 or manifest.sample_size % 3:
        raise ValueError("manifest sample size is not evenly stratified")
    expected_per_level = manifest.sample_size // 3
    level_counts = {
        level: sum(expected == level for expected in manifest.expected_levels.values())
        for level in (1, 2, 3)
    }
    if level_counts != {1: expected_per_level, 2: expected_per_level, 3: expected_per_level}:
        raise ValueError("manifest selected levels are not balanced")

    store = RunStore(root, manifest)
    store.verify_integrity()
    for task_id in manifest.selected_task_ids:
        audit_path = root / "tasks" / task_id / "audit.jsonl"
        if audit_path.exists():
            _next_audit_sequence(audit_path)
    predictions = store.load_predictions()
    prediction_ids = tuple(prediction.task_id for prediction in predictions)
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("predictions contain duplicate task IDs")
    if prediction_ids != manifest.selected_task_ids:
        raise ValueError("run is incomplete or predictions do not match selected task IDs")
    if any(prediction.level != manifest.expected_level(prediction.task_id) for prediction in predictions):
        raise ValueError("prediction levels do not match the manifest")

    raw_summary = _load_json_object(root / "summary.json")
    _require_exact_keys(raw_summary, EvaluationSummary.model_fields, "summary")
    _require_exact_keys(raw_summary.get("usage"), Usage.model_fields, "summary usage")
    by_level = raw_summary.get("by_level")
    if not isinstance(by_level, Mapping) or set(by_level) != {"1", "2", "3"}:
        raise ValueError("summary levels do not match the public schema")
    for value in by_level.values():
        _require_exact_keys(value, LevelMetrics.model_fields, "level summary")
    _reject_forbidden_keys(raw_summary)
    summary = EvaluationSummary.model_validate(raw_summary)
    verify_summary(predictions, summary)
    return summary


def _open_or_create_store(
    config: RunConfig,
    selected: Sequence[GaiaTask],
    revision: str,
    *,
    provider_name: str,
    evaluation_mode: Literal["production", "test"],
) -> RunStore:
    manifest_path = config.output_dir.absolute() / "manifest.json"
    if manifest_path.exists():
        if not config.resume:
            raise ValueError("output directory already contains a run; pass --resume")
        persisted = _load_public_manifest(manifest_path)
        manifest = _manifest_for(
            config,
            selected,
            revision,
            provider_name=provider_name,
            evaluation_mode=evaluation_mode,
            started_at=persisted.started_at,
        )
        return RunStore.open(config.output_dir, manifest)
    manifest = _manifest_for(
        config,
        selected,
        revision,
        provider_name=provider_name,
        evaluation_mode=evaluation_mode,
    )
    return RunStore.create(config.output_dir, manifest)


def _manifest_for(
    config: RunConfig,
    selected: Sequence[GaiaTask],
    revision: str,
    *,
    provider_name: str,
    evaluation_mode: Literal["production", "test"],
    started_at: datetime | None = None,
) -> RunManifest:
    return RunManifest(
        selected_task_ids=tuple(task.task_id for task in selected),
        expected_levels={task.task_id: task.level for task in selected},
        sample_size=config.sample_size,
        seed=config.seed,
        run_id=f"gaia-seed{config.seed}-n{config.sample_size}",
        model=config.model,
        provider_name=provider_name,
        evaluation_mode=evaluation_mode,
        reasoning_effort=config.reasoning_effort,
        quality_policy=config.quality_policy,
        dataset_revision=revision,
        limits={
            "max_steps": config.max_steps,
            "max_tool_calls": config.max_tool_calls,
            "timeout_seconds": config.timeout_seconds,
            "planner_timeout_seconds": config.planner_timeout_seconds,
            "specialist_timeout_seconds": config.specialist_timeout_seconds,
            "critic_timeout_seconds": config.critic_timeout_seconds,
            "specialist_max_tool_calls": config.specialist_max_tool_calls,
            "critic_max_tool_calls": config.critic_max_tool_calls,
            "votes": config.votes,
            "provider_max_attempts": config.provider_max_attempts,
            "provider_backoff_base": config.provider_backoff_base,
            "private_notebook_max_bytes": config.private_notebook_max_bytes,
            "stage_context_max_bytes": config.stage_context_max_bytes,
            "search_circuit_cooldown": config.search_circuit_cooldown,
            **(
                {"langfuse_input_cost_per_million": config.langfuse_input_cost_per_million}
                if config.langfuse_input_cost_per_million is not None
                else {}
            ),
            **(
                {"langfuse_output_cost_per_million": config.langfuse_output_cost_per_million}
                if config.langfuse_output_cost_per_million is not None
                else {}
            ),
        },
        package_versions=_package_versions(),
        started_at=started_at or datetime.now(UTC),
    )


def _package_versions() -> dict[str, str]:
    try:
        package_version = version("gaia-multi-agent")
    except PackageNotFoundError:
        return {}
    return {"gaia-multi-agent": package_version}


@asynccontextmanager
async def _stdio_client(
    task: GaiaTask,
    config: RunConfig,
    *,
    fake_backend: bool,
    observability: Observability | None = None,
) -> Any:
    task_root = task.attachment_path.parent if task.attachment_path is not None else Path.cwd()
    attachments = (task.attachment_path,) if task.attachment_path is not None else ()
    previous = os.environ.get("GAIA_MCP_FAKE_BACKEND")
    os.environ.pop("GAIA_MCP_FAKE_BACKEND", None)
    if fake_backend:
        os.environ["GAIA_MCP_FAKE_BACKEND"] = "1"
    client = MCPToolClient(
        task_root=task_root,
        config=config,
        attachments=attachments,
        observability=observability,
    )
    try:
        entered = await client.__aenter__()
    finally:
        if previous is None:
            os.environ.pop("GAIA_MCP_FAKE_BACKEND", None)
        else:
            os.environ["GAIA_MCP_FAKE_BACKEND"] = previous
    try:
        yield entered
    finally:
        await client.__aexit__(None, None, None)


def _validate_prediction_identity(prediction: TaskPrediction, task: GaiaTask) -> None:
    if prediction.task_id != task.task_id or prediction.level != task.level:
        raise ValueError("orchestrator prediction identity does not match the selected task")


def _is_transient_failure(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return any(base.__name__ in _TRANSIENT_EXCEPTION_NAMES for base in type(error).__mro__)


def _is_transient_prediction(prediction: TaskPrediction) -> bool:
    return prediction.error_category == TerminalErrorCategory.TIMEOUT or (
        prediction.error_category == TerminalErrorCategory.RETRY_EXHAUSTED
        and prediction.retry_reason is not None
    )


def _has_stable_checkpoint(events: Sequence[AuditEvent]) -> bool:
    return any(
        event.event_type in {"checkpoint_saved", "checkpoint_restored"}
        for event in events
    )


def _specialist_timeout_without_stable_candidate(
    events: Sequence[AuditEvent],
) -> bool:
    latest_checkpoint = next(
        (
            event
            for event in reversed(events)
            if event.event_type in {"checkpoint_saved", "checkpoint_restored"}
        ),
        None,
    )
    specialist_timed_out = any(
        event.event_type == "stage_budget_exhausted"
        and event.payload.get("stage") == "specialist"
        and event.payload.get("reason") == "time"
        for event in events
    )
    candidate_present = bool(
        latest_checkpoint
        and latest_checkpoint.payload.get("candidate_present") is True
    )
    return specialist_timed_out and not candidate_present


def _critic_failure_is_terminal(events: Sequence[AuditEvent]) -> bool:
    return any(
        (
            event.event_type == "stage_budget_exhausted"
            and event.payload.get("stage") == "critic"
            and event.payload.get("reason") == "time"
        )
        or (
            event.event_type == "stage_retry_exhausted"
            and event.payload.get("stage") == "critic"
        )
        for event in events
    )


def _transient_failure_category(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "TimeoutError"
    if isinstance(error, ConnectionError):
        return "ConnectionError"
    provider_category = transient_provider_error_category(error)
    return provider_category if provider_category != "provider_transient" else "task_transient"


def _task_recovery_event(
    *,
    sequence: int,
    attempt_number: int,
    retry_reason: str,
    retry_category: str,
    recovery_succeeded: bool,
) -> AuditEvent:
    return AuditEvent(
        sequence=sequence,
        event_type="task_recovery",
        timestamp=datetime.now(UTC),
        payload={
            "attempt_number": attempt_number,
            "retry_reason": retry_reason,
            "retry_category": retry_category,
            "recovery_succeeded": recovery_succeeded,
        },
    )


def _category_for_exception(error: BaseException) -> TerminalErrorCategory:
    if isinstance(error, (FileNotFoundError, IsADirectoryError, PermissionError)):
        return TerminalErrorCategory.ATTACHMENT_ERROR
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return TerminalErrorCategory.MALFORMED_RESPONSE
    return TerminalErrorCategory.RETRY_EXHAUSTED


def _failure_prediction(
    task: GaiaTask,
    category: TerminalErrorCategory,
) -> TaskPrediction:
    return TaskPrediction(
        task_id=task.task_id,
        level=task.level,
        answer=None,
        correct=False,
        critic_verdict="not_run",
        error_category=category,
    )


def _task_trace_outcome(
    prediction: TaskPrediction,
    *,
    checkpoint_restored: bool,
    audit_events: Sequence[AuditEvent] = (),
) -> TaskTraceOutcome:
    terminal_route = (
        "checkpoint_restored"
        if checkpoint_restored
        else "terminal_failure"
        if prediction.error_category is not None
        else "recovered_submit"
        if prediction.recovery_succeeded
        else "submit"
    )
    return TaskTraceOutcome(
        answer_present=prediction.answer is not None,
        evidence_count=len(prediction.evidence_ids),
        critic_verdict=prediction.critic_verdict,
        error_category=(prediction.error_category.value if prediction.error_category is not None else None),
        correct=prediction.correct,
        checkpoint_restored=checkpoint_restored,
        attempt_number=prediction.attempt_number,
        recovery_count=prediction.recovery_count,
        recovery_succeeded=prediction.recovery_succeeded,
        retry_reason=prediction.retry_reason,
        retry_category=prediction.retry_category,
        terminal_route=terminal_route,
        stage_codes=_task_stage_codes(audit_events),
    )


def _task_stage_codes(events: Sequence[AuditEvent]) -> tuple[str, ...]:
    event_codes = {
        "plan_completed": "plan",
        "act_completed": "act",
        "correction_act_completed": "correct",
        "verify_completed": "verify",
        "submit_completed": "submit",
        "task_recovery": "recovery",
    }
    codes: list[str] = []
    for event in events:
        code = event_codes.get(event.event_type)
        if code is not None and code not in codes:
            codes.append(code)
    return tuple(codes)


def _sum_usage(predictions: Sequence[TaskPrediction]) -> Usage:
    return Usage(
        input_tokens=sum(item.usage.input_tokens for item in predictions),
        output_tokens=sum(item.usage.output_tokens for item in predictions),
        reasoning_tokens=sum(item.usage.reasoning_tokens for item in predictions),
        total_tokens=sum(item.usage.total_tokens for item in predictions),
    )


def _cumulative_recovery_usage(
    prior_predictions: Sequence[TaskPrediction],
    prediction: TaskPrediction,
    authenticated_usage: Usage | None = None,
) -> Usage:
    prior = _sum_usage(prior_predictions)
    authenticated = authenticated_usage or Usage()
    return Usage(
        input_tokens=max(
            prior.input_tokens,
            prediction.usage.input_tokens,
            authenticated.input_tokens,
        ),
        output_tokens=max(
            prior.output_tokens,
            prediction.usage.output_tokens,
            authenticated.output_tokens,
        ),
        reasoning_tokens=max(
            prior.reasoning_tokens,
            prediction.usage.reasoning_tokens,
            authenticated.reasoning_tokens,
        ),
        total_tokens=max(
            prior.total_tokens,
            prediction.usage.total_tokens,
            authenticated.total_tokens,
        ),
    )


def _plan_from_audit(events: Sequence[AuditEvent]) -> AgentPlan | None:
    for event in events:
        if event.event_type != "plan_completed":
            continue
        specialists = event.payload.get("specialists")
        if not isinstance(specialists, list):
            return None
        try:
            steps = tuple(
                PlanStep(specialist=SpecialistName(value), instruction="")
                for value in specialists
            )
        except ValueError:
            return None
        return AgentPlan(restatement="", steps=steps)
    return None


def _persist_task_artifacts(store: RunStore, task_id: str, task_run: _TaskRun) -> None:
    if task_run.plan is not None:
        store.write_plan(task_id, task_run.plan)
    audit_path = store.task_directory(task_id) / "audit.jsonl"
    first_sequence = _next_audit_sequence(audit_path)
    public_events = (
        event
        for event in task_run.audit_events
        if event.event_type not in _PRIVATE_RUNTIME_AUDIT_EVENT_TYPES
    )
    for sequence, event in enumerate(public_events, start=first_sequence):
        store.append_audit(task_id, event.model_copy(update={"sequence": sequence}))


def _next_audit_sequence(path: Path) -> int:
    if not path.exists():
        return 1
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("audit artifact is invalid") from error
    sequences = [record.get("sequence") if isinstance(record, Mapping) else None for record in records]
    if sequences != list(range(1, len(records) + 1)):
        raise ValueError("audit sequences must be strictly increasing and contiguous")
    return len(records) + 1


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is invalid") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _load_public_manifest(path: Path) -> RunManifest:
    raw_manifest = _load_json_object(path)
    _require_exact_keys(raw_manifest, RunManifest.model_fields, "manifest")
    _reject_forbidden_keys(raw_manifest)
    return RunManifest.model_validate(raw_manifest)


def _require_exact_keys(value: Any, fields: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} does not match its public schema")


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("result artifact contains a private field")
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)
