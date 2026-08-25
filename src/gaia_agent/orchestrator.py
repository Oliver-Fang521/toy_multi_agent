from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import string
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Required, TypedDict

from langgraph.graph import END, START, StateGraph

from gaia_agent.agents import StageProtocolError, specialist_context_size_bytes
from gaia_agent.config import MAX_STAGE_CONTEXT_BYTES, RunConfig
from gaia_agent.mcp_client import AttachmentPolicyError, ToolExecutionError
from gaia_agent.models import (
    AgentPlan,
    AgentResult,
    AuditEvent,
    CriticDecision,
    Evidence,
    GaiaTask,
    PlanStep,
    SpecialistName,
    StageCheckpoint,
    StageCursor,
    StageToolCallCounts,
    TaskPrediction,
    TerminalErrorCategory,
    Usage,
)
from gaia_agent.observability import Observability, TaskTrace
from gaia_agent.providers import ToolCallError, ToolSpec
from gaia_agent.reliability import (
    InvocationDeadline,
    InvocationDeadlineExceeded,
    ProviderTransientExhaustedError,
    ensure_current_invocation_allowed,
    invocation_deadline_scope,
)
from gaia_agent.research_notebook import ResearchNotebook
from gaia_agent.scoring import parse_number_with_status

_DEFAULT_STAGE_CONTEXT_MAX_BYTES = 24_576
_CONTEXT_BUDGET_PRESSURE_RATIO = 0.7
_MAX_OPERATIONAL_SECONDS = 1_000_000_000.0


class Planner(Protocol):
    last_usage: Usage

    async def plan(self, task: GaiaTask) -> AgentPlan: ...


class Specialist(Protocol):
    last_usage: Usage

    async def run(
        self,
        task: GaiaTask,
        step: PlanStep,
        evidence: tuple[Evidence, ...],
        tools: Any,
    ) -> AgentResult: ...


class Critic(Protocol):
    last_usage: Usage

    async def verify(
        self,
        task: GaiaTask,
        candidate: str | None,
        evidence: tuple[Evidence, ...],
        tools: Any,
    ) -> CriticDecision: ...


class EvidenceLinker(Protocol):
    last_usage: Usage

    async def link(
        self,
        task: GaiaTask,
        candidate: str,
        cards: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]: ...


class _ToolBudgetExhausted(RuntimeError):
    pass


class _RepeatedToolCall(RuntimeError):
    pass


class _CorrectionUnavailable(RuntimeError):
    pass


class _AttachmentUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class StageBudget:
    name: Literal["planner", "specialist", "critic", "submit"]
    deadline: float | None
    max_calls: int
    started_at: float | None = None
    budget_seconds: float | None = None


class StageBudgetExceeded(RuntimeError):
    def __init__(
        self,
        stage: str,
        reason: Literal["time", "tool_calls"],
    ) -> None:
        super().__init__(f"{stage} stage budget exhausted")
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class StageDeadlineEnvelope:
    """One vote's immutable absolute stage and hard-deadline cutoffs."""

    vote_started_at: float
    vote_deadline: float
    planner_deadline: float
    specialist_deadline: float
    critic_deadline: float

    @classmethod
    def for_config(cls, config: RunConfig, *, started_at: float) -> StageDeadlineEnvelope:
        planner_deadline = started_at + config.planner_timeout_seconds
        specialist_deadline = planner_deadline + config.specialist_timeout_seconds
        return cls(
            vote_started_at=started_at,
            vote_deadline=started_at + config.timeout_seconds,
            planner_deadline=planner_deadline,
            specialist_deadline=specialist_deadline,
            critic_deadline=specialist_deadline + config.critic_timeout_seconds,
        )


@dataclass(frozen=True)
class _StageActivity:
    token: int
    stage: Literal["planner", "specialist", "critic", "submit"]


@dataclass
class VoteStageRuntime:
    """Migration-compatible per-vote cutoffs and cumulative active-stage timing."""

    envelope: StageDeadlineEnvelope | None = None
    clock: Callable[[], float] | None = None
    _elapsed: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _active: dict[int, tuple[str, float]] = field(default_factory=dict)
    _next_token: int = 0

    @classmethod
    def for_config(
        cls,
        config: RunConfig,
        *,
        started_at: float,
        clock: Callable[[], float] | None = None,
    ) -> VoteStageRuntime:
        return cls(
            envelope=StageDeadlineEnvelope.for_config(config, started_at=started_at),
            clock=clock,
        )

    def bind_defaults(
        self,
        config: RunConfig,
        *,
        started_at: float,
        clock: Callable[[], float],
    ) -> None:
        if self.envelope is None:
            self.envelope = StageDeadlineEnvelope.for_config(
                config,
                started_at=started_at,
            )
        if self.clock is None:
            self.clock = clock

    def now(self) -> float:
        if self.clock is not None:
            return self.clock()
        return asyncio.get_running_loop().time()

    def begin(
        self,
        stage: Literal["planner", "specialist", "critic", "submit"],
    ) -> _StageActivity:
        self._next_token += 1
        activity = _StageActivity(self._next_token, stage)
        self._active[activity.token] = (stage, self.now())
        return activity

    def finish(self, activity: _StageActivity) -> None:
        active = self._active.pop(activity.token, None)
        if active is None:
            return
        stage, started_at = active
        self._elapsed[stage] += _finite_nonnegative(self.now() - started_at)

    def elapsed_seconds(self, stage: str) -> float:
        elapsed = self._elapsed.get(stage, 0.0)
        now = self.now()
        elapsed += sum(
            _finite_nonnegative(now - started_at)
            for active_stage, started_at in self._active.values()
            if active_stage == stage
        )
        return _finite_nonnegative(elapsed)


@dataclass
class _StageRuntime:
    active: Literal["planner", "specialist", "critic"] | None = None


@dataclass
class _CachedToolResult:
    result: dict[str, Any]
    reused: bool = False


@dataclass
class _TaskState:
    plan: AgentPlan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    evidence_origins: dict[str, int] = field(default_factory=dict)
    raw_evidence_id_digests: set[bytes] = field(default_factory=set)
    candidate_origins: dict[int, str] = field(default_factory=dict)
    candidate: str | None = None
    usage: Usage = field(default_factory=Usage)
    step_count: int = 0
    verify_count: int = 0
    critic_verdict: str = "not_run"


def _ready_for_critic(state: _TaskState) -> bool:
    if state.candidate is None:
        return False
    kinds = {item.kind for item in state.evidence}
    if _numeric_value(state.candidate) is not None:
        final_value = _numeric_value(state.candidate)
        computation_origins = {
            state.evidence_origins[item.evidence_id]
            for item in state.evidence
            if item.kind == "computation"
        }
        agreeing_origins = {
            origin
            for origin in computation_origins
            if _numeric_value(state.candidate_origins.get(origin)) == final_value
        }
        return len(agreeing_origins) >= 2 or (
            bool(agreeing_origins) and "source" in kinds
        )
    if state.plan is not None and any(
        step.specialist == SpecialistName.LOOKUP for step in state.plan.steps
    ):
        return "source" in kinds
    return bool(state.evidence)


class _BudgetedToolClient:
    def __init__(
        self,
        client: Any,
        *,
        max_calls: int,
        audit: Callable[[str, dict[str, Any]], None],
        notebook: ResearchNotebook | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = client
        self._max_calls = max_calls
        self._audit = audit
        self._notebook = notebook
        self._clock = clock
        self._scope = "unscoped"
        self._scope_logical_call_count = 0
        self._stage_budget: StageBudget | None = None
        self._stage_logical_call_counts: dict[str, int] = defaultdict(int)
        self._scope_tool_names: set[str] = set()
        self._results_by_scope: dict[str, dict[str, _CachedToolResult]] = defaultdict(dict)
        self.call_count = 0
        self.logical_call_count = 0
        self.executed_call_count = 0
        self.restored_call_count = 0

    @property
    def tool_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._client.tool_specs)

    async def read_attachment_bytes(self, path: Path | str) -> bytes:
        ensure_current_invocation_allowed()
        return await self._client.read_attachment_bytes(path)

    def start_scope(
        self,
        scope: str,
        *,
        stage: Literal["planner", "specialist", "critic"],
        max_calls: int,
        deadline: float | None,
    ) -> None:
        if not scope.strip():
            raise ValueError("tool-call scope must be non-empty")
        self._scope = scope
        self._scope_logical_call_count = 0
        self._stage_budget = StageBudget(
            name=stage,
            deadline=deadline,
            max_calls=max_calls,
        )
        self._scope_tool_names = set()

    @property
    def scope_tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._scope_tool_names))

    def stage_logical_call_count(self, stage: str) -> int:
        return self._stage_logical_call_counts[stage]

    @property
    def scope_logical_call_count(self) -> int:
        """Logical calls in the active scope, for non-budget reporting only."""

        return self._scope_logical_call_count

    def restore_logical_call_counts(
        self,
        logical_tool_calls: int,
        stage_counts: StageToolCallCounts,
    ) -> None:
        if self.logical_call_count or any(self._stage_logical_call_counts.values()):
            raise ValueError("logical tool-call counters may be restored only once")
        if logical_tool_calls != stage_counts.total:
            raise ValueError("restored logical tool-call counters do not reconcile")
        self.logical_call_count = logical_tool_calls
        self.call_count = logical_tool_calls
        self._stage_logical_call_counts.update(
            {
                "planner": stage_counts.planner,
                "specialist": stage_counts.specialist,
                "critic": stage_counts.critic,
            }
        )

    def stage_logical_call_counts(self) -> StageToolCallCounts:
        return StageToolCallCounts(
            planner=self.stage_logical_call_count("planner"),
            specialist=self.stage_logical_call_count("specialist"),
            critic=self.stage_logical_call_count("critic"),
        )

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ensure_current_invocation_allowed()
        signature = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        if self.call_count >= self._max_calls:
            raise _ToolBudgetExhausted("global tool-call budget exhausted")
        self._check_stage_budget()
        self._scope_tool_names.add(name)
        cached = self._results_by_scope[self._scope].get(signature)
        if cached is not None:
            if cached.reused:
                raise _RepeatedToolCall(f"repeated tool call: {name}")
            cached.reused = True
            self._record_logical_call()
            self._audit(
                "tool_call_reused",
                {
                    "tool": name,
                    "call_number": self.call_count,
                    "argument_keys": sorted(arguments),
                    "result_keys": sorted(cached.result),
                },
            )
            return copy.deepcopy(cached.result)
        self._record_logical_call()
        safe_payload = {
            "tool": name,
            "call_number": self.call_count,
            "argument_keys": sorted(arguments),
        }
        if self._notebook is not None:
            restored = self._notebook.lookup(name, arguments)
            if restored is not None:
                self._results_by_scope[self._scope][signature] = _CachedToolResult(
                    result=copy.deepcopy(restored)
                )
                self.restored_call_count += 1
                self._audit(
                    "tool_call_restored",
                    {**safe_payload, "result_keys": sorted(restored)},
                )
                return copy.deepcopy(restored)
        try:
            if self._stage_budget is None or self._stage_budget.deadline is None:
                ensure_current_invocation_allowed()
                self._audit("tool_call_started", safe_payload)
                self.executed_call_count += 1
                result = await self._client.call(name, arguments)
            else:
                remaining = self._stage_budget.deadline - self._now()
                if remaining <= 0:
                    raise StageBudgetExceeded(self._stage_budget.name, "time")
                loop_deadline = asyncio.get_running_loop().time() + remaining
                async with asyncio.timeout_at(loop_deadline):
                    # Recheck at the actual backend boundary. The stage clock may
                    # have advanced while arguments, cache state, and timeout
                    # scope were prepared.
                    self._check_stage_deadline()
                    ensure_current_invocation_allowed()
                    self._audit("tool_call_started", safe_payload)
                    self.executed_call_count += 1
                    result = await self._client.call(name, arguments)
                if self._now() >= self._stage_budget.deadline:
                    raise StageBudgetExceeded(self._stage_budget.name, "time")
        except StageBudgetExceeded:
            raise
        except InvocationDeadlineExceeded as error:
            assert self._stage_budget is not None
            raise StageBudgetExceeded(self._stage_budget.name, "time") from error
        except TimeoutError as error:
            assert self._stage_budget is not None
            raise StageBudgetExceeded(self._stage_budget.name, "time") from error
        except ToolCallError:
            self._audit(
                "tool_call_failed",
                {**safe_payload, "category": "tool_error"},
            )
            raise
        if self._notebook is not None:
            try:
                self._notebook.record(name, arguments, result)
            except (OSError, TypeError, ValueError):
                pass
        self._results_by_scope[self._scope][signature] = _CachedToolResult(
            result=copy.deepcopy(result)
        )
        self._audit(
            "tool_call_completed",
            {
                **safe_payload,
                "result_keys": sorted(result),
            },
        )
        return result

    def _record_logical_call(self) -> None:
        self.logical_call_count += 1
        self.call_count = self.logical_call_count
        self._scope_logical_call_count += 1
        if self._stage_budget is not None:
            self._stage_logical_call_counts[self._stage_budget.name] += 1

    def _check_stage_budget(self) -> None:
        budget = self._stage_budget
        if budget is None:
            return
        self._check_stage_deadline()
        if self._stage_logical_call_counts[budget.name] >= budget.max_calls:
            raise StageBudgetExceeded(budget.name, "tool_calls")

    def _check_stage_deadline(self) -> None:
        budget = self._stage_budget
        if (
            budget is not None
            and budget.deadline is not None
            and budget.deadline <= self._now()
        ):
            raise StageBudgetExceeded(budget.name, "time")

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_running_loop().time()


class _GraphState(TypedDict, total=False):
    task: Required[GaiaTask]
    runtime: Required[_TaskState]
    tools: Required[_BudgetedToolClient]
    budgets: Required[dict[str, StageBudget]]
    stage_runtime: Required[_StageRuntime]
    next_step_index: int
    soft_step_failures: int
    decision: CriticDecision
    prediction: TaskPrediction
    entry_cursor: Literal[
        "plan", "act", "evidence_repair", "verify", "correct", "submit"
    ]


class Orchestrator:
    """Run one bounded, provider-neutral Plan -> Act -> Verify -> Submit attempt."""

    def __init__(
        self,
        *,
        planner: Planner,
        specialists: Mapping[SpecialistName, Specialist],
        critic: Critic,
        tool_client_factory: Callable[[GaiaTask], AbstractAsyncContextManager[Any]],
        config: RunConfig,
        observability: Observability | None = None,
        notebook: ResearchNotebook | None = None,
        evidence_linker: EvidenceLinker | None = None,
        stage_runtime: VoteStageRuntime | None = None,
    ) -> None:
        if config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if config.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        if config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        unknown_keys = set(specialists) - set(SpecialistName)
        if unknown_keys:
            raise ValueError(f"unknown specialist mappings: {sorted(unknown_keys, key=str)}")
        self._planner = planner
        self._specialists = dict(specialists)
        self._critic = critic
        self._tool_client_factory = tool_client_factory
        self._config = config
        self._observability = observability or Observability()
        self._notebook = notebook
        self._evidence_linker = evidence_linker
        self._vote_stage_runtime = stage_runtime
        self._trace: TaskTrace | None = None
        self._audit_events: list[AuditEvent] = []
        self.correction_count = 0
        self.step_count = 0
        self.tool_call_count = 0
        self._invocation_deadline: InvocationDeadline | None = None
        self._graph = self._build_graph()

    @property
    def audit_events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit_events)

    def set_invocation_deadline(self, deadline: InvocationDeadline) -> None:
        self._invocation_deadline = deadline

    def set_stage_runtime(self, runtime: VoteStageRuntime) -> None:
        self._vote_stage_runtime = runtime

    def authenticated_checkpoint_usage(self) -> Usage | None:
        if self._notebook is None:
            return None
        checkpoint = self._notebook.load_stage_checkpoint()
        return checkpoint.usage if checkpoint is not None else None

    async def run(self, task: GaiaTask) -> TaskPrediction:
        with invocation_deadline_scope(self._invocation_deadline):
            return await self._run(task)

    async def _run(self, task: GaiaTask) -> TaskPrediction:
        loop = asyncio.get_running_loop()
        started = (
            self._vote_stage_runtime.now()
            if self._vote_stage_runtime is not None
            else loop.time()
        )
        if self._vote_stage_runtime is None:
            self._vote_stage_runtime = VoteStageRuntime.for_config(
                self._config,
                started_at=started,
                clock=loop.time,
            )
        else:
            self._vote_stage_runtime.bind_defaults(
                self._config,
                started_at=started,
                clock=loop.time,
            )
        assert self._vote_stage_runtime.envelope is not None
        global_deadline = self._vote_stage_runtime.envelope.vote_deadline
        self._audit_events = []
        self.correction_count = 0
        self.step_count = 0
        self.tool_call_count = 0
        self._trace = self._observability.current_trace()
        state = _TaskState()
        execution_task = task.model_copy(update={"final_answer": ""})
        budgeted_tools: _BudgetedToolClient | None = None
        raw_tools: Any | None = None
        raw_tool_usage_collected = False
        prediction: TaskPrediction | None = None
        category: TerminalErrorCategory | None = None
        provider_exhaustion: ProviderTransientExhaustedError | None = None
        cancellation: asyncio.CancelledError | None = None
        cancellation_checkpoint_refreshed = False
        budgets = self._stage_budgets(started)
        stage_runtime = _StageRuntime()
        try:
            try:
                global_remaining = global_deadline - self._now()
                if global_remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(global_remaining):
                    ensure_current_invocation_allowed()
                    async with self._tool_client_factory(execution_task) as active_tools:
                        raw_tools = active_tools
                        budgeted_tools = _BudgetedToolClient(
                            active_tools,
                            max_calls=self._config.max_tool_calls,
                            audit=self._audit,
                            notebook=self._notebook,
                            clock=self._now,
                        )
                        try:
                            prediction = await self._execute(
                                state,
                                execution_task,
                                budgeted_tools,
                                budgets,
                                stage_runtime,
                            )
                        except asyncio.CancelledError:
                            state.usage = _add_usage(
                                state.usage,
                                getattr(raw_tools, "usage", Usage()),
                            )
                            raw_tool_usage_collected = True
                            self.tool_call_count = budgeted_tools.call_count
                            self.step_count = state.step_count
                            self._refresh_checkpoint_accounting(
                                state,
                                budgeted_tools,
                            )
                            cancellation_checkpoint_refreshed = True
                            raise
                    if self._now() >= global_deadline:
                        raise TimeoutError
            except TimeoutError as error:
                active_stage = stage_runtime.active
                if active_stage is not None:
                    stage_deadline = budgets[active_stage].deadline
                    if stage_deadline is not None and self._now() >= stage_deadline:
                        raise StageBudgetExceeded(active_stage, "time") from error
                raise
        except asyncio.CancelledError as error:
            cancellation = error
        except ProviderTransientExhaustedError as error:
            category = TerminalErrorCategory.RETRY_EXHAUSTED
            provider_exhaustion = error
        except TimeoutError:
            category = TerminalErrorCategory.TIMEOUT
        except StageBudgetExceeded as error:
            category = (
                TerminalErrorCategory.TIMEOUT
                if error.reason == "time"
                else TerminalErrorCategory.TOOL_BUDGET_EXHAUSTED
            )
            self._audit_stage_budget_exhausted(
                error,
                budgets[error.stage],
                budgeted_tools,
            )
        except _ToolBudgetExhausted:
            category = TerminalErrorCategory.TOOL_BUDGET_EXHAUSTED
        except _RepeatedToolCall:
            category = TerminalErrorCategory.LOOP_DETECTED
        except _CorrectionUnavailable:
            category = TerminalErrorCategory.CRITIC_REJECTION
        except _AttachmentUnavailable:
            category = TerminalErrorCategory.ATTACHMENT_ERROR
        except AttachmentPolicyError:
            category = TerminalErrorCategory.ATTACHMENT_ERROR
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            category = TerminalErrorCategory.ATTACHMENT_ERROR
        except (ValueError, TypeError, KeyError):
            category = TerminalErrorCategory.MALFORMED_RESPONSE
        except ToolExecutionError:
            category = TerminalErrorCategory.RETRY_EXHAUSTED
        except Exception:  # noqa: BLE001 - task isolation requires a terminal result
            category = TerminalErrorCategory.RETRY_EXHAUSTED
        finally:
            if raw_tools is not None and not raw_tool_usage_collected:
                state.usage = _add_usage(
                    state.usage,
                    getattr(raw_tools, "usage", Usage()),
                )
        if budgeted_tools is not None:
            self.tool_call_count = budgeted_tools.call_count
        self.step_count = state.step_count
        if cancellation is not None:
            if not cancellation_checkpoint_refreshed:
                self._refresh_checkpoint_accounting(state, budgeted_tools)
            raise cancellation
        if prediction is not None:
            return prediction.model_copy(
                update={
                    "duration_seconds": self._now() - started,
                    "usage": state.usage,
                }
            )
        assert category is not None
        self._refresh_checkpoint_accounting(state, budgeted_tools)
        self._audit(
            "terminal_failure",
            {"category": category.value, "critic_verdict": state.critic_verdict},
        )
        return TaskPrediction(
            task_id=task.task_id,
            level=task.level,
            answer=None,
            duration_seconds=self._now() - started,
            usage=state.usage,
            critic_verdict=state.critic_verdict,
            evidence_ids=tuple(item.evidence_id for item in state.evidence),
            error_category=category,
            retry_reason=(
                "provider_transient_exhausted" if provider_exhaustion is not None else None
            ),
            retry_category=(
                provider_exhaustion.retry_category if provider_exhaustion is not None else None
            ),
            provider_attempt_count=(
                provider_exhaustion.attempt_count if provider_exhaustion is not None else 0
            ),
        )

    def _refresh_checkpoint_accounting(
        self,
        state: _TaskState,
        tools: _BudgetedToolClient | None,
    ) -> None:
        if self._notebook is None or tools is None:
            return
        checkpoint = self._notebook.load_stage_checkpoint()
        if checkpoint is None:
            return
        refreshed = StageCheckpoint.model_validate(
            {
                **checkpoint.model_dump(mode="python"),
                "usage": state.usage,
                "logical_tool_calls": tools.logical_call_count,
                "stage_logical_tool_calls": tools.stage_logical_call_counts(),
            }
        )
        self._notebook.save_stage_checkpoint(refreshed)
        self._audit(
            "checkpoint_saved",
            {
                "cursor": refreshed.cursor,
                "logical_tool_calls": refreshed.logical_tool_calls,
                "step_count": refreshed.step_count,
                "candidate_present": refreshed.candidate is not None,
            },
        )

    async def _execute(
        self,
        state: _TaskState,
        task: GaiaTask,
        tools: _BudgetedToolClient,
        budgets: dict[str, StageBudget],
        stage_runtime: _StageRuntime,
    ) -> TaskPrediction:
        result = await self._graph.ainvoke(
            {
                "task": task,
                "runtime": state,
                "tools": tools,
                "budgets": budgets,
                "stage_runtime": stage_runtime,
                "next_step_index": 0,
                "soft_step_failures": 0,
            },
            config={"recursion_limit": self._graph_recursion_limit()},
        )
        prediction = result.get("prediction")
        if not isinstance(prediction, TaskPrediction):
            raise TypeError("graph completed without a prediction")
        return prediction

    def _build_graph(self) -> Any:
        graph = StateGraph(_GraphState)
        graph.add_node("restore", self._graph_restore)
        graph.add_node("plan", self._graph_plan)
        graph.add_node("act", self._graph_act)
        graph.add_node("evidence_repair", self._graph_evidence_repair)
        graph.add_node("verify", self._graph_verify)
        graph.add_node("correct", self._graph_correct)
        graph.add_node("reject", self._graph_reject)
        graph.add_node("submit", self._graph_submit)
        graph.add_edge(START, "restore")
        graph.add_conditional_edges(
            "restore",
            self._route_after_restore,
            {
                "plan": "plan",
                "act": "act",
                "evidence_repair": "evidence_repair",
                "verify": "verify",
                "correct": "correct",
                "submit": "submit",
            },
        )
        graph.add_edge("plan", "act")
        graph.add_conditional_edges(
            "act",
            self._route_after_act,
            {
                "act": "act",
                "evidence_repair": "evidence_repair",
                "verify": "verify",
            },
        )
        graph.add_edge("evidence_repair", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"correct": "correct", "reject": "reject", "submit": "submit"},
        )
        graph.add_edge("correct", "verify")
        graph.add_edge("reject", END)
        graph.add_edge("submit", END)
        return graph.compile()

    def _graph_recursion_limit(self) -> int:
        return max(25, self._config.max_steps + 10)

    async def _graph_restore(self, graph_state: _GraphState) -> dict[str, Any]:
        checkpoint = (
            self._notebook.load_stage_checkpoint()
            if self._notebook is not None
            else None
        )
        if checkpoint is None:
            return {"entry_cursor": "plan"}
        state = graph_state["runtime"]
        state.plan = checkpoint.plan
        state.evidence = list(checkpoint.evidence)
        state.evidence_origins = dict(checkpoint.evidence_origins)
        state.raw_evidence_id_digests = {
            bytes.fromhex(digest) for digest in checkpoint.raw_evidence_id_digests
        }
        state.candidate_origins = dict(checkpoint.candidate_origins)
        state.candidate = checkpoint.candidate
        state.usage = checkpoint.usage
        state.step_count = checkpoint.step_count
        state.verify_count = checkpoint.verify_count
        state.critic_verdict = checkpoint.critic_verdict
        self.correction_count = checkpoint.correction_count
        self.step_count = checkpoint.step_count
        graph_state["tools"].restore_logical_call_counts(
            checkpoint.logical_tool_calls,
            checkpoint.stage_logical_tool_calls,
        )
        self._audit(
            "checkpoint_restored",
            {
                "cursor": checkpoint.cursor,
                "logical_tool_calls": checkpoint.logical_tool_calls,
                "step_count": checkpoint.step_count,
                "candidate_present": checkpoint.candidate is not None,
            },
        )
        restored: dict[str, Any] = {
            "runtime": state,
            "entry_cursor": checkpoint.cursor,
            "next_step_index": checkpoint.next_step_index,
            "soft_step_failures": checkpoint.soft_step_failures,
        }
        if checkpoint.decision is not None:
            restored["decision"] = checkpoint.decision
        return restored

    def _route_after_restore(
        self,
        graph_state: _GraphState,
    ) -> Literal["plan", "act", "evidence_repair", "verify", "correct", "submit"]:
        return graph_state.get("entry_cursor", "plan")

    async def _graph_plan(self, graph_state: _GraphState) -> dict[str, Any]:
        state = graph_state["runtime"]
        task = graph_state["task"]
        self._audit("plan_started", {"task_id": task.task_id})
        stage_started = self._now()
        with self._stage_span("planner", {"stage": "planner"}) as planner_span:
            if planner_span is not None:
                planner_span.set_summary(
                    lambda: self._stage_operational_summary(
                        graph_state["budgets"]["planner"], None, stage_started
                    )
                )
            try:
                plan = await self._run_in_stage(
                    graph_state["budgets"]["planner"],
                    lambda: self._planner.plan(task),
                    graph_state["stage_runtime"],
                )
            finally:
                state.usage = _add_usage(state.usage, self._planner.last_usage)
            if planner_span is not None:
                specialists = [step.specialist.value for step in plan.steps]
                planner_span.set_summary(
                    {
                        "route": specialists[0] if len(set(specialists)) == 1 else "multi_stage",
                        "specialists": specialists,
                        "step_count": len(plan.steps),
                        "repair_attempted": bool(
                            getattr(self._planner, "last_repair_attempted", False)
                        ),
                        "fallback_used": bool(
                            getattr(self._planner, "last_fallback_used", False)
                        ),
                        **self._stage_operational_summary(
                            graph_state["budgets"]["planner"],
                            None,
                            stage_started,
                        ),
                    }
                )
        if not plan.steps or len(plan.steps) > self._config.max_steps:
            raise ValueError("planner exceeded the global step budget")
        state.plan = plan
        self._audit(
            "plan_completed",
            {
                "task_id": task.task_id,
                "step_count": len(plan.steps),
                "specialists": [step.specialist.value for step in plan.steps],
            },
        )
        update = {"runtime": state, "next_step_index": 0}
        self._save_checkpoint(graph_state, "act", next_step_index=0)
        return update

    async def _graph_act(self, graph_state: _GraphState) -> dict[str, Any]:
        state = graph_state["runtime"]
        task = graph_state["task"]
        tools = graph_state["tools"]
        if state.plan is None:
            raise ValueError("graph act node requires a plan")
        step_index = graph_state.get("next_step_index", 0)
        if step_index < 0 or step_index >= len(state.plan.steps):
            raise ValueError("graph act node received an invalid plan index")
        if pressure := self._context_budget_pressure(
            state,
            task,
            state.plan.steps[step_index],
        ):
            self._audit_context_budget_pressure(pressure)
            return {"runtime": state, "next_step_index": len(state.plan.steps)}
        stable_step_count = state.step_count
        try:
            await self._act(
                state,
                task,
                state.plan.steps[step_index],
                tools,
                correction=False,
                budget=graph_state["budgets"]["specialist"],
                stage_runtime=graph_state["stage_runtime"],
            )
        except StageBudgetExceeded as error:
            if error.reason != "time" or state.candidate is None:
                raise
            state.step_count = stable_step_count
            self.step_count = stable_step_count
            self._audit_stage_budget_exhausted(
                error,
                graph_state["budgets"]["specialist"],
                tools,
            )
            return {"runtime": state, "next_step_index": len(state.plan.steps)}
        except StageProtocolError as error:
            if (
                not error.repairable
                or not error.repair_attempted
                or graph_state.get("soft_step_failures", 0) >= 1
                or state.candidate is None
                or not state.evidence
            ):
                raise
            attempted_step_number = state.step_count
            state.step_count = stable_step_count
            self.step_count = stable_step_count
            self._audit(
                "soft_step_failure",
                {"step_number": attempted_step_number, "code": error.code.value},
            )
            return {
                "runtime": state,
                "next_step_index": len(state.plan.steps),
                "soft_step_failures": graph_state.get("soft_step_failures", 0) + 1,
            }
        return {"runtime": state, "next_step_index": step_index + 1}

    def _route_after_act(
        self,
        graph_state: _GraphState,
    ) -> Literal["act", "evidence_repair", "verify"]:
        state = graph_state["runtime"]
        if state.plan is None:
            raise ValueError("graph act router requires a plan")
        step_index = graph_state.get("next_step_index", 0)
        if step_index < len(state.plan.steps):
            next_step = state.plan.steps[step_index]
            if pressure := self._context_budget_pressure(
                state,
                graph_state["task"],
                next_step,
            ):
                self._audit_context_budget_pressure(pressure)
                route: Literal["act", "evidence_repair", "verify"] = "verify"
                self._save_checkpoint(
                    graph_state,
                    route,
                    next_step_index=step_index,
                )
                return route
            if _ready_for_critic(state):
                self._audit(
                    "plan_early_stop",
                    {
                        "completed_steps": state.step_count,
                        "remaining_steps": len(state.plan.steps) - step_index,
                    },
                )
                route = "verify"
                self._save_checkpoint(
                    graph_state,
                    route,
                    next_step_index=step_index,
                )
                return route
            route = "act"
            self._save_checkpoint(
                graph_state,
                route,
                next_step_index=step_index,
            )
            return route
        if step_index == len(state.plan.steps):
            route = (
                "evidence_repair"
                if state.candidate is not None and not state.evidence
                else "verify"
            )
            self._save_checkpoint(
                graph_state,
                route,
                next_step_index=step_index,
            )
            return route
        raise ValueError("graph act router received an invalid plan index")

    async def _graph_evidence_repair(
        self,
        graph_state: _GraphState,
    ) -> dict[str, Any]:
        state = graph_state["runtime"]
        task = graph_state["task"]
        if state.plan is None or state.candidate is None:
            raise ValueError("evidence repair requires a stable plan and candidate")
        cards = (
            self._notebook.linkable_evidence_cards(
                max_json_bytes=self._config.stage_context_max_bytes
            )
            if self._notebook is not None
            else ()
        )
        selected_hashes: tuple[str, ...] = ()
        if self._evidence_linker is not None and cards and not state.evidence:
            self._audit("evidence_repair_started", {"card_count": len(cards)})
            try:
                selected_hashes = await self._run_in_stage(
                    _evidence_repair_budget(
                        graph_state["budgets"]["critic"],
                        now=self._now(),
                    ),
                    lambda: self._evidence_linker.link(
                        task,
                        state.candidate or "",
                        cards,
                    ),
                    graph_state["stage_runtime"],
                )
            except Exception as error:  # noqa: BLE001 - optional repair boundary
                self._audit(
                    "evidence_repair_failed",
                    {"category": _evidence_repair_failure_category(error)},
                )
            finally:
                state.usage = _add_usage(state.usage, self._evidence_linker.last_usage)
            resolved = self._notebook.evidence_for_hashes(selected_hashes)
            if resolved:
                if state.step_count < 1:
                    raise ValueError("evidence repair requires a completed specialist step")
                source_step = state.plan.steps[
                    min(state.step_count - 1, len(state.plan.steps) - 1)
                ]
                self._record_result(
                    state,
                    source_step,
                    AgentResult(
                        specialist=source_step.specialist,
                        evidence=resolved,
                    ),
                )
            self._audit(
                "evidence_repair_completed",
                {
                    "selected_count": len(selected_hashes),
                    "resolved_count": len(resolved),
                },
            )
        self._save_checkpoint(
            graph_state,
            "verify",
            next_step_index=graph_state.get("next_step_index", len(state.plan.steps)),
        )
        return {"runtime": state}

    async def _graph_verify(self, graph_state: _GraphState) -> dict[str, Any]:
        state = graph_state["runtime"]
        task = graph_state["task"]
        tools = graph_state["tools"]
        try:
            decision = await self._verify(
                state,
                task,
                tools,
                budget=graph_state["budgets"]["critic"],
                stage_runtime=graph_state["stage_runtime"],
            )
        except ProviderTransientExhaustedError as error:
            self._audit(
                "stage_retry_exhausted",
                {"stage": "critic", "retry_category": error.retry_category},
            )
            raise
        decision = self._apply_quality_policy(state, decision)
        return {"runtime": state, "decision": decision}

    def _route_after_verify(
        self,
        graph_state: _GraphState,
    ) -> Literal["correct", "reject", "submit"]:
        decision = graph_state.get("decision")
        if decision is None:
            raise ValueError("graph verify router requires a critic decision")
        if decision.verdict == "approve":
            self._save_checkpoint(graph_state, "submit")
            return "submit"
        if not self.correction_count:
            self._save_checkpoint(graph_state, "correct")
            return "correct"
        self._save_checkpoint(graph_state, "correct")
        return "reject"

    async def _graph_correct(self, graph_state: _GraphState) -> dict[str, Any]:
        state = graph_state["runtime"]
        decision = graph_state.get("decision")
        if decision is None:
            raise ValueError("graph correction node requires a critic decision")
        await self._correct_once(
            state,
            graph_state["task"],
            decision,
            graph_state["tools"],
            budget=graph_state["budgets"]["specialist"],
            stage_runtime=graph_state["stage_runtime"],
        )
        self._save_checkpoint(graph_state, "verify")
        return {"runtime": state}

    async def _graph_reject(self, graph_state: _GraphState) -> dict[str, Any]:
        raise _CorrectionUnavailable("critic rejected the candidate after correction")

    async def _graph_submit(self, graph_state: _GraphState) -> dict[str, Any]:
        decision = graph_state.get("decision")
        if decision is None:
            raise ValueError("graph submit node requires a critic decision")
        budget = graph_state["budgets"]["critic"]
        now = self._now()
        submit_budget = StageBudget(
            name="submit",
            deadline=budget.deadline,
            max_calls=budget.max_calls,
            started_at=now,
            budget_seconds=(max(0.0, budget.deadline - now) if budget.deadline is not None else 0.0),
        )
        prediction = await self._run_in_stage(
            budget,
            lambda: self._submit(
                graph_state["runtime"],
                graph_state["task"],
                decision,
                graph_state["tools"],
                budget=budget,
                summary_budget=submit_budget,
            ),
            graph_state["stage_runtime"],
            telemetry_stage="submit",
        )
        return {"prediction": prediction}

    def _save_checkpoint(
        self,
        graph_state: _GraphState,
        cursor: StageCursor,
        *,
        next_step_index: int | None = None,
    ) -> None:
        if self._notebook is None:
            return
        state = graph_state["runtime"]
        if state.plan is None:
            raise ValueError("stable checkpoint requires a plan")
        decision = (
            graph_state.get("decision") if cursor in {"correct", "submit"} else None
        )
        checkpoint = StageCheckpoint(
            cursor=cursor,
            plan=state.plan,
            evidence=tuple(state.evidence),
            evidence_origins=dict(state.evidence_origins),
            raw_evidence_id_digests=tuple(
                sorted(digest.hex() for digest in state.raw_evidence_id_digests)
            ),
            candidate_origins=dict(state.candidate_origins),
            candidate=state.candidate,
            usage=state.usage,
            step_count=state.step_count,
            verify_count=state.verify_count,
            correction_count=self.correction_count,
            next_step_index=(
                graph_state.get("next_step_index", 0)
                if next_step_index is None
                else next_step_index
            ),
            soft_step_failures=graph_state.get("soft_step_failures", 0),
            logical_tool_calls=graph_state["tools"].logical_call_count,
            stage_logical_tool_calls=graph_state["tools"].stage_logical_call_counts(),
            critic_verdict=state.critic_verdict,
            decision=decision,
        )
        self._notebook.save_stage_checkpoint(checkpoint)
        self._audit(
            "checkpoint_saved",
            {
                "cursor": cursor,
                "logical_tool_calls": checkpoint.logical_tool_calls,
                "step_count": checkpoint.step_count,
                "candidate_present": checkpoint.candidate is not None,
            },
        )

    async def _act(
        self,
        state: _TaskState,
        task: GaiaTask,
        step: PlanStep,
        tools: _BudgetedToolClient,
        *,
        correction: bool,
        budget: StageBudget,
        stage_runtime: _StageRuntime | None = None,
    ) -> None:
        if state.step_count >= self._config.max_steps:
            raise _CorrectionUnavailable("correction would exceed the global step budget")
        if step.specialist in {SpecialistName.FILE, SpecialistName.VISUAL} and (
            task.attachment_path is None or not task.attachment_path.is_file()
        ):
            raise _AttachmentUnavailable("attachment specialist requires a registered file")
        specialist = self._specialists.get(step.specialist)
        if specialist is None:
            raise ValueError(f"no specialist registered for {step.specialist.value}")
        state.step_count += 1
        self.step_count = state.step_count
        event_prefix = "correction_act" if correction else "act"
        scope = (
            f"correction:{self.correction_count}"
            if correction
            else f"act:{state.step_count}"
        )
        tools.start_scope(
            scope,
            stage=budget.name,
            max_calls=budget.max_calls,
            deadline=budget.deadline,
        )
        self._audit(
            f"{event_prefix}_started",
            {
                "specialist": step.specialist.value,
                "step_number": state.step_count,
            },
        )
        stage_started = self._now()
        with self._stage_span(
            "specialist",
            {
                "stage": "act",
                "specialist": step.specialist.value,
                "step_number": state.step_count,
            },
        ) as specialist_span:
            if specialist_span is not None:
                specialist_span.set_summary(
                    lambda: self._stage_operational_summary(budget, tools, stage_started)
                )
            try:
                result = await self._run_in_stage(
                    budget,
                    lambda: specialist.run(task, step, tuple(state.evidence), tools),
                    stage_runtime,
                )
            finally:
                state.usage = _add_usage(state.usage, specialist.last_usage)
            if specialist_span is not None:
                specialist_span.set_summary(
                    {
                        "specialist": step.specialist.value,
                        "step_number": state.step_count,
                        "tool_names": list(tools.scope_tool_names),
                        "evidence_count": len(result.evidence),
                        "evidence_kinds": sorted({item.kind for item in result.evidence}),
                        "candidate_present": result.candidate_answer is not None,
                        "repair_attempted": bool(
                            getattr(specialist, "last_repair_attempted", False)
                        ),
                        **self._stage_operational_summary(budget, tools, stage_started),
                    }
                )
        trusted_evidence_ids = self._record_result(state, step, result)
        self._audit(
            f"{event_prefix}_completed",
            {
                "specialist": step.specialist.value,
                "step_number": state.step_count,
                "evidence_ids": list(trusted_evidence_ids),
                "has_candidate": result.candidate_answer is not None,
            },
        )

    def _record_result(
        self,
        state: _TaskState,
        step: PlanStep,
        result: AgentResult,
    ) -> tuple[str, ...]:
        if result.specialist != step.specialist:
            raise ValueError("specialist result does not match dispatched plan step")
        result_ids = [item.evidence_id for item in result.evidence]
        if any(not evidence_id.strip() for evidence_id in result_ids):
            raise ValueError("evidence IDs must be non-empty")
        result_digests = [
            hashlib.sha256(evidence_id.encode("utf-8")).digest()
            for evidence_id in result_ids
        ]
        if len(result_digests) != len(set(result_digests)) or (
            state.raw_evidence_id_digests.intersection(result_digests)
        ):
            raise ValueError("evidence IDs must be globally unique")
        if any(item.kind == "source" and not item.source_url for item in result.evidence):
            raise ValueError("source evidence must include a URL")
        if len(state.evidence) + len(result.evidence) > 9_999:
            raise ValueError("evidence limit exceeded")
        trusted_ids: list[str] = []
        for item, raw_digest in zip(result.evidence, result_digests, strict=True):
            trusted_id = f"evidence-{len(state.evidence) + 1:04d}"
            trusted_item = item.model_copy(update={"evidence_id": trusted_id})
            state.evidence.append(trusted_item)
            state.evidence_origins[trusted_id] = state.step_count
            state.raw_evidence_id_digests.add(raw_digest)
            trusted_ids.append(trusted_id)
        if result.candidate_answer is not None:
            candidate = result.candidate_answer.strip()
            if not candidate or "\n" in candidate:
                raise ValueError("candidate answer must use the concise GAIA format")
            state.candidate = candidate
            state.candidate_origins[state.step_count] = candidate
        return tuple(trusted_ids)

    async def _verify(
        self,
        state: _TaskState,
        task: GaiaTask,
        tools: _BudgetedToolClient,
        *,
        budget: StageBudget,
        stage_runtime: _StageRuntime | None = None,
    ) -> CriticDecision:
        state.verify_count += 1
        tools.start_scope(
            f"verify:{state.verify_count}",
            stage=budget.name,
            max_calls=budget.max_calls,
            deadline=budget.deadline,
        )
        self._audit(
            "verify_started",
            {
                "evidence_ids": [item.evidence_id for item in state.evidence],
                "has_candidate": state.candidate is not None,
            },
        )
        stage_started = self._now()
        with self._stage_span(
            "critic",
            {
                "stage": "verify",
                "evidence_count": len(state.evidence),
                "has_candidate": state.candidate is not None,
            },
        ) as critic_span:
            if critic_span is not None:
                critic_span.set_summary(
                    lambda: self._stage_operational_summary(budget, tools, stage_started)
                )
            try:
                decision = await self._run_in_stage(
                    budget,
                    lambda: self._critic.verify(
                        task,
                        state.candidate,
                        tuple(state.evidence),
                        tools,
                    ),
                    stage_runtime,
                )
            finally:
                state.usage = _add_usage(state.usage, self._critic.last_usage)
            if critic_span is not None:
                critic_span.set_summary(
                    {
                        "verdict": decision.verdict,
                        "decision": decision.verdict,
                        "checked_count": len(decision.checked_evidence_ids),
                        "tool_names": list(tools.scope_tool_names),
                        "candidate_present": state.candidate is not None,
                        "repair_attempted": bool(
                            getattr(self._critic, "last_repair_attempted", False)
                        ),
                        "fallback_reason": getattr(
                            self._critic,
                            "last_fallback_reason",
                            None,
                        ),
                        **self._stage_operational_summary(budget, tools, stage_started),
                    }
                )
        known_ids = {item.evidence_id for item in state.evidence}
        checked_ids = decision.checked_evidence_ids
        if len(checked_ids) != len(set(checked_ids)):
            raise ValueError("critic checked duplicate evidence IDs")
        if set(checked_ids) - known_ids:
            raise ValueError("critic checked unknown evidence IDs")
        if decision.verdict == "approve" and state.candidate is None:
            raise ValueError("critic approved a missing candidate")
        if decision.verdict == "revise" and not (
            decision.corrected_answer and decision.corrected_answer.strip()
        ):
            raise ValueError("critic revision requires a corrected answer")
        if decision.verdict != "revise" and decision.corrected_answer is not None:
            raise ValueError("only a revision may include a corrected answer")
        state.critic_verdict = decision.verdict
        self._audit(
            "verify_completed",
            {
                "verdict": decision.verdict,
                "checked_evidence_ids": list(checked_ids),
            },
        )
        return decision

    def _apply_quality_policy(
        self,
        state: _TaskState,
        decision: CriticDecision,
    ) -> CriticDecision:
        if decision.verdict != "approve":
            return decision
        if self._config.quality_policy == "benchmark":
            if not decision.checked_evidence_ids:
                raise ValueError("critic approval must check evidence")
            if not state.evidence:
                state.critic_verdict = "insufficient_evidence"
                self._audit(
                    "quality_rejected",
                    {"category": "insufficient_evidence"},
                )
                return CriticDecision(
                    verdict="insufficient_evidence",
                    feedback="critic approval did not include usable evidence",
                    checked_evidence_ids=decision.checked_evidence_ids,
                )
            return decision
        if self._has_sufficient_evidence(state):
            if not decision.checked_evidence_ids:
                raise ValueError("critic approval must check evidence")
            return decision
        state.critic_verdict = "insufficient_evidence"
        self._audit(
            "quality_rejected",
            {"category": "insufficient_evidence"},
        )
        return CriticDecision(
            verdict="insufficient_evidence",
            feedback=self._evidence_gap(state),
            checked_evidence_ids=decision.checked_evidence_ids,
        )

    async def _correct_once(
        self,
        state: _TaskState,
        task: GaiaTask,
        decision: CriticDecision,
        tools: _BudgetedToolClient,
        *,
        budget: StageBudget | None = None,
        stage_runtime: _StageRuntime | None = None,
    ) -> None:
        if self.correction_count:
            raise _CorrectionUnavailable("only one correction cycle is allowed")
        self.correction_count = 1
        active_budget = budget or StageBudget(
            name="specialist",
            deadline=None,
            max_calls=self._config.specialist_max_tool_calls,
        )
        tools.start_scope(
            f"correction:{self.correction_count}",
            stage=active_budget.name,
            max_calls=active_budget.max_calls,
            deadline=active_budget.deadline,
        )
        self._audit(
            "correction_started",
            {"verdict": decision.verdict},
        )
        if decision.verdict == "revise":
            corrected = (decision.corrected_answer or "").strip()
            if not corrected or "\n" in corrected:
                raise ValueError("critic correction must use the concise GAIA format")
            state.candidate = corrected
        else:
            target = self._correction_specialist(state)
            step = PlanStep(specialist=target, instruction=decision.feedback)
            if pressure := self._context_budget_pressure(state, task, step):
                self._audit_context_budget_pressure(pressure)
            else:
                await self._act(
                    state,
                    task,
                    step,
                    tools,
                    correction=True,
                    budget=active_budget,
                    stage_runtime=stage_runtime,
                )
        self._audit(
            "correction_completed",
            {
                "has_candidate": state.candidate is not None,
                "evidence_ids": [item.evidence_id for item in state.evidence],
            },
        )

    def _correction_specialist(self, state: _TaskState) -> SpecialistName:
        if _numeric_value(state.candidate) is not None:
            return SpecialistName.COMPUTATION
        assert state.plan is not None
        if any(step.specialist == SpecialistName.LOOKUP for step in state.plan.steps):
            return SpecialistName.LOOKUP
        return state.plan.steps[-1].specialist

    def _context_budget_pressure(
        self,
        state: _TaskState,
        task: GaiaTask,
        step: PlanStep,
    ) -> dict[str, int] | None:
        max_context_bytes = getattr(
            self._specialists.get(step.specialist),
            "max_context_bytes",
            _DEFAULT_STAGE_CONTEXT_MAX_BYTES,
        )
        max_context_bytes = min(max_context_bytes, MAX_STAGE_CONTEXT_BYTES)
        context_bytes, item_count = specialist_context_size_bytes(
            task,
            step,
            state.evidence,
            max_json_bytes=max_context_bytes,
        )
        if context_bytes < max_context_bytes * _CONTEXT_BUDGET_PRESSURE_RATIO:
            return None
        return {
            "bytes": min(context_bytes, max_context_bytes),
            "item_count": item_count,
            "max_bytes": max_context_bytes,
        }

    def _audit_context_budget_pressure(self, payload: dict[str, int]) -> None:
        self._audit("context_budget_pressure", payload)

    def _has_sufficient_evidence(self, state: _TaskState) -> bool:
        if state.candidate is None:
            return False
        kinds = {item.kind for item in state.evidence}
        if _numeric_value(state.candidate) is not None:
            final_value = _numeric_value(state.candidate)
            computation_origins = {
                state.evidence_origins[item.evidence_id]
                for item in state.evidence
                if item.kind == "computation"
            }
            agreeing_origins = {
                origin
                for origin in computation_origins
                if _numeric_value(state.candidate_origins.get(origin)) == final_value
            }
            return len(agreeing_origins) >= 2 or (
                bool(agreeing_origins) and "source" in kinds
            )
        assert state.plan is not None
        is_research = any(
            step.specialist == SpecialistName.LOOKUP for step in state.plan.steps
        )
        if is_research:
            return "source" in kinds
        return bool(state.evidence)

    def _evidence_gap(self, state: _TaskState) -> str:
        if state.candidate is None:
            return "candidate answer is missing"
        if _numeric_value(state.candidate) is not None:
            return "numerical answer requires two independent computations or computation plus source"
        return "research answer requires source evidence"

    async def _submit(
        self,
        state: _TaskState,
        task: GaiaTask,
        decision: CriticDecision,
        tools: _BudgetedToolClient,
        *,
        budget: StageBudget,
        summary_budget: StageBudget | None = None,
    ) -> TaskPrediction:
        if state.candidate is None:
            raise ValueError("cannot submit a missing candidate")
        evidence_ids = tuple(item.evidence_id for item in state.evidence)
        tools.start_scope(
            "submit",
            stage=budget.name,
            max_calls=budget.max_calls,
            deadline=budget.deadline,
        )
        self._audit(
            "submit_started",
            {"evidence_ids": list(evidence_ids)},
        )
        stage_started = self._now()
        with self._stage_span(
            "submit",
            {"stage": "submit", "evidence_count": len(evidence_ids)},
        ) as submit_span:
            if submit_span is not None:
                submit_span.set_summary(
                    lambda: self._stage_operational_summary(
                        summary_budget or budget,
                        tools,
                        stage_started,
                    )
                )
            await tools.call(
                "submit_answer",
                {"answer": state.candidate, "evidence_ids": list(evidence_ids)},
            )
            if submit_span is not None:
                submit_span.set_summary(
                    {
                        "decision": "submit",
                        "evidence_count": len(evidence_ids),
                        "candidate_present": True,
                        "tool_names": list(tools.scope_tool_names),
                        **self._stage_operational_summary(
                            summary_budget or budget,
                            tools,
                            stage_started,
                        ),
                    }
                )
        self._audit(
            "submit_completed",
            {"evidence_ids": list(evidence_ids)},
        )
        return TaskPrediction(
            task_id=task.task_id,
            level=task.level,
            answer=state.candidate,
            usage=state.usage,
            critic_verdict=decision.verdict,
            evidence_ids=evidence_ids,
        )

    def _now(self) -> float:
        if self._vote_stage_runtime is not None:
            return self._vote_stage_runtime.now()
        return asyncio.get_running_loop().time()

    def _stage_budgets(self, started: float) -> dict[str, StageBudget]:
        envelope = (
            self._vote_stage_runtime.envelope
            if self._vote_stage_runtime is not None
            else None
        ) or StageDeadlineEnvelope.for_config(self._config, started_at=started)
        return {
            "planner": StageBudget(
                "planner",
                envelope.planner_deadline,
                0,
                started_at=envelope.vote_started_at,
                budget_seconds=self._config.planner_timeout_seconds,
            ),
            "specialist": StageBudget(
                "specialist",
                envelope.specialist_deadline,
                self._config.specialist_max_tool_calls,
                started_at=envelope.planner_deadline,
                budget_seconds=self._config.specialist_timeout_seconds,
            ),
            "critic": StageBudget(
                "critic",
                envelope.critic_deadline,
                self._config.critic_max_tool_calls,
                started_at=envelope.specialist_deadline,
                budget_seconds=self._config.critic_timeout_seconds,
            ),
        }

    def _stage_timeout_seconds(self, stage: str) -> float:
        return {
            "planner": self._config.planner_timeout_seconds,
            "specialist": self._config.specialist_timeout_seconds,
            "critic": self._config.critic_timeout_seconds,
            "submit": self._config.critic_timeout_seconds,
        }[stage]

    async def _run_in_stage(
        self,
        budget: StageBudget,
        operation: Callable[[], Awaitable[Any]],
        stage_runtime: _StageRuntime | None = None,
        *,
        telemetry_stage: Literal["planner", "specialist", "critic", "submit"]
        | None = None,
    ) -> Any:
        remaining = None
        loop_deadline = None
        stage_deadline = None
        if budget.deadline is not None:
            remaining = budget.deadline - self._now()
            if remaining <= 0:
                raise StageBudgetExceeded(budget.name, "time")
            loop_deadline = asyncio.get_running_loop().time() + remaining
            stage_deadline = InvocationDeadline(budget.deadline, self._now)
        previous_stage = stage_runtime.active if stage_runtime is not None else None
        if stage_runtime is not None:
            stage_runtime.active = budget.name
        activity = (
            self._vote_stage_runtime.begin(telemetry_stage or budget.name)
            if self._vote_stage_runtime is not None
            else None
        )
        interrupted = False
        try:
            scope = (
                invocation_deadline_scope(stage_deadline)
                if stage_deadline is not None
                else nullcontext()
            )
            with scope:
                if loop_deadline is None:
                    result = await operation()
                else:
                    async with asyncio.timeout_at(loop_deadline):
                        # This is the final synchronous gate before the stage
                        # operation can invoke a provider or tool. The same
                        # absolute deadline remains in context for nested calls.
                        ensure_current_invocation_allowed()
                        result = await operation()
                    if self._now() >= budget.deadline:
                        raise StageBudgetExceeded(budget.name, "time")
            return result
        except asyncio.CancelledError:
            interrupted = True
            raise
        except StageBudgetExceeded:
            raise
        except InvocationDeadlineExceeded as error:
            if stage_deadline is None:
                raise
            raise StageBudgetExceeded(budget.name, "time") from error
        except TimeoutError as error:
            raise StageBudgetExceeded(budget.name, "time") from error
        finally:
            if stage_deadline is not None:
                # Context is copied into detached tasks. Closing the shared
                # object prevents those tasks from starting late invocations
                # after this stage has already returned.
                stage_deadline.close()
            if activity is not None:
                self._vote_stage_runtime.finish(activity)
            if stage_runtime is not None and not interrupted:
                stage_runtime.active = previous_stage

    def _audit_stage_budget_exhausted(
        self,
        error: StageBudgetExceeded,
        budget: StageBudget,
        tools: _BudgetedToolClient | None,
    ) -> None:
        self._audit(
            "stage_budget_exhausted",
            {
                "stage": error.stage,
                "reason": error.reason,
                "timeout_seconds": self._stage_timeout_seconds(error.stage),
                "max_calls": budget.max_calls,
                "logical_calls": (
                    tools.stage_logical_call_count(error.stage) if tools is not None else 0
                ),
            },
        )

    def _audit(self, event_type: str, payload: dict[str, Any]) -> None:
        self._audit_events.append(
            AuditEvent(
                sequence=len(self._audit_events) + 1,
                event_type=event_type,
                timestamp=datetime.now(UTC),
                payload=payload,
            )
        )

    def _stage_operational_summary(
        self,
        budget: StageBudget,
        tools: _BudgetedToolClient | None,
        started: float,
    ) -> dict[str, object]:
        """Return bounded operational state without graph inputs or checkpoint data."""

        budget_seconds = (
            budget.budget_seconds
            if budget.budget_seconds is not None
            else self._stage_timeout_seconds(budget.name)
        )
        now = self._now()
        elapsed = (
            self._vote_stage_runtime.elapsed_seconds(budget.name)
            if self._vote_stage_runtime is not None
            else _finite_nonnegative(now - started)
        )
        remaining = (
            max(0.0, budget.deadline - now)
            if budget.deadline is not None
            else max(0.0, budget_seconds - elapsed)
        )
        remaining_bucket = (
            "exhausted"
            if budget_seconds <= 0.0 or remaining == 0.0
            else "low"
            if remaining <= budget_seconds * 0.25
            else "ample"
        )
        return {
            "stage": budget.name,
            "budget_seconds": budget_seconds,
            "elapsed_seconds": elapsed,
            "remaining_bucket": remaining_bucket,
            "logical_call_count": (
                0
                if tools is None
                else tools.scope_logical_call_count
                if budget.name == "submit"
                else tools.stage_logical_call_count(budget.name)
            ),
            "global_call_count": tools.logical_call_count if tools is not None else 0,
            "executed_call_count": tools.executed_call_count if tools is not None else 0,
            "restored_call_count": tools.restored_call_count if tools is not None else 0,
        }

    def _stage_span(self, name: str, metadata: dict[str, object]) -> Any:
        if self._trace is None:
            return nullcontext()
        return self._trace.span(name, metadata)


def _evidence_repair_budget(
    critic_budget: StageBudget,
    *,
    now: float | None = None,
) -> StageBudget:
    if critic_budget.deadline is None:
        return critic_budget
    active_now = asyncio.get_running_loop().time() if now is None else now
    remaining = max(0.0, critic_budget.deadline - active_now)
    return StageBudget(
        name="critic",
        deadline=active_now + remaining / 2,
        max_calls=critic_budget.max_calls,
        started_at=critic_budget.started_at,
        budget_seconds=critic_budget.budget_seconds,
    )


def _evidence_repair_failure_category(error: Exception) -> str:
    if isinstance(error, (ProviderTransientExhaustedError, ConnectionError)):
        return "provider_transient"
    if isinstance(error, (StageBudgetExceeded, TimeoutError)):
        return "time"
    if isinstance(error, (StageProtocolError, TypeError, ValueError)):
        return "protocol"
    return "linker"


def _add_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )


def _finite_nonnegative(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(_MAX_OPERATIONAL_SECONDS, max(0.0, value))


def _numeric_value(answer: str | None) -> float | None:
    if answer is None:
        return None
    success, normalized = parse_number_with_status(answer)
    return normalized if success else None


def _normalize_for_voting(answer: str | None) -> str:
    if answer is None:
        return "string:none"
    numeric = _numeric_value(answer)
    if numeric is not None:
        canonical_numeric = 0.0 if numeric == 0 else numeric
        return f"number:{canonical_numeric.hex()}"
    if "," in answer or ";" in answer:
        parts = re.split(r"[,;]", answer)
        normalized_parts: list[str] = []
        for part in parts:
            numeric_part = _numeric_value(part)
            if numeric_part is not None:
                canonical_numeric = 0.0 if numeric_part == 0 else numeric_part
                normalized_parts.append(f"number:{canonical_numeric.hex()}")
            else:
                normalized_part = re.sub(r"\s", "", part).casefold()
                normalized_parts.append(f"string:{normalized_part}")
        return "list:" + "|".join(normalized_parts)
    normalized = re.sub(r"\s", "", answer).casefold()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return f"string:{normalized}"


def choose_majority(
    predictions: Sequence[TaskPrediction],
    votes: int,
) -> TaskPrediction:
    """Choose an early or final strict majority without consulting a reference answer."""

    if votes < 1 or votes % 2 == 0:
        raise ValueError("votes must be a positive odd integer")
    if not predictions:
        raise ValueError("predictions must not be empty")
    if len(predictions) > votes:
        raise ValueError("predictions cannot exceed configured votes")
    if len({prediction.task_id for prediction in predictions}) != 1:
        raise ValueError("predictions must share one task ID")
    if len({prediction.level for prediction in predictions}) != 1:
        raise ValueError("predictions must share one level")
    run_indexes = [prediction.run_index for prediction in predictions]
    if any(run_index < 0 or run_index >= votes for run_index in run_indexes):
        raise ValueError("prediction run index must be within configured votes")
    if len(set(run_indexes)) != len(run_indexes):
        raise ValueError("predictions must have unique run indexes")

    groups: dict[str, list[TaskPrediction]] = defaultdict(list)
    for prediction in predictions:
        groups[_normalize_for_voting(prediction.answer)].append(prediction)
    winner = max(
        groups.values(),
        key=lambda group: (len(group), -min(item.run_index for item in group)),
    )
    representative = min(winner, key=lambda item: item.run_index)
    if len(winner) <= votes // 2:
        first = min(predictions, key=lambda item: item.run_index)
        return first.model_copy(
            update={
                "error_category": TerminalErrorCategory.NO_CONSENSUS,
                "vote_count": 1,
            }
        )
    return representative.model_copy(update={"vote_count": len(winner)})
