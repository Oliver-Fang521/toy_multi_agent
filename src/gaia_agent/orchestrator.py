from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import string
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
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
    TaskPrediction,
    TerminalErrorCategory,
    Usage,
)
from gaia_agent.observability import Observability, TaskTrace
from gaia_agent.providers import ToolCallError, ToolSpec
from gaia_agent.reliability import ProviderTransientExhaustedError
from gaia_agent.research_notebook import ResearchNotebook
from gaia_agent.scoring import parse_number_with_status

_DEFAULT_STAGE_CONTEXT_MAX_BYTES = 24_576
_CONTEXT_BUDGET_PRESSURE_RATIO = 0.7


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


class _ToolBudgetExhausted(RuntimeError):
    pass


class _RepeatedToolCall(RuntimeError):
    pass


class _CorrectionUnavailable(RuntimeError):
    pass


class _AttachmentUnavailable(RuntimeError):
    pass


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
    ) -> None:
        self._client = client
        self._max_calls = max_calls
        self._audit = audit
        self._notebook = notebook
        self._scope = "unscoped"
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
        return await self._client.read_attachment_bytes(path)

    def start_scope(self, scope: str) -> None:
        if not scope.strip():
            raise ValueError("tool-call scope must be non-empty")
        self._scope = scope
        self._scope_tool_names = set()

    @property
    def scope_tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._scope_tool_names))

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        signature = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        if self.call_count >= self._max_calls:
            raise _ToolBudgetExhausted("global tool-call budget exhausted")
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
        self._audit("tool_call_started", safe_payload)
        self.executed_call_count += 1
        try:
            result = await self._client.call(name, arguments)
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


class _GraphState(TypedDict, total=False):
    task: Required[GaiaTask]
    runtime: Required[_TaskState]
    tools: Required[_BudgetedToolClient]
    next_step_index: int
    soft_step_failures: int
    decision: CriticDecision
    prediction: TaskPrediction


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
        self._trace: TaskTrace | None = None
        self._audit_events: list[AuditEvent] = []
        self.correction_count = 0
        self.step_count = 0
        self.tool_call_count = 0
        self._graph = self._build_graph()

    @property
    def audit_events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit_events)

    async def run(self, task: GaiaTask) -> TaskPrediction:
        started = time.monotonic()
        self._audit_events = []
        self.correction_count = 0
        self.step_count = 0
        self.tool_call_count = 0
        self._trace = self._observability.current_trace()
        state = _TaskState()
        execution_task = task.model_copy(update={"final_answer": ""})
        budgeted_tools: _BudgetedToolClient | None = None
        raw_tools: Any | None = None
        prediction: TaskPrediction | None = None
        category: TerminalErrorCategory | None = None
        provider_exhaustion: ProviderTransientExhaustedError | None = None
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with self._tool_client_factory(execution_task) as active_tools:
                    raw_tools = active_tools
                    budgeted_tools = _BudgetedToolClient(
                        active_tools,
                        max_calls=self._config.max_tool_calls,
                        audit=self._audit,
                        notebook=self._notebook,
                    )
                    prediction = await self._execute(state, execution_task, budgeted_tools)
        except ProviderTransientExhaustedError as error:
            category = TerminalErrorCategory.RETRY_EXHAUSTED
            provider_exhaustion = error
        except TimeoutError:
            category = TerminalErrorCategory.TIMEOUT
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
            if raw_tools is not None:
                state.usage = _add_usage(
                    state.usage,
                    getattr(raw_tools, "usage", Usage()),
                )
        if budgeted_tools is not None:
            self.tool_call_count = budgeted_tools.call_count
        self.step_count = state.step_count
        if prediction is not None:
            return prediction.model_copy(
                update={
                    "duration_seconds": time.monotonic() - started,
                    "usage": state.usage,
                }
            )
        assert category is not None
        self._audit(
            "terminal_failure",
            {"category": category.value, "critic_verdict": state.critic_verdict},
        )
        return TaskPrediction(
            task_id=task.task_id,
            level=task.level,
            answer=None,
            duration_seconds=time.monotonic() - started,
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

    async def _execute(
        self,
        state: _TaskState,
        task: GaiaTask,
        tools: _BudgetedToolClient,
    ) -> TaskPrediction:
        result = await self._graph.ainvoke(
            {
                "task": task,
                "runtime": state,
                "tools": tools,
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
        graph.add_node("plan", self._graph_plan)
        graph.add_node("act", self._graph_act)
        graph.add_node("verify", self._graph_verify)
        graph.add_node("correct", self._graph_correct)
        graph.add_node("reject", self._graph_reject)
        graph.add_node("submit", self._graph_submit)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "act")
        graph.add_conditional_edges(
            "act",
            self._route_after_act,
            {"act": "act", "verify": "verify"},
        )
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
        return max(25, self._config.max_steps + 8)

    async def _graph_plan(self, graph_state: _GraphState) -> dict[str, Any]:
        state = graph_state["runtime"]
        task = graph_state["task"]
        self._audit("plan_started", {"task_id": task.task_id})
        with self._stage_span("planner", {"stage": "planner"}) as planner_span:
            try:
                plan = await self._planner.plan(task)
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
        return {"runtime": state, "next_step_index": 0}

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
        try:
            await self._act(
                state,
                task,
                state.plan.steps[step_index],
                tools,
                correction=False,
            )
        except StageProtocolError as error:
            if (
                not error.repairable
                or not error.repair_attempted
                or graph_state.get("soft_step_failures", 0) >= 1
                or state.candidate is None
                or not state.evidence
            ):
                raise
            self._audit(
                "soft_step_failure",
                {"step_number": state.step_count, "code": error.code.value},
            )
            return {
                "runtime": state,
                "next_step_index": len(state.plan.steps),
                "soft_step_failures": graph_state.get("soft_step_failures", 0) + 1,
            }
        return {"runtime": state, "next_step_index": step_index + 1}

    def _route_after_act(self, graph_state: _GraphState) -> Literal["act", "verify"]:
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
                return "verify"
            if _ready_for_critic(state):
                self._audit(
                    "plan_early_stop",
                    {
                        "completed_steps": state.step_count,
                        "remaining_steps": len(state.plan.steps) - step_index,
                    },
                )
                return "verify"
            return "act"
        if step_index == len(state.plan.steps):
            return "verify"
        raise ValueError("graph act router received an invalid plan index")

    async def _graph_verify(self, graph_state: _GraphState) -> dict[str, Any]:
        state = graph_state["runtime"]
        task = graph_state["task"]
        tools = graph_state["tools"]
        decision = await self._verify(state, task, tools)
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
            return "submit"
        if not self.correction_count:
            return "correct"
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
        )
        return {"runtime": state}

    async def _graph_reject(self, graph_state: _GraphState) -> dict[str, Any]:
        raise _CorrectionUnavailable("critic rejected the candidate after correction")

    async def _graph_submit(self, graph_state: _GraphState) -> dict[str, Any]:
        decision = graph_state.get("decision")
        if decision is None:
            raise ValueError("graph submit node requires a critic decision")
        prediction = await self._submit(
            graph_state["runtime"],
            graph_state["task"],
            decision,
            graph_state["tools"],
        )
        return {"prediction": prediction}

    async def _act(
        self,
        state: _TaskState,
        task: GaiaTask,
        step: PlanStep,
        tools: _BudgetedToolClient,
        *,
        correction: bool,
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
        tools.start_scope(scope)
        self._audit(
            f"{event_prefix}_started",
            {
                "specialist": step.specialist.value,
                "step_number": state.step_count,
            },
        )
        with self._stage_span(
            "specialist",
            {
                "stage": "act",
                "specialist": step.specialist.value,
                "step_number": state.step_count,
            },
        ) as specialist_span:
            try:
                result = await specialist.run(task, step, tuple(state.evidence), tools)
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
    ) -> CriticDecision:
        state.verify_count += 1
        tools.start_scope(f"verify:{state.verify_count}")
        self._audit(
            "verify_started",
            {
                "evidence_ids": [item.evidence_id for item in state.evidence],
                "has_candidate": state.candidate is not None,
            },
        )
        with self._stage_span(
            "critic",
            {
                "stage": "verify",
                "evidence_count": len(state.evidence),
                "has_candidate": state.candidate is not None,
            },
        ) as critic_span:
            try:
                decision = await self._critic.verify(
                    task,
                    state.candidate,
                    tuple(state.evidence),
                    tools,
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
    ) -> None:
        if self.correction_count:
            raise _CorrectionUnavailable("only one correction cycle is allowed")
        self.correction_count = 1
        tools.start_scope(f"correction:{self.correction_count}")
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
                await self._act(state, task, step, tools, correction=True)
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
    ) -> TaskPrediction:
        if state.candidate is None:
            raise ValueError("cannot submit a missing candidate")
        evidence_ids = tuple(item.evidence_id for item in state.evidence)
        tools.start_scope("submit")
        self._audit(
            "submit_started",
            {"evidence_ids": list(evidence_ids)},
        )
        with self._stage_span(
            "submit",
            {"stage": "submit", "evidence_count": len(evidence_ids)},
        ) as submit_span:
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

    def _audit(self, event_type: str, payload: dict[str, Any]) -> None:
        self._audit_events.append(
            AuditEvent(
                sequence=len(self._audit_events) + 1,
                event_type=event_type,
                timestamp=datetime.now(UTC),
                payload=payload,
            )
        )

    def _stage_span(self, name: str, metadata: dict[str, object]) -> Any:
        if self._trace is None:
            return nullcontext()
        return self._trace.span(name, metadata)


def _add_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )


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
