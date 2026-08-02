from __future__ import annotations

import asyncio
import json
import re
import string
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from gaia_agent.config import RunConfig
from gaia_agent.mcp_client import ToolExecutionError
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
from gaia_agent.providers import ToolSpec


class Planner(Protocol):
    async def plan(self, task: GaiaTask) -> AgentPlan: ...


class Specialist(Protocol):
    async def run(
        self,
        task: GaiaTask,
        step: PlanStep,
        evidence: tuple[Evidence, ...],
        tools: Any,
    ) -> AgentResult: ...


class Critic(Protocol):
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
class _TaskState:
    task: GaiaTask
    plan: AgentPlan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    evidence_origins: dict[str, int] = field(default_factory=dict)
    candidate: str | None = None
    usage: Usage = field(default_factory=Usage)
    step_count: int = 0
    critic_verdict: str = "not_run"


class _BudgetedToolClient:
    def __init__(
        self,
        client: Any,
        *,
        max_calls: int,
        audit: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self._client = client
        self._max_calls = max_calls
        self._audit = audit
        self._signatures: set[str] = set()
        self.call_count = 0

    @property
    def tool_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._client.tool_specs)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        signature = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in self._signatures:
            raise _RepeatedToolCall(f"repeated tool call: {name}")
        if self.call_count >= self._max_calls:
            raise _ToolBudgetExhausted("global tool-call budget exhausted")
        self._signatures.add(signature)
        self.call_count += 1
        safe_payload = {
            "tool": name,
            "call_number": self.call_count,
            "argument_keys": sorted(arguments),
        }
        self._audit("tool_call_started", safe_payload)
        result = await self._client.call(name, arguments)
        self._audit(
            "tool_call_completed",
            {
                **safe_payload,
                "result_keys": sorted(result),
            },
        )
        return result


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
        self._audit_events: list[AuditEvent] = []
        self.correction_count = 0
        self.step_count = 0
        self.tool_call_count = 0

    @property
    def audit_events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit_events)

    async def run(self, task: GaiaTask) -> TaskPrediction:
        started = time.monotonic()
        self._audit_events = []
        self.correction_count = 0
        self.step_count = 0
        self.tool_call_count = 0
        state = _TaskState(task=task)
        execution_task = task.model_copy(update={"final_answer": ""})
        budgeted_tools: _BudgetedToolClient | None = None
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with self._tool_client_factory(execution_task) as raw_tools:
                    budgeted_tools = _BudgetedToolClient(
                        raw_tools,
                        max_calls=self._config.max_tool_calls,
                        audit=self._audit,
                    )
                    prediction = await self._execute(state, execution_task, budgeted_tools)
            self.tool_call_count = budgeted_tools.call_count
            return prediction.model_copy(
                update={"duration_seconds": time.monotonic() - started}
            )
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
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            category = TerminalErrorCategory.ATTACHMENT_ERROR
        except (ValueError, TypeError, KeyError):
            category = TerminalErrorCategory.MALFORMED_RESPONSE
        except ToolExecutionError:
            category = TerminalErrorCategory.RETRY_EXHAUSTED
        except Exception:  # noqa: BLE001 - task isolation requires a terminal result
            category = TerminalErrorCategory.RETRY_EXHAUSTED
        if budgeted_tools is not None:
            self.tool_call_count = budgeted_tools.call_count
        self.step_count = state.step_count
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
        )

    async def _execute(
        self,
        state: _TaskState,
        task: GaiaTask,
        tools: _BudgetedToolClient,
    ) -> TaskPrediction:
        self._audit("plan_started", {"task_id": task.task_id})
        plan = await self._planner.plan(task)
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

        for step in plan.steps:
            await self._act(state, task, step, tools, correction=False)

        decision = await self._verify(state, task, tools)
        decision = self._apply_quality_policy(state, decision)
        if decision.verdict != "approve":
            await self._correct_once(state, task, decision, tools)
            decision = await self._verify(state, task, tools)
            decision = self._apply_quality_policy(state, decision)

        if decision.verdict != "approve":
            raise _CorrectionUnavailable("critic rejected the candidate after correction")
        return await self._submit(state, task, decision, tools)

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
        self._audit(
            f"{event_prefix}_started",
            {
                "specialist": step.specialist.value,
                "step_number": state.step_count,
            },
        )
        result = await specialist.run(task, step, tuple(state.evidence), tools)
        self._record_result(state, step, result)
        self._audit(
            f"{event_prefix}_completed",
            {
                "specialist": step.specialist.value,
                "step_number": state.step_count,
                "evidence_ids": [item.evidence_id for item in result.evidence],
                "has_candidate": result.candidate_answer is not None,
            },
        )

    def _record_result(
        self,
        state: _TaskState,
        step: PlanStep,
        result: AgentResult,
    ) -> None:
        if result.specialist != step.specialist:
            raise ValueError("specialist result does not match dispatched plan step")
        existing_ids = set(state.evidence_origins)
        result_ids = [item.evidence_id for item in result.evidence]
        if any(not evidence_id.strip() for evidence_id in result_ids):
            raise ValueError("evidence IDs must be non-empty")
        if len(result_ids) != len(set(result_ids)) or existing_ids.intersection(result_ids):
            raise ValueError("evidence IDs must be globally unique")
        if any(item.kind == "source" and not item.source_url for item in result.evidence):
            raise ValueError("source evidence must include a URL")
        for item in result.evidence:
            state.evidence.append(item)
            state.evidence_origins[item.evidence_id] = state.step_count
        if result.candidate_answer is not None:
            candidate = result.candidate_answer.strip()
            if not candidate or "\n" in candidate:
                raise ValueError("candidate answer must use the concise GAIA format")
            state.candidate = candidate
        state.usage = _add_usage(state.usage, result.usage)

    async def _verify(
        self,
        state: _TaskState,
        task: GaiaTask,
        tools: _BudgetedToolClient,
    ) -> CriticDecision:
        self._audit(
            "verify_started",
            {
                "evidence_ids": [item.evidence_id for item in state.evidence],
                "has_candidate": state.candidate is not None,
            },
        )
        decision = await self._critic.verify(
            task,
            state.candidate,
            tuple(state.evidence),
            tools,
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
        if self._has_sufficient_evidence(state):
            if not decision.checked_evidence_ids:
                raise ValueError("critic approval must check evidence")
            return decision
        state.critic_verdict = "insufficient_evidence"
        self._audit(
            "quality_rejected",
            {"reason": self._evidence_gap(state)},
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
        self._audit(
            "correction_started",
            {"reason": decision.verdict},
        )
        if decision.verdict == "revise":
            corrected = (decision.corrected_answer or "").strip()
            if not corrected or "\n" in corrected:
                raise ValueError("critic correction must use the concise GAIA format")
            state.candidate = corrected
        else:
            target = self._correction_specialist(state)
            step = PlanStep(specialist=target, instruction=decision.feedback)
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

    def _has_sufficient_evidence(self, state: _TaskState) -> bool:
        if state.candidate is None:
            return False
        kinds = {item.kind for item in state.evidence}
        if _numeric_value(state.candidate) is not None:
            computation_origins = {
                state.evidence_origins[item.evidence_id]
                for item in state.evidence
                if item.kind == "computation"
            }
            return len(computation_origins) >= 2 or (
                bool(computation_origins) and "source" in kinds
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
        self._audit(
            "submit_started",
            {"evidence_ids": list(evidence_ids)},
        )
        await tools.call(
            "submit_answer",
            {"answer": state.candidate, "evidence_ids": list(evidence_ids)},
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


def _add_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )


def _numeric_value(answer: str | None) -> Decimal | None:
    if answer is None:
        return None
    stripped = answer.strip()
    if not stripped:
        return None
    has_numeric_marker = stripped.startswith("$") or stripped.endswith("%")
    thousands = bool(re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", stripped))
    plain_number = bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", stripped))
    if not (has_numeric_marker or thousands or plain_number):
        return None
    normalized = stripped.replace("$", "").replace("%", "").replace(",", "")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _normalize_for_voting(answer: str | None) -> str:
    if answer is None:
        return "string:none"
    numeric = _numeric_value(answer)
    if numeric is not None:
        return f"number:{numeric.normalize()}"
    if "," in answer or ";" in answer:
        parts = re.split(r"[,;]", answer)
        normalized_parts: list[str] = []
        for part in parts:
            numeric_part = _numeric_value(part)
            if numeric_part is not None:
                normalized_parts.append(f"number:{numeric_part.normalize()}")
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
    run_indexes = [prediction.run_index for prediction in predictions]
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
