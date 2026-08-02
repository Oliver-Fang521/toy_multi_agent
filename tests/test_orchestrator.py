import asyncio
from collections import deque
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest

from gaia_agent.config import RunConfig
from gaia_agent.mcp_client import ToolExecutionError
from gaia_agent.models import (
    AgentPlan,
    AgentResult,
    CriticDecision,
    Evidence,
    GaiaTask,
    PlanStep,
    SpecialistName,
    TaskPrediction,
    TerminalErrorCategory,
)
from gaia_agent.orchestrator import Orchestrator, choose_majority
from gaia_agent.providers import ToolSpec


class FakePlanner:
    def __init__(
        self,
        calls: list[str],
        plan: AgentPlan,
        *,
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        self.calls = calls
        self.result = plan
        self.error = error
        self.delay = delay
        self.tasks: list[GaiaTask] = []

    async def plan(self, task: GaiaTask) -> AgentPlan:
        self.calls.append("planner")
        self.tasks.append(task)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result


class FakeSpecialist:
    def __init__(
        self,
        name: SpecialistName,
        calls: list[str],
        results: list[AgentResult],
        *,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.results = deque(results)
        self.tool_calls = tool_calls or []
        self.error = error
        self.tasks: list[GaiaTask] = []
        self.prior_evidence: list[tuple[Evidence, ...]] = []

    async def run(
        self,
        task: GaiaTask,
        step: PlanStep,
        evidence: tuple[Evidence, ...],
        tools: Any,
    ) -> AgentResult:
        self.calls.append(self.name.value)
        self.tasks.append(task)
        self.prior_evidence.append(evidence)
        if self.error is not None:
            raise self.error
        for name, arguments in self.tool_calls:
            await tools.call(name, arguments)
        return self.results.popleft()


class FakeCritic:
    def __init__(
        self,
        calls: list[str],
        decisions: list[CriticDecision],
        *,
        error: Exception | None = None,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.calls = calls
        self.decisions = deque(decisions)
        self.error = error
        self.tool_calls = tool_calls or []
        self.tasks: list[GaiaTask] = []
        self.candidates: list[str | None] = []

    async def verify(
        self,
        task: GaiaTask,
        candidate: str | None,
        evidence: tuple[Evidence, ...],
        tools: Any,
    ) -> CriticDecision:
        self.calls.append("critic")
        self.tasks.append(task)
        self.candidates.append(candidate)
        if self.error is not None:
            raise self.error
        for name, arguments in self.tool_calls:
            await tools.call(name, arguments)
        return self.decisions.popleft()


class FakeTools(AbstractAsyncContextManager["FakeTools"]):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.call_arguments: list[tuple[str, dict[str, Any]]] = []
        self.tool_specs = tuple(
            ToolSpec(
                name=name,
                description=name,
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            )
            for name in (
                "web_search",
                "read_webpage",
                "execute_python",
                "read_attachment",
                "submit_answer",
            )
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        self.call_arguments.append((name, arguments))
        return {"accepted": True} if name == "submit_answer" else {"result": "observable"}


class FakeToolFactory:
    def __init__(self, tools: FakeTools) -> None:
        self.tools = tools
        self.tasks: list[GaiaTask] = []

    def __call__(self, task: GaiaTask) -> FakeTools:
        self.tasks.append(task)
        return self.tools


def _config(**overrides: Any) -> RunConfig:
    values: dict[str, Any] = {
        "openai_api_key": "sk-test",
        "hf_token": "hf-test",
        "sample_size": 3,
        "max_steps": 8,
        "max_tool_calls": 24,
        "timeout_seconds": 1,
    }
    values.update(overrides)
    return RunConfig(
        **values,
    )


def _task(
    *,
    question: str = "Who discovered the synthetic element?",
    attachment_path: Path | None = None,
) -> GaiaTask:
    return GaiaTask(
        task_id="orchestration-task",
        question=question,
        level=1,
        final_answer="Hidden Gold Answer",
        attachment_path=attachment_path,
    )


def _plan(*specialists: SpecialistName) -> AgentPlan:
    return AgentPlan(
        restatement="Gather and verify observable evidence",
        steps=tuple(
            PlanStep(specialist=specialist, instruction=f"Run {specialist.value}")
            for specialist in specialists
        ),
    )


def _evidence(
    evidence_id: str,
    kind: str,
    *,
    summary: str = "Observable support",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        kind=kind,
        summary=summary,
        source_url="https://primary.example/fact" if kind == "source" else None,
    )


def _result(
    specialist: SpecialistName,
    evidence: tuple[Evidence, ...],
    candidate: str | None,
) -> AgentResult:
    return AgentResult(
        specialist=specialist,
        evidence=evidence,
        candidate_answer=candidate,
    )


def _decision(
    verdict: str,
    checked_ids: tuple[str, ...],
    *,
    corrected_answer: str | None = None,
) -> CriticDecision:
    return CriticDecision(
        verdict=verdict,
        feedback=f"critic {verdict}",
        checked_evidence_ids=checked_ids,
        corrected_answer=corrected_answer,
    )


def _orchestrator(
    *,
    planner: FakePlanner,
    specialists: dict[SpecialistName, FakeSpecialist],
    critic: FakeCritic,
    tools: FakeTools,
    config: RunConfig | None = None,
) -> Orchestrator:
    return Orchestrator(
        planner=planner,
        specialists=specialists,
        critic=critic,
        tool_client_factory=FakeToolFactory(tools),
        config=config or _config(),
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_plan_act_verify_submit() -> None:
    calls: list[str] = []
    source = _evidence("source-1", "source")
    planner = FakePlanner(calls, _plan(SpecialistName.LOOKUP))
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [_result(SpecialistName.LOOKUP, (source,), "Marie Curie")],
    )
    critic = FakeCritic(calls, [_decision("approve", ("source-1",))])
    tools = FakeTools(calls)

    result = await _orchestrator(
        planner=planner,
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=tools,
    ).run(_task())

    assert calls == ["planner", "lookup", "critic", "submit_answer"]
    assert result.answer == "Marie Curie"
    assert result.critic_verdict == "approve"
    assert result.error_category is None
    assert tools.call_arguments == [
        ("submit_answer", {"answer": "Marie Curie", "evidence_ids": ["source-1"]})
    ]


@pytest.mark.asyncio
async def test_orchestrator_routes_every_specialist_in_plan_order(tmp_path: Path) -> None:
    calls: list[str] = []
    roles = tuple(SpecialistName)
    results = {
        SpecialistName.LOOKUP: _result(
            SpecialistName.LOOKUP, (_evidence("source-1", "source"),), None
        ),
        SpecialistName.COMPUTATION: _result(
            SpecialistName.COMPUTATION,
            (_evidence("computation-1", "computation"),),
            None,
        ),
        SpecialistName.FILE: _result(
            SpecialistName.FILE, (_evidence("file-1", "file"),), None
        ),
        SpecialistName.VISUAL: _result(
            SpecialistName.VISUAL,
            (_evidence("visual-1", "visual"),),
            "Marie Curie",
        ),
    }
    specialists = {
        role: FakeSpecialist(role, calls, [results[role]]) for role in roles
    }
    critic = FakeCritic(calls, [_decision("approve", ("source-1", "visual-1"))])
    tools = FakeTools(calls)

    attachment = tmp_path / "registered.png"
    attachment.write_bytes(b"synthetic")
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(*roles)),
        specialists=specialists,
        critic=critic,
        tools=tools,
    ).run(_task(attachment_path=attachment))

    assert calls == ["planner", *[role.value for role in roles], "critic", "submit_answer"]
    assert result.evidence_ids == (
        "source-1",
        "computation-1",
        "file-1",
        "visual-1",
    )


@pytest.mark.asyncio
async def test_missing_specialist_mapping_is_a_malformed_terminal_result() -> None:
    calls: list[str] = []
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={},
        critic=FakeCritic(calls, []),
        tools=FakeTools(calls),
    ).run(_task())

    assert result.answer is None
    assert result.error_category == TerminalErrorCategory.MALFORMED_RESPONSE
    assert calls == ["planner"]


@pytest.mark.asyncio
async def test_plan_over_global_step_budget_is_rejected_without_partial_execution() -> None:
    calls: list[str] = []
    result = await _orchestrator(
        planner=FakePlanner(
            calls,
            _plan(SpecialistName.LOOKUP, SpecialistName.COMPUTATION),
        ),
        specialists={},
        critic=FakeCritic(calls, []),
        tools=FakeTools(calls),
        config=_config(max_steps=1),
    ).run(_task())

    assert result.error_category == TerminalErrorCategory.MALFORMED_RESPONSE
    assert calls == ["planner"]


@pytest.mark.asyncio
async def test_submit_cannot_bypass_global_tool_call_budget() -> None:
    calls: list[str] = []
    source = _evidence("source-1", "source")
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [_result(SpecialistName.LOOKUP, (source,), "Marie Curie")],
        tool_calls=[("web_search", {"query": "synthetic element"})],
    )
    tools = FakeTools(calls)
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=FakeCritic(calls, [_decision("approve", ("source-1",))]),
        tools=tools,
        config=_config(max_tool_calls=1),
    ).run(_task())

    assert result.answer is None
    assert result.error_category == TerminalErrorCategory.TOOL_BUDGET_EXHAUSTED
    assert tools.call_arguments == [("web_search", {"query": "synthetic element"})]


@pytest.mark.asyncio
async def test_identical_tool_call_across_roles_stops_as_a_loop() -> None:
    calls: list[str] = []
    repeated = [("web_search", {"query": "same query"})]
    source = _evidence("source-1", "source")
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [_result(SpecialistName.LOOKUP, (source,), "Marie Curie")],
        tool_calls=repeated,
    )
    critic = FakeCritic(
        calls,
        [_decision("approve", ("source-1",))],
        tool_calls=repeated,
    )
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=FakeTools(calls),
    ).run(_task())

    assert result.error_category == TerminalErrorCategory.LOOP_DETECTED
    assert "submit_answer" not in calls


@pytest.mark.asyncio
async def test_numerical_answer_requires_independent_double_check() -> None:
    calls: list[str] = []
    first = _result(
        SpecialistName.COMPUTATION,
        (_evidence("computation-1", "computation"),),
        "42",
    )
    second = _result(
        SpecialistName.COMPUTATION,
        (_evidence("computation-2", "computation"),),
        "42",
    )
    computation = FakeSpecialist(
        SpecialistName.COMPUTATION,
        calls,
        [first, second],
    )
    critic = FakeCritic(
        calls,
        [
            _decision("approve", ("computation-1",)),
            _decision("approve", ("computation-1", "computation-2")),
        ],
    )
    orchestrator = _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.COMPUTATION)),
        specialists={SpecialistName.COMPUTATION: computation},
        critic=critic,
        tools=FakeTools(calls),
    )

    result = await orchestrator.run(_task(question="What is six times seven?"))

    assert calls == ["planner", "computation", "critic", "computation", "critic", "submit_answer"]
    assert result.answer == "42"
    assert orchestrator.correction_count == 1
    assert computation.prior_evidence[1][0].evidence_id == "computation-1"


@pytest.mark.asyncio
async def test_research_answer_requires_source_evidence_before_submission() -> None:
    calls: list[str] = []
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [
            _result(SpecialistName.LOOKUP, (), "Marie Curie"),
            _result(
                SpecialistName.LOOKUP,
                (_evidence("source-1", "source"),),
                "Marie Curie",
            ),
        ],
    )
    critic = FakeCritic(
        calls,
        [_decision("approve", ()), _decision("approve", ("source-1",))],
    )
    orchestrator = _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=FakeTools(calls),
    )

    result = await orchestrator.run(_task())

    assert result.answer == "Marie Curie"
    assert calls.count("lookup") == 2
    assert calls.count("critic") == 2
    assert orchestrator.correction_count == 1


@pytest.mark.asyncio
async def test_critic_revision_is_reverified_before_corrected_answer_submission() -> None:
    calls: list[str] = []
    source = _evidence("source-1", "source", summary="The evidence contradicts the candidate")
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [_result(SpecialistName.LOOKUP, (source,), "Polonium")],
    )
    critic = FakeCritic(
        calls,
        [
            _decision("revise", ("source-1",), corrected_answer="Radium"),
            _decision("approve", ("source-1",)),
        ],
    )
    tools = FakeTools(calls)
    orchestrator = _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=tools,
    )

    result = await orchestrator.run(_task())

    assert critic.candidates == ["Polonium", "Radium"]
    assert result.answer == "Radium"
    assert orchestrator.correction_count == 1
    assert calls == ["planner", "lookup", "critic", "critic", "submit_answer"]


@pytest.mark.asyncio
async def test_orchestrator_allows_only_one_targeted_correction() -> None:
    calls: list[str] = []
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [
            _result(SpecialistName.LOOKUP, (), "Marie Curie"),
            _result(SpecialistName.LOOKUP, (), "Marie Curie"),
        ],
    )
    critic = FakeCritic(
        calls,
        [_decision("insufficient_evidence", ()), _decision("insufficient_evidence", ())],
    )
    orchestrator = _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=FakeTools(calls),
    )

    result = await orchestrator.run(_task())

    assert result.error_category == TerminalErrorCategory.CRITIC_REJECTION
    assert orchestrator.correction_count == 1
    assert calls == ["planner", "lookup", "critic", "lookup", "critic"]


@pytest.mark.asyncio
async def test_duplicate_evidence_ids_are_rejected_across_specialists() -> None:
    calls: list[str] = []
    duplicate_source = _evidence("global-1", "source")
    duplicate_computation = _evidence("global-1", "computation")
    result = await _orchestrator(
        planner=FakePlanner(
            calls,
            _plan(SpecialistName.LOOKUP, SpecialistName.COMPUTATION),
        ),
        specialists={
            SpecialistName.LOOKUP: FakeSpecialist(
                SpecialistName.LOOKUP,
                calls,
                [_result(SpecialistName.LOOKUP, (duplicate_source,), None)],
            ),
            SpecialistName.COMPUTATION: FakeSpecialist(
                SpecialistName.COMPUTATION,
                calls,
                [
                    _result(
                        SpecialistName.COMPUTATION,
                        (duplicate_computation,),
                        "answer",
                    )
                ],
            ),
        },
        critic=FakeCritic(calls, []),
        tools=FakeTools(calls),
    ).run(_task())

    assert result.error_category == TerminalErrorCategory.MALFORMED_RESPONSE
    assert calls == ["planner", "lookup", "computation"]


@pytest.mark.asyncio
async def test_single_deadline_covers_the_complete_state_machine() -> None:
    calls: list[str] = []
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP), delay=0.05),
        specialists={},
        critic=FakeCritic(calls, []),
        tools=FakeTools(calls),
        config=_config(timeout_seconds=0.01),
    ).run(_task())

    assert result.error_category == TerminalErrorCategory.TIMEOUT
    assert calls == ["planner"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ValueError("bad structured response"), TerminalErrorCategory.MALFORMED_RESPONSE),
        (FileNotFoundError("attachment absent"), TerminalErrorCategory.ATTACHMENT_ERROR),
        (ToolExecutionError("server unavailable"), TerminalErrorCategory.RETRY_EXHAUSTED),
    ],
)
async def test_exceptions_map_to_stable_terminal_categories(
    error: Exception,
    expected: TerminalErrorCategory,
) -> None:
    calls: list[str] = []
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP), error=error),
        specialists={},
        critic=FakeCritic(calls, []),
        tools=FakeTools(calls),
    ).run(_task())

    assert result.error_category == expected
    assert result.answer is None


@pytest.mark.asyncio
async def test_attachment_specialist_without_registered_file_is_an_attachment_error() -> None:
    calls: list[str] = []
    visual = FakeSpecialist(SpecialistName.VISUAL, calls, [])

    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.VISUAL)),
        specialists={SpecialistName.VISUAL: visual},
        critic=FakeCritic(calls, []),
        tools=FakeTools(calls),
    ).run(_task())

    assert result.error_category == TerminalErrorCategory.ATTACHMENT_ERROR
    assert calls == ["planner"]


@pytest.mark.asyncio
async def test_audit_is_ordered_and_excludes_question_gold_and_raw_errors() -> None:
    calls: list[str] = []
    source = _evidence("source-1", "source")
    planner = FakePlanner(calls, _plan(SpecialistName.LOOKUP))
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [_result(SpecialistName.LOOKUP, (source,), "Marie Curie")],
    )
    critic = FakeCritic(calls, [_decision("approve", ("source-1",))])
    tools = FakeTools(calls)
    orchestrator = _orchestrator(
        planner=planner,
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=tools,
    )

    await orchestrator.run(_task())

    audit = orchestrator.audit_events
    assert [event.sequence for event in audit] == list(range(1, len(audit) + 1))
    assert [event.event_type for event in audit] == [
        "plan_started",
        "plan_completed",
        "act_started",
        "act_completed",
        "verify_started",
        "verify_completed",
        "submit_started",
        "tool_call_started",
        "tool_call_completed",
        "submit_completed",
    ]
    assert all(isinstance(event.timestamp, datetime) for event in audit)
    assert "Hidden Gold Answer" not in repr(audit)
    assert "Who discovered the synthetic element?" not in repr(audit)
    assert all(received.final_answer == "" for received in [*planner.tasks, *lookup.tasks, *critic.tasks])


def _prediction(answer: str | None, run_index: int) -> TaskPrediction:
    return TaskPrediction(
        task_id="vote-task",
        level=2,
        answer=answer,
        critic_verdict="approve",
        run_index=run_index,
    )


def test_choose_majority_uses_scorer_compatible_normalization_and_first_winner() -> None:
    first = _prediction("$1,234", 0)
    second = _prediction("1234.0", 1)
    third = _prediction("999", 2)

    winner = choose_majority([first, second, third], votes=3)

    assert winner.answer == "$1,234"
    assert winner.run_index == 0
    assert winner.vote_count == 2
    assert winner.error_category is None


def test_choose_majority_returns_typed_no_consensus_without_gold() -> None:
    predictions = [_prediction("A", 0), _prediction("B", 1), _prediction("C", 2)]

    result = choose_majority(predictions, votes=3)

    assert result.answer == "A"
    assert result.error_category is TerminalErrorCategory.NO_CONSENSUS
    assert result.vote_count == 1


@pytest.mark.parametrize(
    ("predictions", "votes"),
    [([], 1), ([_prediction("A", 0)], 2), ([_prediction("A", 0)] * 4, 3)],
)
def test_choose_majority_rejects_invalid_vote_bounds(
    predictions: list[TaskPrediction],
    votes: int,
) -> None:
    with pytest.raises(ValueError, match="votes|predictions"):
        choose_majority(predictions, votes=votes)
