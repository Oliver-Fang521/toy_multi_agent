import asyncio
import os
from collections import deque
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest

from gaia_agent.config import RunConfig
from gaia_agent.mcp_client import AttachmentPolicyError, MCPToolClient, ToolExecutionError
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
    Usage,
)
from gaia_agent.orchestrator import (
    Orchestrator,
    _BudgetedToolClient,
    _ToolBudgetExhausted,
    choose_majority,
)
from gaia_agent.providers import ToolSpec


class FakePlanner:
    def __init__(
        self,
        calls: list[str],
        plan: AgentPlan,
        *,
        error: Exception | None = None,
        delay: float = 0,
        usage: Usage | None = None,
    ) -> None:
        self.calls = calls
        self.result = plan
        self.error = error
        self.delay = delay
        self.invocation_usage = usage or Usage()
        self.last_usage = Usage()
        self.tasks: list[GaiaTask] = []

    async def plan(self, task: GaiaTask) -> AgentPlan:
        self.last_usage = self.invocation_usage
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
        usage_on_error: Usage | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.results = deque(results)
        self.tool_calls = tool_calls or []
        self.error = error
        self.usage_on_error = usage_on_error or Usage()
        self.last_usage = Usage()
        self.tasks: list[GaiaTask] = []
        self.prior_evidence: list[tuple[Evidence, ...]] = []

    async def run(
        self,
        task: GaiaTask,
        step: PlanStep,
        evidence: tuple[Evidence, ...],
        tools: Any,
    ) -> AgentResult:
        self.last_usage = Usage()
        self.calls.append(self.name.value)
        self.tasks.append(task)
        self.prior_evidence.append(evidence)
        for name, arguments in self.tool_calls:
            await tools.call(name, arguments)
        if self.error is not None:
            self.last_usage = self.usage_on_error
            raise self.error
        result = self.results.popleft()
        self.last_usage = result.usage
        return result


class FakeCritic:
    def __init__(
        self,
        calls: list[str],
        decisions: list[CriticDecision],
        *,
        error: Exception | None = None,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
        usages: list[Usage] | None = None,
    ) -> None:
        self.calls = calls
        self.decisions = deque(decisions)
        self.error = error
        self.tool_calls = tool_calls or []
        self.usages = deque(usages or [Usage() for _ in decisions])
        self.last_usage = Usage()
        self.tasks: list[GaiaTask] = []
        self.candidates: list[str | None] = []
        self.evidence_seen: list[tuple[Evidence, ...]] = []

    async def verify(
        self,
        task: GaiaTask,
        candidate: str | None,
        evidence: tuple[Evidence, ...],
        tools: Any,
    ) -> CriticDecision:
        self.last_usage = self.usages.popleft()
        self.calls.append("critic")
        self.tasks.append(task)
        self.candidates.append(candidate)
        self.evidence_seen.append(evidence)
        if self.error is not None:
            raise self.error
        for name, arguments in self.tool_calls:
            await tools.call(name, arguments)
        return self.decisions.popleft()


class FakeTools(AbstractAsyncContextManager["FakeTools"]):
    def __init__(self, calls: list[str], *, usage_by_tool: dict[str, Usage] | None = None) -> None:
        self.calls = calls
        self.call_arguments: list[tuple[str, dict[str, Any]]] = []
        self.attachment_reads: list[Path] = []
        self.usage = Usage()
        self.usage_by_tool = usage_by_tool or {}
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
        charged = self.usage_by_tool.get(name, Usage())
        self.usage = Usage(
            input_tokens=self.usage.input_tokens + charged.input_tokens,
            output_tokens=self.usage.output_tokens + charged.output_tokens,
            reasoning_tokens=self.usage.reasoning_tokens + charged.reasoning_tokens,
            total_tokens=self.usage.total_tokens + charged.total_tokens,
        )
        return {"accepted": True} if name == "submit_answer" else {"result": "observable"}

    async def read_attachment_bytes(self, path: Path) -> bytes:
        self.attachment_reads.append(path)
        return path.read_bytes()


class FakeToolFactory:
    def __init__(self, tools: FakeTools) -> None:
        self.tools = tools
        self.tasks: list[GaiaTask] = []

    def __call__(self, task: GaiaTask) -> FakeTools:
        self.tasks.append(task)
        return self.tools


@pytest.mark.asyncio
async def test_budgeted_tool_client_audits_recoverable_failure_without_message() -> None:
    secret = "sk-private-tool-failure"

    class FailingTools:
        tool_specs: tuple[ToolSpec, ...] = ()

        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise ToolExecutionError(secret)

    audit: list[tuple[str, dict[str, Any]]] = []
    client = _BudgetedToolClient(
        FailingTools(),
        max_calls=2,
        audit=lambda event_type, payload: audit.append((event_type, payload)),
    )

    with pytest.raises(ToolExecutionError):
        await client.call(
            "execute_python",
            {"instruction": "private instruction", "paths": ["private.mp3"]},
        )

    assert client.call_count == 1
    assert audit == [
        (
            "tool_call_started",
            {
                "tool": "execute_python",
                "call_number": 1,
                "argument_keys": ["instruction", "paths"],
            },
        ),
        (
            "tool_call_failed",
            {
                "tool": "execute_python",
                "call_number": 1,
                "argument_keys": ["instruction", "paths"],
                "category": "tool_error",
            },
        ),
    ]
    assert secret not in repr(audit)
    assert "private instruction" not in repr(audit)
    assert "private.mp3" not in repr(audit)


@pytest.mark.asyncio
async def test_budgeted_tool_client_allows_same_transcription_in_a_later_scope() -> None:
    tools = FakeTools([])
    client = _BudgetedToolClient(tools, max_calls=3, audit=lambda *_: None)
    arguments = {"path": "sample.mp3"}

    client.start_scope("act:file:1")
    await client.call("transcribe_audio", arguments)
    client.start_scope("verify:1")
    await client.call("transcribe_audio", arguments)

    assert client.call_count == 2
    assert tools.call_arguments == [
        ("transcribe_audio", arguments),
        ("transcribe_audio", arguments),
    ]


@pytest.mark.asyncio
async def test_budgeted_tool_client_reuses_one_audio_result_without_double_charging() -> None:
    class AudioTools:
        tool_specs: tuple[ToolSpec, ...] = ()

        def __init__(self) -> None:
            self.calls = 0
            self.usage = Usage()

        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            self.usage = Usage(input_tokens=11, output_tokens=4, total_tokens=15)
            return {"text": "observable transcript"}

    backend = AudioTools()
    audit: list[tuple[str, dict[str, Any]]] = []
    client = _BudgetedToolClient(
        backend,
        max_calls=3,
        audit=lambda event_type, payload: audit.append((event_type, payload)),
    )
    client.start_scope("act:file:1")
    arguments = {"path": "sample.mp3"}

    first = await client.call("transcribe_audio", arguments)
    second = await client.call("transcribe_audio", arguments)

    assert first == second == {"text": "observable transcript"}
    assert backend.calls == 1
    assert client.call_count == 2
    assert backend.usage == Usage(input_tokens=11, output_tokens=4, total_tokens=15)
    assert audit[-1] == (
        "tool_call_reused",
        {
            "tool": "transcribe_audio",
            "call_number": 2,
            "argument_keys": ["path"],
            "result_keys": ["text"],
        },
    )
    assert "sample.mp3" not in repr(audit)
    assert "observable transcript" not in repr(audit)


@pytest.mark.asyncio
async def test_budgeted_tool_client_returns_a_safe_copy_of_cached_result() -> None:
    class MutableTools:
        tool_specs: tuple[ToolSpec, ...] = ()

        def __init__(self) -> None:
            self.calls = 0

        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            return {
                "answer": "observable",
                "sources": [{"url": "https://source.example"}],
            }

    backend = MutableTools()
    client = _BudgetedToolClient(backend, max_calls=3, audit=lambda *_: None)
    client.start_scope("act:lookup:1")
    arguments = {"query": "public query"}

    first = await client.call("web_search", arguments)
    first["sources"][0]["url"] = "caller mutation"
    second = await client.call("web_search", arguments)

    assert second == {
        "answer": "observable",
        "sources": [{"url": "https://source.example"}],
    }
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_budgeted_tool_client_checks_global_budget_before_cached_reuse() -> None:
    tools = FakeTools([])
    client = _BudgetedToolClient(tools, max_calls=1, audit=lambda *_: None)
    client.start_scope("verify:1")
    arguments = {"query": "same query"}
    await client.call("web_search", arguments)

    with pytest.raises(_ToolBudgetExhausted):
        await client.call("web_search", arguments)

    assert client.call_count == 1
    assert tools.call_arguments == [("web_search", arguments)]


@pytest.mark.asyncio
async def test_budgeted_tool_client_does_not_cache_failed_result() -> None:
    class RecoveringTools:
        tool_specs: tuple[ToolSpec, ...] = ()

        def __init__(self) -> None:
            self.calls = 0

        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise ToolExecutionError("temporary failure")
            return {"text": "retry succeeded"}

    backend = RecoveringTools()
    client = _BudgetedToolClient(backend, max_calls=3, audit=lambda *_: None)
    client.start_scope("act:file:1")
    arguments = {"path": "sample.mp3"}

    with pytest.raises(ToolExecutionError):
        await client.call("transcribe_audio", arguments)
    result = await client.call("transcribe_audio", arguments)

    assert result == {"text": "retry succeeded"}
    assert backend.calls == 2
    assert client.call_count == 2


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
    *,
    usage: Usage | None = None,
) -> AgentResult:
    return AgentResult(
        specialist=specialist,
        evidence=evidence,
        candidate_answer=candidate,
        usage=usage or Usage(),
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
    critic = FakeCritic(calls, [_decision("approve", ("evidence-0001",))])
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
        ("submit_answer", {"answer": "Marie Curie", "evidence_ids": ["evidence-0001"]})
    ]


@pytest.mark.asyncio
async def test_orchestrator_aggregates_planner_specialist_and_every_critic_usage() -> None:
    calls: list[str] = []
    source = _evidence("source-1", "source")
    planner = FakePlanner(
        calls,
        _plan(SpecialistName.LOOKUP),
        usage=Usage(input_tokens=3, total_tokens=3),
    )
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [
            _result(
                SpecialistName.LOOKUP,
                (source,),
                "Polonium",
                usage=Usage(output_tokens=5, total_tokens=5),
            )
        ],
    )
    critic = FakeCritic(
        calls,
        [
            _decision("revise", ("evidence-0001",), corrected_answer="Radium"),
            _decision("approve", ("evidence-0001",)),
        ],
        usages=[
            Usage(reasoning_tokens=7, total_tokens=7),
            Usage(input_tokens=11, total_tokens=11),
        ],
    )

    result = await _orchestrator(
        planner=planner,
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=FakeTools(calls),
    ).run(_task())

    assert result.usage == Usage(
        input_tokens=14,
        output_tokens=5,
        reasoning_tokens=7,
        total_tokens=26,
    )


@pytest.mark.asyncio
async def test_orchestrator_adds_hosted_tool_usage_once_on_success() -> None:
    calls: list[str] = []
    source = _evidence("source-1", "source")
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [
            _result(
                SpecialistName.LOOKUP,
                (source,),
                "Marie Curie",
                usage=Usage(input_tokens=2, total_tokens=2),
            )
        ],
        tool_calls=[("web_search", {"query": "synthetic element"})],
    )
    tools = FakeTools(
        calls,
        usage_by_tool={"web_search": Usage(input_tokens=13, output_tokens=5, total_tokens=18)},
    )

    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=FakeCritic(calls, [_decision("approve", ("evidence-0001",))]),
        tools=tools,
    ).run(_task())

    assert result.usage == Usage(input_tokens=15, output_tokens=5, total_tokens=20)


@pytest.mark.asyncio
async def test_orchestrator_keeps_hosted_and_malformed_specialist_usage_on_failure() -> None:
    calls: list[str] = []
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [],
        tool_calls=[("web_search", {"query": "synthetic element"})],
        error=ValueError("malformed specialist output"),
        usage_on_error=Usage(output_tokens=7, total_tokens=7),
    )
    tools = FakeTools(
        calls,
        usage_by_tool={"web_search": Usage(input_tokens=11, output_tokens=3, total_tokens=14)},
    )

    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=FakeCritic(calls, []),
        tools=tools,
    ).run(_task())

    assert result.error_category == TerminalErrorCategory.MALFORMED_RESPONSE
    assert result.usage == Usage(input_tokens=11, output_tokens=10, total_tokens=21)


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
    critic = FakeCritic(
        calls,
        [_decision("approve", ("evidence-0001", "evidence-0004"))],
    )
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
        "evidence-0001",
        "evidence-0002",
        "evidence-0003",
        "evidence-0004",
    )


@pytest.mark.asyncio
async def test_budget_wrapper_delegates_visual_attachment_policy_read(tmp_path: Path) -> None:
    calls: list[str] = []
    attachment = tmp_path / "registered.png"
    attachment.write_bytes(b"image")

    class AttachmentReadingVisual:
        last_usage = Usage()

        async def run(
            self,
            task: GaiaTask,
            step: PlanStep,
            evidence: tuple[Evidence, ...],
            tools: Any,
        ) -> AgentResult:
            assert await tools.read_attachment_bytes(task.attachment_path) == b"image"
            return _result(
                SpecialistName.VISUAL,
                (_evidence("visual-1", "visual"),),
                "blue",
            )

    tools = FakeTools(calls)
    result = await Orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.VISUAL)),
        specialists={SpecialistName.VISUAL: AttachmentReadingVisual()},
        critic=FakeCritic(calls, [_decision("approve", ("evidence-0001",))]),
        tool_client_factory=FakeToolFactory(tools),
        config=_config(),
    ).run(_task(attachment_path=attachment))

    assert result.answer == "blue"
    assert tools.attachment_reads == [attachment]


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
        critic=FakeCritic(calls, [_decision("approve", ("evidence-0001",))]),
        tools=tools,
        config=_config(max_tool_calls=1),
    ).run(_task())

    assert result.answer is None
    assert result.error_category == TerminalErrorCategory.TOOL_BUDGET_EXHAUSTED
    assert tools.call_arguments == [("web_search", {"query": "synthetic element"})]


@pytest.mark.asyncio
async def test_identical_tool_call_across_specialist_and_critic_scopes_is_allowed() -> None:
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
        [_decision("approve", ("evidence-0001",))],
        tool_calls=repeated,
    )
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=FakeTools(calls),
    ).run(_task())

    assert result.answer == "Marie Curie"
    assert result.error_category is None
    assert calls == [
        "planner",
        "lookup",
        "web_search",
        "critic",
        "web_search",
        "submit_answer",
    ]


@pytest.mark.asyncio
async def test_third_identical_tool_call_within_one_verify_scope_stops_as_a_loop() -> None:
    calls: list[str] = []
    repeated = ("web_search", {"query": "same query"})
    source = _evidence("source-1", "source")
    orchestrator = _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={
            SpecialistName.LOOKUP: FakeSpecialist(
                SpecialistName.LOOKUP,
                calls,
                [_result(SpecialistName.LOOKUP, (source,), "Marie Curie")],
            )
        },
        critic=FakeCritic(
            calls,
            [_decision("approve", ("evidence-0001",))],
            tool_calls=[repeated, repeated, repeated],
        ),
        tools=FakeTools(calls),
    )

    result = await orchestrator.run(_task())

    assert result.error_category == TerminalErrorCategory.LOOP_DETECTED
    assert calls.count("web_search") == 1
    assert orchestrator.tool_call_count == 2
    assert "submit_answer" not in calls


@pytest.mark.asyncio
async def test_identical_tool_call_in_second_verify_scope_is_allowed() -> None:
    calls: list[str] = []
    source = _evidence("source-1", "source")
    tools = FakeTools(calls)
    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.LOOKUP)),
        specialists={
            SpecialistName.LOOKUP: FakeSpecialist(
                SpecialistName.LOOKUP,
                calls,
                [_result(SpecialistName.LOOKUP, (source,), "Polonium")],
            )
        },
        critic=FakeCritic(
            calls,
            [
                _decision(
                    "revise",
                    ("evidence-0001",),
                    corrected_answer="Radium",
                ),
                _decision("approve", ("evidence-0001",)),
            ],
            tool_calls=[("web_search", {"query": "same query"})],
        ),
        tools=tools,
    ).run(_task())

    assert result.answer == "Radium"
    assert result.error_category is None
    assert tools.call_arguments == [
        ("web_search", {"query": "same query"}),
        ("web_search", {"query": "same query"}),
        (
            "submit_answer",
            {"answer": "Radium", "evidence_ids": ["evidence-0001"]},
        ),
    ]


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
            _decision("approve", ("evidence-0001",)),
            _decision("approve", ("evidence-0001", "evidence-0002")),
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
    assert computation.prior_evidence[1][0].evidence_id == "evidence-0001"


@pytest.mark.asyncio
async def test_contradictory_computation_candidates_cannot_satisfy_numerical_check() -> None:
    calls: list[str] = []
    computation = FakeSpecialist(
        SpecialistName.COMPUTATION,
        calls,
        [
            _result(
                SpecialistName.COMPUTATION,
                (_evidence("computation-1", "computation"),),
                "42",
            ),
            _result(
                SpecialistName.COMPUTATION,
                (_evidence("computation-2", "computation"),),
                "41",
            ),
        ],
    )
    result = await _orchestrator(
        planner=FakePlanner(
            calls,
            _plan(SpecialistName.COMPUTATION, SpecialistName.COMPUTATION),
        ),
        specialists={SpecialistName.COMPUTATION: computation},
        critic=FakeCritic(
            calls,
            [_decision("approve", ("evidence-0001", "evidence-0002"))],
        ),
        tools=FakeTools(calls),
        config=_config(max_steps=2),
    ).run(_task(question="What is six times seven?"))

    assert result.answer is None
    assert result.error_category == TerminalErrorCategory.CRITIC_REJECTION
    assert "submit_answer" not in calls


@pytest.mark.asyncio
async def test_agreeing_computation_candidates_satisfy_numerical_check() -> None:
    calls: list[str] = []
    computation = FakeSpecialist(
        SpecialistName.COMPUTATION,
        calls,
        [
            _result(
                SpecialistName.COMPUTATION,
                (_evidence("computation-1", "computation"),),
                "42",
            ),
            _result(
                SpecialistName.COMPUTATION,
                (_evidence("computation-2", "computation"),),
                "42.0",
            ),
        ],
    )
    result = await _orchestrator(
        planner=FakePlanner(
            calls,
            _plan(SpecialistName.COMPUTATION, SpecialistName.COMPUTATION),
        ),
        specialists={SpecialistName.COMPUTATION: computation},
        critic=FakeCritic(
            calls,
            [_decision("approve", ("evidence-0001", "evidence-0002"))],
        ),
        tools=FakeTools(calls),
    ).run(_task(question="What is six times seven?"))

    assert result.answer == "42.0"
    assert result.error_category is None
    assert calls[-1] == "submit_answer"


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
        [_decision("approve", ()), _decision("approve", ("evidence-0001",))],
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
            _decision("revise", ("evidence-0001",), corrected_answer="Radium"),
            _decision("approve", ("evidence-0001",)),
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
async def test_model_evidence_ids_are_replaced_before_trusted_state_and_artifacts() -> None:
    calls: list[str] = []
    malicious_id = (
        "Who discovered the synthetic element?|/private/attachment.png|model-secret-token"
    )
    lookup = FakeSpecialist(
        SpecialistName.LOOKUP,
        calls,
        [
            _result(
                SpecialistName.LOOKUP,
                (_evidence(malicious_id, "source"),),
                None,
            ),
            _result(
                SpecialistName.LOOKUP,
                (_evidence("second-raw-id", "source"),),
                "Marie Curie",
            ),
        ],
    )
    critic = FakeCritic(
        calls,
        [_decision("approve", ("evidence-0001", "evidence-0002"))],
    )
    tools = FakeTools(calls)
    orchestrator = _orchestrator(
        planner=FakePlanner(
            calls,
            _plan(SpecialistName.LOOKUP, SpecialistName.LOOKUP),
        ),
        specialists={SpecialistName.LOOKUP: lookup},
        critic=critic,
        tools=tools,
    )

    result = await orchestrator.run(_task())

    assert result.evidence_ids == ("evidence-0001", "evidence-0002")
    assert [item.evidence_id for item in lookup.prior_evidence[1]] == ["evidence-0001"]
    assert [item.evidence_id for item in critic.evidence_seen[0]] == [
        "evidence-0001",
        "evidence-0002",
    ]
    assert tools.call_arguments[-1][1]["evidence_ids"] == [
        "evidence-0001",
        "evidence-0002",
    ]
    assert malicious_id not in repr(orchestrator.audit_events)
    assert malicious_id not in repr(result)
    assert "/private/attachment.png" not in repr(orchestrator.audit_events)
    assert "model-secret-token" not in repr(orchestrator.audit_events)


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
async def test_visual_attachment_policy_failure_is_an_attachment_error(tmp_path: Path) -> None:
    calls: list[str] = []
    attachment = tmp_path / "registered.png"
    attachment.write_bytes(b"image")
    visual = FakeSpecialist(
        SpecialistName.VISUAL,
        calls,
        [],
        error=AttachmentPolicyError("attachment changed during read"),
    )

    result = await _orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.VISUAL)),
        specialists={SpecialistName.VISUAL: visual},
        critic=FakeCritic(calls, []),
        tools=FakeTools(calls),
    ).run(_task(attachment_path=attachment))

    assert result.error_category == TerminalErrorCategory.ATTACHMENT_ERROR


@pytest.mark.asyncio
async def test_visual_low_level_read_oserror_is_an_attachment_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    attachment = tmp_path / "registered.png"
    attachment.write_bytes(b"image")
    policy_client = MCPToolClient(
        task_root=tmp_path,
        config=_config(),
        attachments=(attachment,),
    )

    class PolicyTools(FakeTools):
        async def read_attachment_bytes(self, path: Path) -> bytes:
            return await policy_client.read_attachment_bytes(path)

    class AttachmentReadingVisual:
        last_usage = Usage()

        async def run(
            self,
            task: GaiaTask,
            step: PlanStep,
            evidence: tuple[Evidence, ...],
            tools: Any,
        ) -> AgentResult:
            await tools.read_attachment_bytes(task.attachment_path)
            raise AssertionError("failed attachment read must stop visual execution")

    def fail_read(fd: int, size: int) -> bytes:
        raise OSError(5, "raw device failure /private/sensitive-image.png")

    monkeypatch.setattr(os, "read", fail_read)
    result = await Orchestrator(
        planner=FakePlanner(calls, _plan(SpecialistName.VISUAL)),
        specialists={SpecialistName.VISUAL: AttachmentReadingVisual()},
        critic=FakeCritic(calls, []),
        tool_client_factory=FakeToolFactory(PolicyTools(calls)),
        config=_config(),
    ).run(_task(attachment_path=attachment))

    assert result.error_category == TerminalErrorCategory.ATTACHMENT_ERROR


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
    critic = FakeCritic(calls, [_decision("approve", ("evidence-0001",))])
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


@pytest.mark.parametrize(
    ("first_answer", "second_answer"),
    [("1e3", "1000"), ("-0", "0")],
)
def test_choose_majority_uses_scorer_numeric_equivalence(
    first_answer: str,
    second_answer: str,
) -> None:
    winner = choose_majority(
        [
            _prediction(first_answer, 0),
            _prediction(second_answer, 1),
            _prediction("999", 2),
        ],
        votes=3,
    )

    assert winner.answer == first_answer
    assert winner.vote_count == 2
    assert winner.error_category is None


def test_choose_majority_does_not_treat_parse_failure_as_valid_infinity() -> None:
    result = choose_majority(
        [
            _prediction("not a number", 0),
            _prediction("inf", 1),
            _prediction("different", 2),
        ],
        votes=3,
    )

    assert result.error_category is TerminalErrorCategory.NO_CONSENSUS


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


@pytest.mark.parametrize(
    "predictions",
    [
        [
            _prediction("A", 0),
            _prediction("A", 1).model_copy(update={"task_id": "other-task"}),
        ],
        [
            _prediction("A", 0),
            _prediction("A", 1).model_copy(update={"level": 3}),
        ],
        [_prediction("A", -1)],
        [_prediction("A", 3)],
    ],
)
def test_choose_majority_rejects_mixed_identity_or_out_of_range_run_index(
    predictions: list[TaskPrediction],
) -> None:
    with pytest.raises(ValueError, match="task|level|run index"):
        choose_majority(predictions, votes=3)
