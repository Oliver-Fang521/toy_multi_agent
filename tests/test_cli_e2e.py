from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gaia_agent import cli as cli_module
from gaia_agent import runner as runner_module
from gaia_agent.agents import (
    ComputationAgent,
    CriticAgent,
    EvidenceLinkerAgent,
    LookupAgent,
    PlannerAgent,
)
from gaia_agent.cli import app
from gaia_agent.config import RunConfig
from gaia_agent.models import AuditEvent, GaiaTask, TaskPrediction, TerminalErrorCategory, Usage
from gaia_agent.providers import ModelRequest, ModelResponse
from gaia_agent.reporting import summarize
from gaia_agent.runner import (
    EvaluationRunner,
    ProviderRuntime,
    _private_run_root,
    build_runner,
    verify_run_directory,
)

cli_runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "gaia_records.json"


def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-credential")
    monkeypatch.setenv("HF_TOKEN", "hf_fake_credential")


def _evaluate_arguments(output_dir: Path, *extra: str) -> list[str]:
    return [
        "evaluate",
        "--sample-size",
        "3",
        "--seed",
        "42",
        "--output-dir",
        str(output_dir),
        *extra,
    ]


def _fixture_tasks() -> tuple[list[GaiaTask], str]:
    records = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return (
        [
            GaiaTask(
                task_id=record["task_id"],
                question=record["Question"],
                level=int(record["Level"]),
                final_answer=record["Final answer"],
            )
            for record in records
        ],
        "synthetic-revision",
    )


def _config(output_dir: Path, **updates: Any) -> RunConfig:
    values: dict[str, Any] = {
        "openai_api_key": "sk-fake-credential",
        "hf_token": "hf_fake_credential",
        "sample_size": 3,
        "seed": 42,
        "output_dir": output_dir,
    }
    values.update(updates)
    return RunConfig.model_validate(values)


class _ScriptedOrchestrator:
    def __init__(self, task: GaiaTask, outcomes: deque[object], calls: list[str]) -> None:
        self._task = task
        self._outcomes = outcomes
        self._calls = calls
        self.audit_events: tuple[()] = ()

    async def run(self, task: GaiaTask) -> TaskPrediction:
        self._calls.append(task.task_id)
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, TaskPrediction)
        return outcome


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _TimedOrchestratorFactory:
    def __init__(
        self,
        clock: _FakeClock,
        outcomes: deque[tuple[float, TaskPrediction, tuple[AuditEvent, ...]]],
    ) -> None:
        self.clock = clock
        self.outcomes = outcomes
        self.calls = 0

    def __call__(self, task: GaiaTask) -> Any:
        self.calls += 1
        duration, prediction, events = self.outcomes.popleft()
        clock = self.clock

        class TimedOrchestrator:
            audit_events = events

            async def run(self, active_task: GaiaTask) -> TaskPrediction:
                assert active_task.task_id == task.task_id
                clock.advance(duration)
                return prediction

        return TimedOrchestrator()


def _checkpoint_audit_event(
    cursor: str,
    *,
    candidate_present: bool = False,
) -> AuditEvent:
    return AuditEvent(
        sequence=1,
        event_type="checkpoint_saved",
        timestamp=datetime.now(UTC),
        payload={
            "cursor": cursor,
            "logical_tool_calls": 0,
            "step_count": 1,
            "candidate_present": candidate_present,
        },
    )


def _stage_budget_audit_event(stage: str, reason: str) -> AuditEvent:
    return AuditEvent(
        sequence=2,
        event_type="stage_budget_exhausted",
        timestamp=datetime.now(UTC),
        payload={
            "stage": stage,
            "reason": reason,
            "timeout_seconds": 1.0,
            "max_calls": 1,
            "logical_calls": 0,
        },
    )


class _OfflineProvider:
    async def complete(
        self,
        request: ModelRequest,
        executor: Any | None = None,
    ) -> ModelResponse:
        context = _request_context(request)
        if request.instructions == PlannerAgent.instructions:
            return ModelResponse(
                parsed={
                    "restatement": "Gather one source and verify the concise answer",
                    "steps": [{"specialist": "lookup", "instruction": "Use the tool plane"}],
                }
            )
        if request.instructions == LookupAgent.instructions:
            assert executor is not None
            task_id = str(context["task_id"])
            result = await executor.call("web_search", {"query": task_id})
            sources = result["sources"]
            return ModelResponse(
                parsed={
                    "specialist": "lookup",
                    "evidence": [
                        {
                            "evidence_id": f"offline-{task_id}",
                            "kind": "source",
                            "summary": "Deterministic test-process backend result",
                            "source_url": sources[0]["url"],
                        }
                    ],
                    "candidate_answer": "synthetic gold "
                    + str(context["question"]).removeprefix("Synthetic question "),
                }
            )
        if request.instructions == CriticAgent.instructions:
            evidence = context["evidence"]
            return ModelResponse(
                parsed={
                    "verdict": "approve",
                    "feedback": "Observable source evidence is present",
                    "checked_evidence_ids": [item["evidence_id"] for item in evidence],
                    "corrected_answer": None,
                }
            )
        raise AssertionError("unexpected agent request")


class _ReliabilityAuditProvider:
    supports_vision = True

    def __init__(self, mode: str) -> None:
        self._mode = mode

    async def complete(
        self,
        request: ModelRequest,
        executor: Any | None = None,
    ) -> ModelResponse:
        context = _request_context(request)
        if request.instructions == PlannerAgent.instructions:
            steps = {
                "context": [{"specialist": "lookup", "instruction": "Gather a source"}],
                "early": [
                    {"specialist": "lookup", "instruction": "Gather a source"},
                    {"specialist": "computation", "instruction": "Calculate"},
                ],
                "soft": [
                    {"specialist": "computation", "instruction": "Calculate"},
                    {"specialist": "lookup", "instruction": "Gather corroboration"},
                ],
            }[self._mode]
            return ModelResponse(parsed={"restatement": "Offline reliability route", "steps": steps})
        if request.instructions == ComputationAgent.instructions:
            return ModelResponse(
                parsed={
                    "specialist": "computation",
                    "evidence": [
                        {
                            "evidence_id": f"computation-{context['task_id']}",
                            "kind": "computation",
                            "summary": "Deterministic computation",
                        }
                    ],
                    "candidate_answer": "42",
                }
            )
        if request.instructions.startswith(LookupAgent.instructions):
            if self._mode == "soft":
                return ModelResponse(
                    parsed={"specialist": "computation", "evidence": [], "candidate_answer": "42"}
                )
            return ModelResponse(
                parsed={
                    "specialist": "lookup",
                    "evidence": [
                        {
                            "evidence_id": f"source-{context['task_id']}",
                            "kind": "source",
                            "summary": "Deterministic source",
                            "source_url": "https://offline.example/source",
                        }
                    ],
                    "candidate_answer": "offline answer",
                }
            )
        if request.instructions == CriticAgent.instructions:
            evidence = context["evidence"]
            if self._mode == "context":
                return ModelResponse(
                    parsed={
                        "verdict": "insufficient_evidence",
                        "feedback": "No bounded context remains",
                        "checked_evidence_ids": [],
                        "corrected_answer": None,
                    }
                )
            return ModelResponse(
                parsed={
                    "verdict": "approve",
                    "feedback": "Offline evidence is sufficient",
                    "checked_evidence_ids": [item["evidence_id"] for item in evidence],
                    "corrected_answer": None,
                }
            )
        raise AssertionError("unexpected reliability audit request")


def _request_context(request: ModelRequest) -> dict[str, Any]:
    content = request.input_items[0]["content"]
    text = next(item["text"] for item in content if item["type"] == "input_text")
    value = json.loads(text)
    assert isinstance(value, dict)
    return value


def _inject_offline_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def injected_build_runner(config: RunConfig) -> EvaluationRunner:
        return build_runner(
            config,
            dataset_loader=lambda _config: _fixture_tasks(),
            provider_factory=lambda _config: ProviderRuntime(
                provider=_OfflineProvider(),
                provider_name="offline-fixture",
                evaluation_mode="test",
                fake_mcp_backend=True,
            ),
        )

    monkeypatch.setattr(cli_module, "build_runner", injected_build_runner)


def _inject_capturing_runner(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[RunConfig],
) -> None:
    class _CapturingRunner:
        async def run(self):
            return summarize([])

    def capture(config: RunConfig):
        captured.append(config)
        return _CapturingRunner()

    monkeypatch.setattr(cli_module, "build_runner", capture)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_event"),
    [
        ("context", "context_budget_pressure"),
        ("early", "plan_early_stop"),
        ("soft", "soft_step_failure"),
    ],
)
async def test_offline_evaluation_runner_persists_reliability_audit_events(
    tmp_path: Path,
    mode: str,
    expected_event: str,
) -> None:
    tasks, revision = _fixture_tasks()
    if mode == "context":
        tasks = [task.model_copy(update={"question": "x" * 4_096}) for task in tasks]
    runner = build_runner(
        _config(
            tmp_path / mode,
            max_steps=2,
            stage_context_max_bytes=5_000 if mode == "context" else 24_576,
        ),
        dataset_loader=lambda _config: (tasks, revision),
        provider_factory=lambda _config: ProviderRuntime(
            provider=_ReliabilityAuditProvider(mode),
            provider_name="offline-reliability",
            evaluation_mode="test",
            fake_mcp_backend=True,
        ),
    )

    await runner.run()

    audit_events = [
        json.loads(line)["event_type"]
        for audit_path in (tmp_path / mode / "tasks").rglob("audit.jsonl")
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert expected_event in audit_events
    assert verify_run_directory(tmp_path / mode).total == 3


@pytest.mark.asyncio
async def test_evaluation_runner_persists_context_pressure_at_configured_one_mebibyte_boundary(
    tmp_path: Path,
) -> None:
    """A valid maximum context budget must not abort public artifact persistence."""

    class _BoundaryOrchestrator:
        def __init__(self, task: GaiaTask) -> None:
            self._task = task
            self.audit_events = (
                AuditEvent(
                    sequence=1,
                    event_type="context_budget_pressure",
                    timestamp=datetime(2026, 8, 17, tzinfo=UTC),
                    payload={
                        "bytes": 1_048_576,
                        "item_count": 1,
                        "max_bytes": 1_048_576,
                    },
                ),
            )

        async def run(self, task: GaiaTask) -> TaskPrediction:
            assert task.task_id == self._task.task_id
            return TaskPrediction(
                task_id=task.task_id,
                level=task.level,
                answer=task.final_answer,
                critic_verdict="approve",
            )

    tasks, revision = _fixture_tasks()
    output_dir = tmp_path / "one-mebibyte-context"
    runner = EvaluationRunner(
        config=_config(output_dir, stage_context_max_bytes=1_048_576),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=_BoundaryOrchestrator,
    )

    await runner.run()

    assert verify_run_directory(output_dir).total == 3
    persisted_payloads = [
        json.loads(line)["payload"]
        for audit_path in (output_dir / "tasks").rglob("audit.jsonl")
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted_payloads == [
        {"bytes": 1_048_576, "item_count": 1, "max_bytes": 1_048_576}
    ] * 3


@pytest.mark.asyncio
async def test_evaluation_runner_persists_safe_stage_budget_audit(
    tmp_path: Path,
) -> None:
    class _StageBudgetAuditOrchestrator:
        def __init__(self, task: GaiaTask) -> None:
            self.audit_events = (
                AuditEvent(
                    sequence=1,
                    event_type="stage_budget_exhausted",
                    timestamp=datetime(2026, 8, 17, tzinfo=UTC),
                    payload={
                        "stage": "specialist",
                        "reason": "time",
                        "timeout_seconds": 70.0,
                        "max_calls": 18,
                        "logical_calls": 7,
                    },
                ),
            )

        async def run(self, task: GaiaTask) -> TaskPrediction:
            return TaskPrediction(
                task_id=task.task_id,
                level=task.level,
                error_category=TerminalErrorCategory.TIMEOUT,
            )

    tasks, revision = _fixture_tasks()
    output_dir = tmp_path / "stage-budget-audit"
    runner = EvaluationRunner(
        config=_config(output_dir),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=_StageBudgetAuditOrchestrator,
    )

    await runner.run()

    assert verify_run_directory(output_dir).total == 3
    persisted_payloads = [
        json.loads(line)["payload"]
        for audit_path in (output_dir / "tasks").rglob("audit.jsonl")
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted_payloads == [
        {
            "stage": "specialist",
            "reason": "time",
            "timeout_seconds": 70.0,
            "max_calls": 18,
            "logical_calls": 7,
        }
    ] * 3


@pytest.mark.asyncio
async def test_evaluation_runner_persists_read_webpage_search_fallback_audit(
    tmp_path: Path,
) -> None:
    class _FallbackAuditOrchestrator:
        def __init__(self, task: GaiaTask) -> None:
            self._task = task
            self.audit_events = (
                AuditEvent(
                    sequence=1,
                    event_type="tool_call_completed",
                    timestamp=datetime(2026, 8, 17, tzinfo=UTC),
                    payload={
                        "tool": "read_webpage",
                        "call_number": 1,
                        "argument_keys": ["url"],
                        "result_keys": ["answer", "fallback", "sources", "url"],
                    },
                ),
            )

        async def run(self, task: GaiaTask) -> TaskPrediction:
            return TaskPrediction(
                task_id=task.task_id,
                level=task.level,
                answer=task.final_answer,
                critic_verdict="approve",
            )

    tasks, revision = _fixture_tasks()
    output_dir = tmp_path / "webpage-fallback-audit"
    runner = EvaluationRunner(
        config=_config(output_dir),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=_FallbackAuditOrchestrator,
    )

    await runner.run()

    assert verify_run_directory(output_dir).total == 3
    persisted_result_keys = [
        json.loads(line)["payload"]["result_keys"]
        for audit_path in (output_dir / "tasks").rglob("audit.jsonl")
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted_result_keys == [
        ["answer", "fallback", "sources", "url"],
    ] * 3


def test_build_runner_rejects_dataset_only_injection_before_default_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_client_calls: list[dict[str, Any]] = []

    def unexpected_async_openai(**kwargs: Any) -> None:
        provider_client_calls.append(kwargs)
        raise AssertionError("default provider client must not be constructed")

    monkeypatch.setattr(runner_module, "AsyncOpenAI", unexpected_async_openai)

    with pytest.raises(ValueError, match="dataset_loader and provider_factory must be injected together"):
        build_runner(
            _config(tmp_path),
            dataset_loader=lambda _config: _fixture_tasks(),
        )

    assert provider_client_calls == []
    assert not (tmp_path / "manifest.json").exists()


def test_build_runner_rejects_provider_only_injection_before_factory_or_store(
    tmp_path: Path,
) -> None:
    provider_factory_calls: list[RunConfig] = []

    def injected_provider_factory(config: RunConfig) -> ProviderRuntime:
        provider_factory_calls.append(config)
        return ProviderRuntime(
            provider=_OfflineProvider(),
            provider_name="offline-fixture",
            evaluation_mode="test",
            fake_mcp_backend=True,
        )

    with pytest.raises(ValueError, match="dataset_loader and provider_factory must be injected together"):
        build_runner(
            _config(tmp_path),
            provider_factory=injected_provider_factory,
        )

    assert provider_factory_calls == []
    assert not (tmp_path / "manifest.json").exists()


def test_build_runner_applies_configured_retry_policy_to_deepseek(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    class _Client:
        def with_options(self, *, max_retries: int) -> _Client:
            assert max_retries == 0
            return self

    class _DeepSeekProvider:
        supports_vision = False

        def __init__(self, *_: object, retry_policy: object, **__: object) -> None:
            captured.append(retry_policy)

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.example/v1")
    monkeypatch.setattr(runner_module, "AsyncOpenAI", lambda **_: _Client())
    monkeypatch.setattr(runner_module, "DeepSeekProvider", _DeepSeekProvider)

    build_runner(_config(tmp_path, provider_max_attempts=6, provider_backoff_base=1.25))

    assert len(captured) == 1
    policy = captured[0]
    assert policy.max_attempts == 6
    assert policy.base_delay_seconds == 1.25


def test_build_runner_applies_configured_retry_policy_to_openai(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    class _Client:
        def with_options(self, *, max_retries: int) -> _Client:
            assert max_retries == 0
            return self

    class _OpenAIProvider:
        supports_vision = True

        def __init__(self, *_: object, retry_policy: object, **__: object) -> None:
            captured.append(retry_policy)

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(runner_module, "AsyncOpenAI", lambda **_: _Client())
    monkeypatch.setattr(runner_module, "OpenAIProvider", _OpenAIProvider)

    build_runner(_config(tmp_path, provider_max_attempts=6, provider_backoff_base=1.25))

    assert len(captured) == 1
    policy = captured[0]
    assert policy.max_attempts == 6
    assert policy.base_delay_seconds == 1.25


def test_build_runner_rejects_non_luna_max_production_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "AsyncOpenAI",
        lambda **_: pytest.fail("production client must not be constructed for rejected config"),
    )

    with pytest.raises(ValueError, match="requires model=gpt-5.6-luna"):
        build_runner(_config(tmp_path, model="other-model", reasoning_effort="high"))


def test_build_runner_composes_luna_max_for_every_production_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_models: list[str] = []

    class _Client:
        def with_options(self, *, max_retries: int) -> _Client:
            assert max_retries == 0
            return self

    class _OpenAIProvider:
        supports_vision = True

        def __init__(self, _: object, model: str, **__: object) -> None:
            captured_models.append(model)

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setattr(runner_module, "AsyncOpenAI", lambda **_: _Client())
    monkeypatch.setattr(runner_module, "OpenAIProvider", _OpenAIProvider)
    config = RunConfig(
        openai_api_key="sk-fake-credential",
        hf_token="hf-fake-credential",
        sample_size=3,
        output_dir=tmp_path,
    )

    runner = build_runner(config)
    orchestrator = runner._orchestrator_factory(
        GaiaTask(task_id="offline-l1", question="Synthetic", level=1, final_answer="unused")
    )

    assert captured_models == ["gpt-5.6-luna"]
    assert orchestrator._planner.reasoning_effort == "max"
    assert {agent.reasoning_effort for agent in orchestrator._specialists.values()} == {"max"}
    assert orchestrator._critic.reasoning_effort == "max"
    assert isinstance(orchestrator._evidence_linker, EvidenceLinkerAgent)
    assert orchestrator._evidence_linker.reasoning_effort == "max"
    assert orchestrator._evidence_linker.max_context_bytes == config.stage_context_max_bytes


@pytest.mark.parametrize(
    ("provider_name", "evaluation_mode"),
    [
        pytest.param("offline-fixture", "production", id="production-mode"),
        pytest.param("openai", "test", id="production-provider-name"),
    ],
)
def test_complete_injected_composition_requires_non_production_identity(
    tmp_path: Path,
    provider_name: str,
    evaluation_mode: Any,
) -> None:
    with pytest.raises(ValueError, match="non-production test identity"):
        build_runner(
            _config(tmp_path),
            dataset_loader=lambda _config: _fixture_tasks(),
            provider_factory=lambda _config: ProviderRuntime(
                provider=_OfflineProvider(),
                provider_name=provider_name,
                evaluation_mode=evaluation_mode,
            ),
        )

    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize(
    ("global_limit", "agent_limit"),
    [(24, 6), (5, 5)],
)
def test_build_runner_caps_each_specialist_and_critic_request(
    tmp_path: Path,
    global_limit: int,
    agent_limit: int,
) -> None:
    runner = build_runner(
        _config(tmp_path, max_tool_calls=global_limit),
        dataset_loader=lambda _config: _fixture_tasks(),
        provider_factory=lambda _config: ProviderRuntime(
            provider=_OfflineProvider(),
            provider_name="offline-fixture",
            evaluation_mode="test",
            fake_mcp_backend=True,
        ),
    )
    orchestrator = runner._orchestrator_factory(_fixture_tasks()[0][0])

    assert {agent.max_tool_calls for agent in orchestrator._specialists.values()} == {
        agent_limit
    }
    assert orchestrator._critic.max_tool_calls == agent_limit
    assert orchestrator._config.max_tool_calls == global_limit


def test_private_notebook_is_a_sibling_and_never_enters_public_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    tasks, revision = _fixture_tasks()
    runner = build_runner(
        _config(output_dir),
        dataset_loader=lambda _config: (tasks, revision),
        provider_factory=lambda _config: ProviderRuntime(
            provider=_OfflineProvider(),
            provider_name="offline-fixture",
            evaluation_mode="test",
            fake_mcp_backend=True,
        ),
    )
    sentinel = "https://private.example/sentinel-research"

    asyncio.run(runner.run())

    first_vote_notebook = runner._orchestrator_factory(
        tasks[0], "vote-0001"
    )._notebook
    second_vote_notebook = runner._orchestrator_factory(
        tasks[0], "vote-0002"
    )._notebook
    assert first_vote_notebook is not None
    assert second_vote_notebook is not None
    assert first_vote_notebook.task_id == second_vote_notebook.task_id == tasks[0].task_id
    assert first_vote_notebook.root != second_vote_notebook.root
    first_vote_notebook.record(
        "unused_private_tool",
        {"query": sentinel},
        {"private_text": sentinel},
    )

    private_root = _private_run_root(output_dir)
    assert private_root == tmp_path / ".run.private"
    assert private_root.is_dir()
    public_artifacts = "".join(
        path.read_text(encoding="utf-8") for path in output_dir.rglob("*") if path.is_file()
    )
    assert sentinel not in public_artifacts
    assert sentinel in "".join(
        path.read_text(encoding="utf-8") for path in private_root.rglob("*.json")
    )
    assert verify_run_directory(output_dir).total == 3


def test_new_run_at_same_path_rotates_private_notebook_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    tasks, revision = _fixture_tasks()

    def make_runner() -> EvaluationRunner:
        return build_runner(
            _config(output_dir),
            dataset_loader=lambda _config: (tasks, revision),
            provider_factory=lambda _config: ProviderRuntime(
                provider=_OfflineProvider(),
                provider_name="offline-fixture",
                evaluation_mode="test",
                fake_mcp_backend=True,
            ),
        )

    asyncio.run(make_runner().run())
    first_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    private_root = _private_run_root(output_dir)
    assert list(private_root.rglob("*.json"))
    shutil.rmtree(output_dir)

    asyncio.run(make_runner().run())

    second_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["started_at"] != first_manifest["started_at"]
    second_audit = "".join(
        path.read_text(encoding="utf-8") for path in output_dir.rglob("audit.jsonl")
    )
    assert '"event_type":"tool_call_restored"' not in second_audit


def test_manifest_persists_stage_budgets_for_safe_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    tasks, revision = _fixture_tasks()
    runner = build_runner(
        _config(
            output_dir,
            provider_max_attempts=6,
            provider_backoff_base=1.25,
            private_notebook_max_bytes=80_000,
            stage_context_max_bytes=32_768,
            search_circuit_cooldown=90.0,
            planner_timeout_seconds=12,
            specialist_timeout_seconds=70,
            critic_timeout_seconds=18,
            specialist_max_tool_calls=7,
            critic_max_tool_calls=2,
            langfuse_input_cost_per_million="0.25",
            langfuse_output_cost_per_million="1.50",
        ),
        dataset_loader=lambda _config: (tasks, revision),
        provider_factory=lambda _config: ProviderRuntime(
            provider=_OfflineProvider(),
            provider_name="offline-fixture",
            evaluation_mode="test",
            fake_mcp_backend=True,
        ),
    )

    asyncio.run(runner.run())

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["limits"] == {
        "max_steps": 8,
        "max_tool_calls": 24,
        "timeout_seconds": 300.0,
        "votes": 1,
        "provider_max_attempts": 6,
        "provider_backoff_base": 1.25,
        "private_notebook_max_bytes": 80_000,
        "stage_context_max_bytes": 32_768,
        "search_circuit_cooldown": 90.0,
        "planner_timeout_seconds": 12.0,
        "specialist_timeout_seconds": 70.0,
        "critic_timeout_seconds": 18.0,
        "specialist_max_tool_calls": 7,
        "critic_max_tool_calls": 2,
        "langfuse_input_cost_per_million": "0.25",
        "langfuse_output_cost_per_million": "1.50",
    }


def test_evaluate_cli_runs_four_planes_over_real_stdio_and_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)
    output_dir = tmp_path / "run"

    result = cli_runner.invoke(app, _evaluate_arguments(output_dir))

    assert result.exit_code == 0, result.output
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider_name"] == "offline-fixture"
    assert manifest["evaluation_mode"] == "test"
    audit = "".join(path.read_text(encoding="utf-8") for path in output_dir.rglob("audit.jsonl"))
    assert '"tool":"web_search"' in audit
    assert '"tool":"submit_answer"' in audit


def test_evaluate_cli_reports_only_langfuse_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)

    result = cli_runner.invoke(app, _evaluate_arguments(tmp_path / "run"))

    assert result.exit_code == 0, result.output
    assert "langfuse=disabled" in result.output
    assert "langfuse_flush=ok" in result.output
    assert "LANGFUSE_" not in result.output


def test_public_cli_does_not_expose_or_accept_fixture_backend_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    captured: list[RunConfig] = []
    _inject_capturing_runner(monkeypatch, captured)

    help_result = cli_runner.invoke(app, ["evaluate", "--help"])

    assert help_result.exit_code == 0
    assert "--fake-provider" not in help_result.output
    assert "--dataset-fixture" not in help_result.output
    for arguments in (
        ["evaluate", "--fake-provider"],
        ["evaluate", "--dataset-fixture", str(FIXTURE)],
    ):
        result = cli_runner.invoke(app, arguments)
        assert result.exit_code == 2
        assert "No such option" in result.output
    assert captured == []


def test_cli_uses_run_config_environment_when_options_are_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    monkeypatch.setenv("GAIA_MODEL", "env-model")
    monkeypatch.setenv("GAIA_REASONING_EFFORT", "high")
    monkeypatch.setenv("GAIA_SAMPLE_SIZE", "6")
    monkeypatch.setenv("GAIA_SEED", "9")
    monkeypatch.setenv("GAIA_MAX_STEPS", "3")
    monkeypatch.setenv("GAIA_MAX_TOOL_CALLS", "7")
    monkeypatch.setenv("GAIA_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("GAIA_PLANNER_TIMEOUT_SECONDS", "1.25")
    monkeypatch.setenv("GAIA_SPECIALIST_TIMEOUT_SECONDS", "8.75")
    monkeypatch.setenv("GAIA_CRITIC_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("GAIA_SPECIALIST_MAX_TOOL_CALLS", "5")
    monkeypatch.setenv("GAIA_CRITIC_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("GAIA_VOTES", "3")
    monkeypatch.setenv("GAIA_OUTPUT_DIR", str(tmp_path / "env-run"))
    monkeypatch.setenv("GAIA_RESUME", "true")
    captured: list[RunConfig] = []
    _inject_capturing_runner(monkeypatch, captured)

    result = cli_runner.invoke(app, ["evaluate"])

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    config = captured[0]
    assert config.model == "env-model"
    assert config.reasoning_effort == "high"
    assert config.sample_size == 6
    assert config.seed == 9
    assert config.max_steps == 3
    assert config.max_tool_calls == 7
    assert config.timeout_seconds == 12.5
    assert config.planner_timeout_seconds == 1.25
    assert config.specialist_timeout_seconds == 8.75
    assert config.critic_timeout_seconds == 2.5
    assert config.specialist_max_tool_calls == 5
    assert config.critic_max_tool_calls == 2
    assert config.votes == 3
    assert config.output_dir == tmp_path / "env-run"
    assert config.resume is True


def test_explicit_cli_options_override_run_config_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    monkeypatch.setenv("GAIA_MODEL", "env-model")
    monkeypatch.setenv("GAIA_REASONING_EFFORT", "low")
    monkeypatch.setenv("GAIA_MAX_STEPS", "2")
    monkeypatch.setenv("GAIA_MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("GAIA_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("GAIA_PLANNER_TIMEOUT_SECONDS", "0.4")
    monkeypatch.setenv("GAIA_SPECIALIST_TIMEOUT_SECONDS", "2.8")
    monkeypatch.setenv("GAIA_CRITIC_TIMEOUT_SECONDS", "0.8")
    monkeypatch.setenv("GAIA_SPECIALIST_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("GAIA_CRITIC_MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("GAIA_VOTES", "1")
    captured: list[RunConfig] = []
    _inject_capturing_runner(monkeypatch, captured)

    result = cli_runner.invoke(
        app,
        [
            "evaluate",
            "--model",
            "cli-model",
            "--reasoning-effort",
            "xhigh",
            "--max-steps",
            "5",
            "--max-tool-calls",
            "9",
            "--timeout-seconds",
            "20",
            "--planner-timeout-seconds",
            "2",
            "--specialist-timeout-seconds",
            "14",
            "--critic-timeout-seconds",
            "4",
            "--specialist-max-tool-calls",
            "6",
            "--critic-max-tool-calls",
            "3",
            "--votes",
            "3",
            "--output-dir",
            str(tmp_path / "cli-run"),
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured[0]
    assert config.model == "cli-model"
    assert config.reasoning_effort == "xhigh"
    assert config.max_steps == 5
    assert config.max_tool_calls == 9
    assert config.timeout_seconds == 20
    assert config.planner_timeout_seconds == 2
    assert config.specialist_timeout_seconds == 14
    assert config.critic_timeout_seconds == 4
    assert config.specialist_max_tool_calls == 6
    assert config.critic_max_tool_calls == 3
    assert config.votes == 3
    assert config.output_dir == tmp_path / "cli-run"
    assert config.resume is True


def test_invalid_run_config_environment_fails_without_printing_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    invalid_value = "not-a-private-limit"
    monkeypatch.setenv("GAIA_MAX_TOOL_CALLS", invalid_value)
    captured: list[RunConfig] = []
    _inject_capturing_runner(monkeypatch, captured)

    result = cli_runner.invoke(app, ["evaluate"])

    assert result.exit_code == 2
    assert "configuration error" in result.output.lower()
    assert invalid_value not in result.output
    assert captured == []


@pytest.mark.asyncio
async def test_production_stdio_client_removes_ambient_fake_backend_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str | None] = []

    class _CapturingClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            observed.append(os.environ.get("GAIA_MCP_FAKE_BACKEND"))
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setenv("GAIA_MCP_FAKE_BACKEND", "1")
    monkeypatch.setattr(runner_module, "MCPToolClient", _CapturingClient)
    task = GaiaTask(task_id="ambient-test", question="q", level=1, final_answer="a")

    async with runner_module._stdio_client(task, _config(tmp_path), fake_backend=False):
        pass

    assert observed == [None]
    assert os.environ["GAIA_MCP_FAKE_BACKEND"] == "1"


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "HF_TOKEN"])
def test_evaluate_preflight_names_missing_credentials_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)
    monkeypatch.delenv(missing)

    result = cli_runner.invoke(app, _evaluate_arguments(tmp_path / "run"))

    assert result.exit_code == 2
    assert missing in result.output
    assert "sk-fake-credential" not in result.output
    assert "hf_fake_credential" not in result.output
    assert not (tmp_path / "run" / "manifest.json").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--model", ""),
        ("--max-steps", "0"),
        ("--max-tool-calls", "0"),
        ("--timeout-seconds", "0"),
        ("--votes", "2"),
        ("--sample-size", "4"),
    ],
)
def test_evaluate_preflight_rejects_invalid_model_and_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, str],
) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)

    result = cli_runner.invoke(app, _evaluate_arguments(tmp_path / "run", *arguments))

    assert result.exit_code == 2
    assert "configuration error" in result.output.lower()
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_seed_selection_and_task_level_manifest_are_literal_and_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = cli_runner.invoke(app, _evaluate_arguments(first))
    second_result = cli_runner.invoke(app, _evaluate_arguments(second))

    assert first_result.exit_code == second_result.exit_code == 0
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    expected_ids = ["synthetic-l1-a", "synthetic-l2-b", "synthetic-l3-b"]
    expected_levels = {"synthetic-l1-a": 1, "synthetic-l2-b": 2, "synthetic-l3-b": 3}
    assert first_manifest["selected_task_ids"] == expected_ids
    assert first_manifest["expected_levels"] == expected_levels
    assert second_manifest["selected_task_ids"] == expected_ids
    assert second_manifest["expected_levels"] == expected_levels


@pytest.mark.asyncio
async def test_runner_retries_one_transient_failure_once_only(tmp_path: Path) -> None:
    class PrivateConnectionError(ConnectionError):
        pass

    tasks, revision = _fixture_tasks()
    selected_ids = ("synthetic-l1-a", "synthetic-l2-b", "synthetic-l3-b")
    selected = {task.task_id: task for task in tasks if task.task_id in selected_ids}
    calls: list[str] = []
    first_failures: tuple[object, ...] = (
        TimeoutError("temporary"),
        TaskPrediction(
            task_id="synthetic-l2-b",
            level=2,
            answer=None,
            usage=Usage(total_tokens=1),
            critic_verdict="not_run",
            error_category=TerminalErrorCategory.TIMEOUT,
        ),
        PrivateConnectionError("temporary"),
    )
    outcomes: dict[str, deque[object]] = {}
    for task_id, first_failure in zip(selected, first_failures, strict=True):
        task = selected[task_id]
        outcomes[task_id] = deque(
            [
                first_failure,
                TaskPrediction(
                    task_id=task_id,
                    level=task.level,
                    answer="offline answer",
                    usage=Usage(total_tokens=2),
                    critic_verdict="approve",
                    evidence_ids=("evidence-0001",),
                ),
            ]
        )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    runner = EvaluationRunner(
        config=_config(tmp_path),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=lambda task: _ScriptedOrchestrator(task, outcomes[task.task_id], calls),
        sleep=record_sleep,
    )

    summary = await runner.run()

    assert summary.total == 3
    assert calls == [
        "synthetic-l1-a",
        "synthetic-l1-a",
        "synthetic-l2-b",
        "synthetic-l2-b",
        "synthetic-l3-b",
        "synthetic-l3-b",
    ]
    assert sleeps == [0.05, 0.05, 0.05]
    assert summary.total_tokens == 7
    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert predictions[2]["retry_category"] == "ConnectionError"
    assert "PrivateConnectionError" not in repr(predictions)


@pytest.mark.asyncio
async def test_runner_early_transient_without_checkpoint_retries_inside_vote_deadline(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        usage=Usage(total_tokens=2),
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.TIMEOUT,
                    ),
                    (),
                ),
                (
                    2.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer="recovered answer",
                        usage=Usage(total_tokens=3),
                        critic_verdict="approve",
                        evidence_ids=("evidence-0001",),
                    ),
                    (),
                ),
            ]
        ),
    )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=record_sleep,
        clock=clock,
    )

    result = await runner._run_vote(task)

    assert factory.calls == 2
    assert clock() == 3.0
    assert sleeps == [0.05]
    assert result.prediction.answer == "recovered answer"
    assert result.prediction.usage == Usage(total_tokens=5)
    assert result.prediction.attempt_number == 2
    assert result.prediction.recovery_count == 1
    assert result.prediction.recovery_succeeded is True


@pytest.mark.asyncio
async def test_runner_late_transient_without_checkpoint_is_not_replayed_from_plan(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    6.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.TIMEOUT,
                    ),
                    (),
                )
            ]
        ),
    )
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
        clock=clock,
    )

    result = await runner._run_vote(task)

    assert factory.calls == 1
    assert result.prediction.attempt_number == 1
    assert result.prediction.recovery_count == 0
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT


@pytest.mark.asyncio
async def test_runner_near_deadline_checkpoint_is_not_replayed_from_plan(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    9.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.TIMEOUT,
                    ),
                    (_checkpoint_audit_event("verify"),),
                )
            ]
        ),
    )
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
        clock=clock,
    )

    result = await runner._run_vote(task)

    assert factory.calls == 1
    assert result.prediction.attempt_number == 1
    assert result.prediction.recovery_count == 0
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT


@pytest.mark.asyncio
async def test_runner_specialist_timeout_without_stable_candidate_is_terminal(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.TIMEOUT,
                    ),
                    (
                        _checkpoint_audit_event("act"),
                        _stage_budget_audit_event("specialist", "time"),
                        _checkpoint_audit_event("act").model_copy(
                            update={"sequence": 3}
                        ),
                    ),
                )
            ]
        ),
    )
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
        clock=clock,
    )

    result = await runner._run_vote(task)

    assert factory.calls == 1
    assert result.prediction.attempt_number == 1
    assert result.prediction.recovery_count == 0
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT


@pytest.mark.asyncio
async def test_runner_specialist_timeout_with_checkpoint_candidate_resumes_from_act(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        usage=Usage(total_tokens=3),
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.TIMEOUT,
                    ),
                    (
                        _checkpoint_audit_event("act", candidate_present=True),
                        _stage_budget_audit_event("specialist", "time"),
                        _checkpoint_audit_event(
                            "act", candidate_present=True
                        ).model_copy(update={"sequence": 3}),
                    ),
                ),
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer="42",
                        usage=Usage(total_tokens=5),
                        critic_verdict="approve",
                    ),
                    (
                        AuditEvent(
                            sequence=1,
                            event_type="checkpoint_restored",
                            timestamp=datetime.now(UTC),
                            payload={
                                "cursor": "act",
                                "logical_tool_calls": 0,
                                "step_count": 1,
                                "candidate_present": True,
                            },
                        ),
                    ),
                ),
            ]
        ),
    )
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
        clock=clock,
    )

    result = await runner._run_vote(task, vote_namespace="vote-0001")

    assert factory.calls == 2
    assert result.prediction.answer == "42"
    assert result.prediction.usage == Usage(total_tokens=5)
    assert result.prediction.recovery_succeeded is True


@pytest.mark.asyncio
async def test_runner_checkpoint_transient_resumes_once_with_remaining_vote_budget(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    2.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        usage=Usage(total_tokens=5),
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.RETRY_EXHAUSTED,
                        retry_reason="provider_transient_exhausted",
                        retry_category="APIConnectionError",
                        provider_attempt_count=2,
                    ),
                    (_checkpoint_audit_event("verify"),),
                ),
                (
                    3.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer="resumed answer",
                        usage=Usage(total_tokens=8),
                        critic_verdict="approve",
                        evidence_ids=("evidence-0001",),
                    ),
                    (
                        AuditEvent(
                            sequence=1,
                            event_type="checkpoint_restored",
                            timestamp=datetime.now(UTC),
                            payload={
                                "cursor": "verify",
                                "logical_tool_calls": 0,
                                "step_count": 1,
                            },
                        ),
                    ),
                ),
            ]
        ),
    )
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
        clock=clock,
    )

    result = await runner._run_vote(task)

    assert factory.calls == 2
    assert clock() == 5.0
    assert result.prediction.answer == "resumed answer"
    assert result.prediction.usage == Usage(total_tokens=8)
    assert result.prediction.attempt_number == 2
    assert result.prediction.recovery_count == 1
    assert any(
        event.event_type == "checkpoint_restored" for event in result.audit_events
    )


@pytest.mark.asyncio
async def test_runner_scopes_checkpoints_per_vote_but_reuses_scope_for_recovery(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    calls: list[tuple[str, int]] = []
    attempts: defaultdict[str, int] = defaultdict(int)

    def factory(active_task: GaiaTask, vote_namespace: str) -> Any:
        assert active_task.task_id == task.task_id
        attempts[vote_namespace] += 1
        vote_attempt = attempts[vote_namespace]
        calls.append((vote_namespace, vote_attempt))

        class NamespacedOrchestrator:
            audit_events: tuple[AuditEvent, ...]

            def __init__(self) -> None:
                if vote_namespace == "vote-0001" and vote_attempt == 1:
                    self.audit_events = (
                        _checkpoint_audit_event("verify", candidate_present=True),
                    )
                elif vote_namespace == "vote-0001":
                    self.audit_events = (
                        AuditEvent(
                            sequence=1,
                            event_type="checkpoint_restored",
                            timestamp=datetime.now(UTC),
                            payload={
                                "cursor": "verify",
                                "logical_tool_calls": 0,
                                "step_count": 1,
                                "candidate_present": True,
                            },
                        ),
                    )
                else:
                    self.audit_events = ()

            async def run(self, selected_task: GaiaTask) -> TaskPrediction:
                assert selected_task.task_id == task.task_id
                if vote_namespace == "vote-0001" and vote_attempt == 1:
                    return TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        usage=Usage(total_tokens=5),
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.TIMEOUT,
                    )
                answer, total_tokens = {
                    "vote-0001": ("independent-a", 7),
                    "vote-0002": ("independent-b", 11),
                    "vote-0003": ("independent-b", 13),
                }[vote_namespace]
                return TaskPrediction(
                    task_id=task.task_id,
                    level=task.level,
                    answer=answer,
                    usage=Usage(total_tokens=total_tokens),
                    critic_verdict="approve",
                )

        return NamespacedOrchestrator()

    runner = EvaluationRunner(
        config=_config(tmp_path, votes=3, timeout_seconds=1.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
    )

    result = await runner._run_task(task)

    assert calls == [
        ("vote-0001", 1),
        ("vote-0001", 2),
        ("vote-0002", 1),
        ("vote-0003", 1),
    ]
    assert result.prediction.task_id == task.task_id
    assert result.prediction.answer == "independent-b"
    assert result.prediction.vote_count == 2
    assert result.prediction.usage == Usage(total_tokens=31)


@pytest.mark.asyncio
async def test_runner_cancelled_restored_attempt_retains_prior_cumulative_usage(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]

    class CancelledResumeFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, active_task: GaiaTask, vote_namespace: str) -> Any:
            assert active_task.task_id == task.task_id
            assert vote_namespace == "vote-0001"
            self.calls += 1
            call_number = self.calls

            class CancelledResumeOrchestrator:
                audit_events = (
                    (_checkpoint_audit_event("verify", candidate_present=True),)
                    if call_number == 1
                    else (
                        AuditEvent(
                            sequence=1,
                            event_type="checkpoint_restored",
                            timestamp=datetime.now(UTC),
                            payload={
                                "cursor": "verify",
                                "logical_tool_calls": 0,
                                "step_count": 1,
                                "candidate_present": True,
                            },
                        ),
                    )
                )

                async def run(self, selected_task: GaiaTask) -> TaskPrediction:
                    if call_number == 1:
                        return TaskPrediction(
                            task_id=selected_task.task_id,
                            level=selected_task.level,
                            answer=None,
                            usage=Usage(total_tokens=7),
                            critic_verdict="not_run",
                            error_category=TerminalErrorCategory.TIMEOUT,
                        )
                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")

            return CancelledResumeOrchestrator()

    factory = CancelledResumeFactory()
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=0.03),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
    )

    result = await runner._run_vote(task, vote_namespace="vote-0001")

    assert factory.calls == 2
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT
    assert result.prediction.usage.total_tokens >= 7
    assert result.prediction.usage.total_tokens == 7


@pytest.mark.asyncio
async def test_runner_critic_budget_exhaustion_at_verify_is_terminal(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        usage=Usage(total_tokens=4),
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.TIMEOUT,
                    ),
                    (
                        _checkpoint_audit_event("verify", candidate_present=True),
                        _stage_budget_audit_event("critic", "time"),
                    ),
                ),
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer="must not run",
                        critic_verdict="approve",
                    ),
                    (),
                ),
            ]
        ),
    )
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
        clock=clock,
    )

    result = await runner._run_vote(task, vote_namespace="vote-0001")

    assert factory.calls == 1
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT
    assert result.prediction.attempt_number == 1
    assert result.prediction.usage == Usage(total_tokens=4)


@pytest.mark.asyncio
async def test_runner_critic_provider_retry_exhaustion_at_verify_is_terminal(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    clock = _FakeClock()
    retry_event = AuditEvent(
        sequence=2,
        event_type="stage_retry_exhausted",
        timestamp=datetime.now(UTC),
        payload={"stage": "critic", "retry_category": "APIConnectionError"},
    )
    factory = _TimedOrchestratorFactory(
        clock,
        deque(
            [
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer=None,
                        usage=Usage(total_tokens=4),
                        critic_verdict="not_run",
                        error_category=TerminalErrorCategory.RETRY_EXHAUSTED,
                        retry_reason="provider_transient_exhausted",
                        retry_category="APIConnectionError",
                    ),
                    (
                        _checkpoint_audit_event("verify", candidate_present=True),
                        retry_event,
                    ),
                ),
                (
                    1.0,
                    TaskPrediction(
                        task_id=task.task_id,
                        level=task.level,
                        answer="must not run",
                        critic_verdict="approve",
                    ),
                    (),
                ),
            ]
        ),
    )
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=10.0),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
        clock=clock,
    )

    result = await runner._run_vote(task, vote_namespace="vote-0001")

    assert factory.calls == 1
    assert result.prediction.error_category == TerminalErrorCategory.RETRY_EXHAUSTED
    assert result.prediction.attempt_number == 1


@pytest.mark.asyncio
async def test_runner_vote_deadline_cancels_blocking_recovery_attempt(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]

    class DeadlineFactory:
        def __init__(self) -> None:
            self.calls = 0
            self.cancelled = False

        def __call__(self, active_task: GaiaTask) -> Any:
            self.calls += 1
            factory = self

            class DeadlineOrchestrator:
                audit_events: tuple[AuditEvent, ...] = ()

                async def run(self, selected_task: GaiaTask) -> TaskPrediction:
                    assert selected_task.task_id == active_task.task_id
                    if factory.calls == 1:
                        return TaskPrediction(
                            task_id=selected_task.task_id,
                            level=selected_task.level,
                            answer=None,
                            critic_verdict="not_run",
                            error_category=TerminalErrorCategory.TIMEOUT,
                        )
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        factory.cancelled = True
                        raise
                    raise AssertionError("unreachable")

            return DeadlineOrchestrator()

    factory = DeadlineFactory()
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=0.05),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
    )
    started = asyncio.get_running_loop().time()

    result = await runner._run_vote(task)

    assert factory.calls == 2
    assert factory.cancelled is True
    assert asyncio.get_running_loop().time() - started < 0.15
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT
    assert result.prediction.attempt_number == 2
    assert result.prediction.recovery_count == 1


@pytest.mark.asyncio
async def test_runner_deadline_does_not_wait_for_cancellation_suppression(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    cleanup_finished = asyncio.Event()

    class CancellationSuppressingFactory:
        def __init__(self) -> None:
            self.calls = 0
            self.cancelled = False

        def __call__(self, active_task: GaiaTask, vote_namespace: str) -> Any:
            assert active_task.task_id == task.task_id
            assert vote_namespace == "vote-0001"
            self.calls += 1
            factory = self

            class CancellationSuppressingOrchestrator:
                audit_events: tuple[AuditEvent, ...] = ()

                async def run(self, selected_task: GaiaTask) -> TaskPrediction:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        factory.cancelled = True
                        await asyncio.sleep(0.08)
                        cleanup_finished.set()
                    return TaskPrediction(
                        task_id=selected_task.task_id,
                        level=selected_task.level,
                        answer="late result",
                        critic_verdict="approve",
                    )

            return CancellationSuppressingOrchestrator()

    factory = CancellationSuppressingFactory()
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=0.02),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
    )
    started = asyncio.get_running_loop().time()

    result = await runner._run_vote(task, vote_namespace="vote-0001")
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.06
    assert factory.cancelled is True
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT
    await asyncio.wait_for(cleanup_finished.wait(), timeout=0.15)


@pytest.mark.asyncio
async def test_runner_deadline_starts_before_synchronous_factory_work(
    tmp_path: Path,
) -> None:
    task = _fixture_tasks()[0][0]
    factory_finished = threading.Event()

    class SlowFactory:
        def __init__(self) -> None:
            self.calls = 0
            self.run_calls = 0

        def __call__(self, active_task: GaiaTask, vote_namespace: str) -> Any:
            assert active_task.task_id == task.task_id
            assert vote_namespace == "vote-0001"
            self.calls += 1
            time.sleep(0.08)
            factory = self

            class MustNotRunOrchestrator:
                audit_events: tuple[AuditEvent, ...] = ()

                async def run(self, selected_task: GaiaTask) -> TaskPrediction:
                    factory.run_calls += 1
                    return TaskPrediction(
                        task_id=selected_task.task_id,
                        level=selected_task.level,
                        answer="too late",
                        critic_verdict="approve",
                    )

            factory_finished.set()
            return MustNotRunOrchestrator()

    factory = SlowFactory()
    runner = EvaluationRunner(
        config=_config(tmp_path, timeout_seconds=0.02),
        dataset_loader=lambda _config: ([task], "unused"),
        orchestrator_factory=factory,
        sleep=lambda _delay: asyncio.sleep(0),
    )

    started = asyncio.get_running_loop().time()

    result = await runner._run_vote(task, vote_namespace="vote-0001")
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.06
    assert factory.calls == 1
    assert factory.run_calls == 0
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT
    assert result.prediction.attempt_number == 1
    assert await asyncio.to_thread(factory_finished.wait, 0.15) is True


@pytest.mark.asyncio
async def test_runner_recovers_one_typed_provider_exhaustion_and_persists_safe_counters(
    tmp_path: Path,
) -> None:
    tasks, revision = _fixture_tasks()
    selected_ids = ("synthetic-l1-a", "synthetic-l2-b", "synthetic-l3-b")
    calls: list[str] = []
    outcomes: dict[str, deque[object]] = {}
    for task_id in selected_ids:
        task = next(item for item in tasks if item.task_id == task_id)
        scripted: list[object] = []
        if task_id == selected_ids[0]:
            scripted.append(
                TaskPrediction(
                    task_id=task_id,
                    level=task.level,
                    answer=None,
                    critic_verdict="not_run",
                    error_category=TerminalErrorCategory.RETRY_EXHAUSTED,
                    retry_reason="provider_transient_exhausted",
                    retry_category="APIConnectionError",
                    provider_attempt_count=4,
                )
            )
        scripted.append(
            TaskPrediction(
                task_id=task_id,
                level=task.level,
                answer="offline answer",
                critic_verdict="approve",
                evidence_ids=("evidence-0001",),
            )
        )
        outcomes[task_id] = deque(scripted)

    runner = EvaluationRunner(
        config=_config(tmp_path),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=lambda task: _ScriptedOrchestrator(
            task, outcomes[task.task_id], calls
        ),
        sleep=lambda _delay: asyncio.sleep(0),
    )

    await runner.run()

    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    recovered = predictions[0]
    assert calls.count(selected_ids[0]) == 2
    assert recovered["attempt_number"] == 2
    assert recovered["recovery_count"] == 1
    assert recovered["recovery_succeeded"] is True
    assert recovered["retry_reason"] == "provider_transient_exhausted"
    assert recovered["retry_category"] == "APIConnectionError"
    assert recovered["provider_attempt_count"] == 4
    audit = [
        json.loads(line)
        for line in (tmp_path / "tasks" / selected_ids[0] / "audit.jsonl")
        .read_text()
        .splitlines()
    ]
    assert audit[-1]["event_type"] == "task_recovery"
    assert audit[-1]["payload"] == {
        "attempt_number": 2,
        "recovery_succeeded": True,
        "retry_category": "APIConnectionError",
        "retry_reason": "provider_transient_exhausted",
    }


@pytest.mark.asyncio
async def test_runner_stops_after_one_failed_provider_exhaustion_recovery(tmp_path: Path) -> None:
    tasks, revision = _fixture_tasks()
    selected_ids = ("synthetic-l1-a", "synthetic-l2-b", "synthetic-l3-b")
    calls: list[str] = []
    outcomes: dict[str, deque[object]] = {}
    for task_id in selected_ids:
        task = next(item for item in tasks if item.task_id == task_id)
        if task_id == selected_ids[0]:
            scripted = [
                TaskPrediction(
                    task_id=task_id,
                    level=task.level,
                    answer=None,
                    critic_verdict="not_run",
                    error_category=TerminalErrorCategory.RETRY_EXHAUSTED,
                    retry_reason="provider_transient_exhausted",
                    retry_category="APIConnectionError",
                    provider_attempt_count=4,
                )
                for _ in range(2)
            ]
        else:
            scripted = [
                TaskPrediction(
                    task_id=task_id,
                    level=task.level,
                    answer="offline answer",
                    critic_verdict="approve",
                )
            ]
        outcomes[task_id] = deque(scripted)

    runner = EvaluationRunner(
        config=_config(tmp_path),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=lambda task: _ScriptedOrchestrator(
            task, outcomes[task.task_id], calls
        ),
        sleep=lambda _delay: asyncio.sleep(0),
    )

    await runner.run()

    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    exhausted = predictions[0]
    assert calls.count(selected_ids[0]) == 2
    assert exhausted["error_category"] == "retry_exhausted"
    assert exhausted["attempt_number"] == 2
    assert exhausted["recovery_count"] == 1
    assert exhausted["recovery_succeeded"] is False
    assert exhausted["provider_attempt_count"] == 8


@pytest.mark.asyncio
async def test_runner_does_not_retry_terminal_errors_and_continues(tmp_path: Path) -> None:
    tasks, revision = _fixture_tasks()
    selected_ids = ("synthetic-l1-a", "synthetic-l2-b", "synthetic-l3-b")
    calls: list[str] = []
    outcomes: dict[str, deque[object]] = {
        selected_ids[0]: deque(
            [
                TaskPrediction(
                    task_id=selected_ids[0],
                    level=1,
                    answer=None,
                    critic_verdict="not_run",
                    error_category=TerminalErrorCategory.RETRY_EXHAUSTED,
                )
            ]
        ),
        selected_ids[1]: deque(
            [TaskPrediction(task_id=selected_ids[1], level=2, answer="x", critic_verdict="approve")]
        ),
        selected_ids[2]: deque(
            [TaskPrediction(task_id=selected_ids[2], level=3, answer="x", critic_verdict="approve")]
        ),
    }
    runner = EvaluationRunner(
        config=_config(tmp_path),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=lambda task: _ScriptedOrchestrator(task, outcomes[task.task_id], calls),
    )

    summary = await runner.run()
    predictions = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]

    assert summary.total == 3
    assert calls == list(selected_ids)
    assert [item["task_id"] for item in predictions] == list(selected_ids)
    assert predictions[0]["error_category"] == "retry_exhausted"
    assert predictions[0]["correct"] is False
    assert predictions[1]["error_category"] is None


@pytest.mark.asyncio
async def test_runner_uses_configured_odd_votes_and_stops_only_after_strict_majority(
    tmp_path: Path,
) -> None:
    tasks, revision = _fixture_tasks()
    calls: list[str] = []
    outcomes: dict[str, deque[object]] = defaultdict(deque)
    for task in tasks:
        for answer in ("winner", "other", "third", "winner", "winner"):
            outcomes[task.task_id].append(
                TaskPrediction(
                    task_id=task.task_id,
                    level=task.level,
                    answer=answer,
                    critic_verdict="approve",
                )
            )
    runner = EvaluationRunner(
        config=_config(tmp_path, votes=5),
        dataset_loader=lambda _config: (tasks, revision),
        orchestrator_factory=lambda task: _ScriptedOrchestrator(task, outcomes[task.task_id], calls),
    )

    await runner.run()

    assert calls.count("synthetic-l1-a") == 5
    predictions = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert all(item["answer"] == "winner" for item in predictions)
    assert all(item["vote_count"] == 3 for item in predictions)
    assert all(item["attempt_number"] == 1 for item in predictions)
    assert all(item["recovery_count"] == 0 for item in predictions)


def test_resume_skips_complete_records_and_plain_append_repairs_incomplete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)
    output_dir = tmp_path / "run"
    first = cli_runner.invoke(app, _evaluate_arguments(output_dir))
    assert first.exit_code == 0, first.output
    original = (output_dir / "predictions.jsonl").read_text(encoding="utf-8")

    skipped = cli_runner.invoke(app, _evaluate_arguments(output_dir, "--resume"))
    assert skipped.exit_code == 0, skipped.output
    assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8") == original

    records = original.splitlines()
    records[1] = '{"task_id":"synthetic-l2-b","level":2}'
    (output_dir / "predictions.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    audit_path = output_dir / "tasks" / "synthetic-l2-b" / "audit.jsonl"
    original_audit_count = len(audit_path.read_text(encoding="utf-8").splitlines())
    repaired = cli_runner.invoke(app, _evaluate_arguments(output_dir, "--resume"))

    assert repaired.exit_code == 0, repaired.output
    fixed = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
    assert [item["task_id"] for item in fixed] == [
        "synthetic-l1-a",
        "synthetic-l2-b",
        "synthetic-l3-b",
    ]
    audit_sequences = [
        json.loads(line)["sequence"] for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(audit_sequences) > original_audit_count
    assert audit_sequences == list(range(1, len(audit_sequences) + 1))
    resumed_events = [
        json.loads(line)["event_type"]
        for line in audit_path.read_text(encoding="utf-8").splitlines()[original_audit_count:]
    ]
    assert "tool_call_restored" in resumed_events


def test_outputs_stay_contained_and_exclude_secret_and_gated_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)
    output_dir = tmp_path / "chosen" / "run"

    result = cli_runner.invoke(app, _evaluate_arguments(output_dir))

    assert result.exit_code == 0, result.output
    assert list(tmp_path.rglob("manifest.json")) == [output_dir / "manifest.json"]
    artifacts = "".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    for forbidden in (
        "sk-fake-credential",
        "hf_fake_credential",
        '"Question"',
        '"Final answer"',
        '"question"',
        '"final_answer"',
        '"attachment_path"',
    ):
        assert forbidden not in artifacts


def test_verify_results_reloads_artifacts_and_rejects_mismatch_privacy_and_incompleteness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    _inject_offline_runner(monkeypatch)
    output_dir = tmp_path / "run"
    evaluated = cli_runner.invoke(app, _evaluate_arguments(output_dir))
    assert evaluated.exit_code == 0, evaluated.output
    verified = cli_runner.invoke(app, ["verify-results", str(output_dir)])
    assert verified.exit_code == 0, verified.output

    summary_path = output_dir / "summary.json"
    original_summary = summary_path.read_text(encoding="utf-8")
    summary = json.loads(original_summary)
    summary["correct"] += 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    mismatch = cli_runner.invoke(app, ["verify-results", str(output_dir)])
    assert mismatch.exit_code == 2

    summary_path.write_text(original_summary, encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original_manifest)
    manifest["question"] = "gated content"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    privacy = cli_runner.invoke(app, ["verify-results", str(output_dir)])
    assert privacy.exit_code == 2
    predictions_before_resume = (output_dir / "predictions.jsonl").read_text(encoding="utf-8")
    private_resume = cli_runner.invoke(app, _evaluate_arguments(output_dir, "--resume"))
    assert private_resume.exit_code == 2
    assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8") == predictions_before_resume

    manifest_path.write_text(original_manifest, encoding="utf-8")
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    incomplete = cli_runner.invoke(app, ["verify-results", str(output_dir)])
    assert incomplete.exit_code == 2
