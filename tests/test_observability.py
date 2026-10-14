from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from gaia_agent.models import GaiaTask, TaskPrediction, Usage
from gaia_agent.observability import Observability, ObservabilityConfig, TaskTraceOutcome
from gaia_agent.runner import EvaluationRunner


def test_response_diagnostics_summary_rejects_content_and_unknown_values():
    from gaia_agent.observability import _sanitize_summary

    diagnostic = {
        "phase": "repair",
        "code": "schema",
        "response_status": "incomplete",
        "incomplete_reason": "max_output_tokens",
        "raw_output": "PRIVATE",
    }
    safe = _sanitize_summary(
        {"response_diagnostics": [diagnostic, {**diagnostic, "code": "PRIVATE"}]}
    )
    assert safe == {
        "response_diagnostics": [
            {key: value for key, value in diagnostic.items() if key != "raw_output"}
        ]
    }


class _FakeObservation:
    def __init__(self, *, name: str, as_type: str, kwargs: dict[str, Any]) -> None:
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> None:
        self.ended = True
        if kwargs:
            self.updates.append(kwargs)


class _FakeObservationContext(AbstractContextManager[_FakeObservation]):
    def __init__(self, observation: _FakeObservation) -> None:
        self.observation = observation

    def __enter__(self) -> _FakeObservation:
        return self.observation

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.observation.ended = True


class _FakeClient:
    def __init__(self, *, fail_flush: bool = False) -> None:
        self.observations: list[_FakeObservation] = []
        self.fail_flush = fail_flush
        self.flush_count = 0

    def start_as_current_observation(self, *, as_type: str, name: str, **kwargs: Any):
        observation = _FakeObservation(name=name, as_type=as_type, kwargs=kwargs)
        self.observations.append(observation)
        return _FakeObservationContext(observation)

    def flush(self) -> None:
        self.flush_count += 1
        if self.fail_flush:
            raise RuntimeError("transport unavailable")


def _assert_payload_excludes(payload: object, forbidden: set[str]) -> None:
    """Assert every telemetry value remains free of private test sentinels."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            _assert_payload_excludes(key, forbidden)
            _assert_payload_excludes(value, forbidden)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _assert_payload_excludes(value, forbidden)
    elif isinstance(payload, str):
        assert not any(sentinel in payload for sentinel in forbidden)


def test_langfuse_summaries_keep_only_bounded_stage_search_and_recovery_metadata() -> None:
    """Changing the summary allowlist to pass content or omit operational recovery data is a bug."""

    client = _FakeClient()
    trace = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
    ).start_task(
        "task-safe-operational-summary",
        level=2,
        model="test-model",
        reasoning_effort="max",
        evaluation_mode="test",
    )
    private = {
        "QUESTION-SENTINEL",
        "ANSWER-SENTINEL",
        "CANDIDATE-SENTINEL",
        "PROMPT-SENTINEL",
        "https://private.example/SENTINEL",
        "EVIDENCE-EXCERPT-SENTINEL",
        "RAW-ERROR-SENTINEL",
        "CHECKPOINT-PAYLOAD-SENTINEL",
        "/private/checkpoint/SENTINEL",
        "NOTEBOOK-CONTENT-SENTINEL",
        "SECRET-VALUE-SENTINEL",
    }

    stage = trace.span(
        "specialist",
        {
            "stage": "QUESTION-SENTINEL",
            "specialist": "CANDIDATE-SENTINEL",
            "tool": "https://private.example/SENTINEL",
        },
    )
    stage.end(
        status="ok",
        summary={
            "stage": "specialist",
            "budget_seconds": 120.0,
            "elapsed_seconds": 91.5,
            "remaining_bucket": "low",
            "logical_call_count": 4,
            "global_call_count": 7,
            "executed_call_count": 3,
            "restored_call_count": 1,
            "question": "QUESTION-SENTINEL",
            "answer": "ANSWER-SENTINEL",
            "candidate": "CANDIDATE-SENTINEL",
            "prompt": "PROMPT-SENTINEL",
            "url": "https://private.example/SENTINEL",
            "evidence": "EVIDENCE-EXCERPT-SENTINEL",
            "error": "RAW-ERROR-SENTINEL",
            "checkpoint": "CHECKPOINT-PAYLOAD-SENTINEL",
            "checkpoint_path": "/private/checkpoint/SENTINEL",
            "notebook": "NOTEBOOK-CONTENT-SENTINEL",
            "secret": "SECRET-VALUE-SENTINEL",
        },
    )
    search = trace.span("mcp-tool", {"tool": "web_search"})
    search.end(
        status="error",
        metadata={
            "provider": "serper",
            "category": "timeout",
            "attempt_count": 2,
            "fallback_used": True,
            "error": "RAW-ERROR-SENTINEL",
        },
    )
    trace.finish(
        TaskTraceOutcome(
            answer_present=True,
            evidence_count=1,
            critic_verdict="approve",
            error_category=None,
            correct=True,
            checkpoint_restored=True,
            recovery_source="checkpoint",
            terminal_route="checkpoint_restored",
        )
    )

    assert client.observations[1].updates[-1]["output"] == {
        "reasoning_summary": {
            "stage": "specialist",
            "budget_seconds": 120.0,
            "elapsed_seconds": 91.5,
            "remaining_bucket": "low",
            "logical_call_count": 4,
            "global_call_count": 7,
            "executed_call_count": 3,
            "restored_call_count": 1,
        }
    }
    assert client.observations[2].updates[-1]["output"] == {
        "reasoning_summary": {
            "provider": "serper",
            "category": "timeout",
            "attempt_count": 2,
            "fallback_used": True,
        }
    }
    root_output = client.observations[0].updates[-1]["output"]
    assert root_output["checkpoint_restored"] is True
    assert root_output["recovery_source"] == "checkpoint"
    _assert_payload_excludes(
        [(observation.kwargs, observation.updates) for observation in client.observations],
        private,
    )


@pytest.mark.asyncio
async def test_runner_marks_root_checkpoint_recovery_from_the_checkpoint_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the restored-checkpoint event with a tool-cache event is a telemetry bug."""

    from gaia_agent import runner as runner_module
    from gaia_agent.config import RunConfig

    tasks = [
        GaiaTask(
            task_id=f"checkpoint-trace-{level}", question="private", level=level, final_answer="a"
        )
        for level in (1, 2, 3)
    ]

    class _Orchestrator:
        audit_events = (type("_CheckpointEvent", (), {"event_type": "checkpoint_restored"})(),)

        async def run(self, task: GaiaTask) -> TaskPrediction:
            return TaskPrediction(
                task_id=task.task_id,
                level=task.level,
                answer="a",
                critic_verdict="approve",
            )

    client = _FakeClient()
    monkeypatch.setattr(runner_module, "sample_stratified", lambda *_args: tasks)
    monkeypatch.setattr(runner_module, "question_scorer", lambda *_args: True)
    await EvaluationRunner(
        config=RunConfig(
            openai_api_key="sk-test",
            hf_token="hf-test",
            sample_size=3,
            output_dir=tmp_path / "run",
        ),
        dataset_loader=lambda _config: (tasks, "revision"),
        orchestrator_factory=lambda _task: _Orchestrator(),
        observability=Observability(
            ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
        ),
    )._run()

    root_output = client.observations[0].updates[-1]["output"]
    assert root_output["checkpoint_restored"] is True
    assert root_output["recovery_source"] == "checkpoint"


def test_observability_defaults_to_noop_without_both_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    config = ObservabilityConfig.from_env()
    observability = Observability(config)
    trace = observability.start_task(
        "task-1",
        level=1,
        model="test-model",
        reasoning_effort="low",
        run_index=0,
        evaluation_mode="test",
    )
    span = trace.span("planner", {"question": "must not be sent"})
    span.end(status="ok", metadata={"answer": "must not be sent"})
    trace.close()
    observability.flush()

    assert not observability.enabled
    assert observability.client is None


def test_observability_is_enabled_only_with_keys_and_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    assert ObservabilityConfig.from_env().active

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert not ObservabilityConfig.from_env().active

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not ObservabilityConfig.from_env().active


def test_langfuse_adapter_sanitizes_metadata_and_records_usage() -> None:
    client = _FakeClient()
    config = ObservabilityConfig(
        enabled=True,
        public_key="pk-test",
        secret_key="sk-test",
        base_url="https://example.test",
        project_name="gaia-test",
    )
    observability = Observability(config, client=client)
    trace = observability.start_task(
        "task-1",
        level=2,
        model="test-model",
        reasoning_effort="max",
        run_index=1,
        evaluation_mode="test",
    )
    span = trace.generation(
        "provider-call",
        {
            "model": "test-model",
            "parsed": True,
            "question": "secret question",
            "prompt": "secret prompt",
        },
    )
    span.end(
        status="ok",
        metadata={"output": "secret output", "refused": False},
        usage=Usage(input_tokens=3, output_tokens=4, reasoning_tokens=5, total_tokens=7),
    )
    trace.close()

    assert [item.name for item in client.observations] == ["gaia-task", "provider-call"]
    root, generation = client.observations
    assert root.kwargs["metadata"] == {
        "task_id": "task-1",
        "level": 2,
        "model": "test-model",
        "reasoning_effort": "max",
        "run_index": 1,
        "evaluation_mode": "test",
        "project_name": "gaia-test",
    }
    assert generation.kwargs["metadata"] == {"model": "test-model", "parsed": True}
    assert generation.updates[-1]["usage_details"] == {
        "input_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 5,
        "total_tokens": 7,
    }
    assert all(
        forbidden not in repr(item.kwargs) and forbidden not in repr(item.updates)
        for item in client.observations
        for forbidden in ("secret question", "secret prompt", "secret output")
    )


def test_trace_decision_metadata_rejects_content_bearing_or_unbounded_values() -> None:
    client = _FakeClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
    )
    trace = observability.start_task(
        "task-safe-decisions",
        level=1,
        model="test-model",
        reasoning_effort="max",
        run_index=0,
        evaluation_mode="test",
    )
    span = trace.span(
        "mcp-tool",
        {
            "provider": "https://private.test/provider",
            "category": "classified body",
            "retryable": True,
            "attempt_count": 10_000_000,
            "http_status": 999,
            "circuit_open_count": -1,
            "fallback_attempted": True,
        },
    )
    span.end(status="error")
    trace.close()

    metadata = client.observations[-1].kwargs["metadata"]
    assert metadata == {"retryable": True, "fallback_attempted": True}
    assert "private.test" not in repr(client.observations)
    assert "classified" not in repr(client.observations)


def test_langfuse_adapter_records_only_safe_structural_io() -> None:
    client = _FakeClient()
    config = ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk")
    observability = Observability(config, client=client)
    trace = observability.start_task(
        "task-io",
        level=1,
        model="test-model",
        reasoning_effort="low",
        run_index=0,
        evaluation_mode="test",
    )

    span = trace.span(
        "mcp-tool",
        {"tool": "read_webpage"},
        input={
            "argument_count": 1,
            "argument_keys": ["url"],
            "url": "https://private.example/should-not-send",
        },
    )
    span.end(
        status="ok",
        output={
            "result_key_count": 2,
            "result_keys": ["text", "sources"],
            "text": "private webpage content",
        },
    )

    observation = client.observations[-1]
    assert observation.kwargs["input"] == {
        "argument_count": 1,
        "argument_keys": ["url"],
    }
    assert observation.updates[-1]["output"] == {
        "result_key_count": 2,
        "result_keys": ["text", "sources"],
    }
    assert "private.example" not in repr(observation.kwargs)
    assert "private webpage content" not in repr(observation.updates)


def test_langfuse_adapter_records_safe_reasoning_summary_only() -> None:
    client = _FakeClient()
    config = ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk")
    observability = Observability(config, client=client)
    trace = observability.start_task(
        "task-summary",
        level=2,
        model="test-model",
        reasoning_effort="max",
        run_index=0,
        evaluation_mode="test",
    )

    span = trace.span("critic", {"stage": "verify"})
    span.end(
        status="ok",
        summary={
            "route": "lookup",
            "specialist": "lookup",
            "step_count": 2,
            "evidence_count": 3,
            "evidence_kinds": ["source", "calculation"],
            "tool_names": ["web_search", "read_webpage"],
            "verdict": "approve",
            "repair_attempted": False,
            "fallback_used": False,
            "candidate_present": True,
            "question": "must not be sent",
            "candidate": "must not be sent",
        },
    )

    observation = client.observations[-1]
    assert observation.updates[-1]["output"] == {
        "reasoning_summary": {
            "route": "lookup",
            "specialist": "lookup",
            "step_count": 2,
            "evidence_count": 3,
            "evidence_kinds": ["source", "calculation"],
            "tool_names": ["web_search", "read_webpage"],
            "verdict": "approve",
            "repair_attempted": False,
            "fallback_used": False,
            "candidate_present": True,
        }
    }
    assert "must not be sent" not in repr(observation.updates)


def test_langfuse_span_set_summary_publishes_on_context_exit() -> None:
    client = _FakeClient()
    config = ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk")
    observability = Observability(config, client=client)
    trace = observability.start_task(
        "task-summary-context",
        level=1,
        model="test-model",
        reasoning_effort="low",
        run_index=0,
        evaluation_mode="test",
    )

    with trace.span("planner", {"stage": "planner"}) as span:
        span.set_summary({"route": "computation", "step_count": 1})

    assert client.observations[-1].updates[-1]["output"] == {
        "reasoning_summary": {"route": "computation", "step_count": 1}
    }


def test_generation_records_configured_model_cost_details() -> None:
    client = _FakeClient()
    config = ObservabilityConfig(
        enabled=True,
        public_key="pk",
        secret_key="sk",
        input_cost_per_million="1.25",
        output_cost_per_million="2",
    )
    observability = Observability(config, client=client)
    trace = observability.start_task(
        "task-cost",
        level=1,
        model="test-model",
        reasoning_effort="max",
        run_id="gaia-run",
        attempt_number=1,
        evaluation_mode="test",
    )

    generation = trace.generation("provider-generation")
    generation.end(
        status="ok",
        usage=Usage(input_tokens=2_000_000, output_tokens=3_000_000),
    )

    assert client.observations[-1].updates[-1]["cost_details"] == {
        "input": 2.5,
        "output": 6.0,
    }


def test_generation_omits_non_finite_cost_details() -> None:
    client = _FakeClient()
    observability = Observability(
        ObservabilityConfig(
            enabled=True,
            public_key="pk",
            secret_key="sk",
            input_cost_per_million=Decimal("1e999999"),
        ),
        client=client,
    )
    trace = observability.start_task(
        "task-cost-overflow",
        level=1,
        model="test-model",
        reasoning_effort="max",
        evaluation_mode="test",
    )

    trace.generation("provider-generation").end(status="ok", usage=Usage(input_tokens=1))

    assert "cost_details" not in client.observations[-1].updates[-1]


def test_flush_reports_a_safe_warning_on_sdk_failure(caplog: pytest.LogCaptureFixture) -> None:
    client = _FakeClient(fail_flush=True)
    config = ObservabilityConfig(
        enabled=True,
        public_key="pk-test-credential",
        secret_key="sk-test-credential",
    )
    observability = Observability(config, client=client)

    with caplog.at_level("WARNING"):
        success = observability.flush()

    assert client.flush_count == 1
    assert not success
    assert [record.message for record in caplog.records] == ["Langfuse flush warning"]
    assert "credential" not in caplog.text


def test_flush_reports_success() -> None:
    client = _FakeClient()
    config = ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk")

    assert Observability(config, client=client).flush()


class _RecordingTrace:
    def __init__(self) -> None:
        self.closed: list[str] = []

    def close(self, *, status: str = "ok", metadata: dict[str, object] | None = None) -> None:
        self.closed.append(status)


class _RecordingObservability:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.traces: list[_RecordingTrace] = []
        self.flush_count = 0

    def start_task(self, task_id: str, **metadata: object) -> _RecordingTrace:
        self.started.append({"task_id": task_id, **metadata})
        trace = _RecordingTrace()
        self.traces.append(trace)
        return trace

    def flush(self) -> bool:
        self.flush_count += 1
        return True


@pytest.mark.asyncio
async def test_runner_finishes_one_task_root_after_scoring_and_persistence(
    tmp_path: __import__("pathlib").Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gaia_agent import runner as runner_module

    tasks = [
        GaiaTask(
            task_id=f"trace-task-{level}", question="private", level=level, final_answer="answer"
        )
        for level in (1, 2, 3)
    ]
    config = __import__("gaia_agent.config", fromlist=["RunConfig"]).RunConfig(
        openai_api_key="sk-test",
        hf_token="hf-test",
        sample_size=3,
        output_dir=tmp_path / "run",
    )
    prediction = TaskPrediction(
        task_id="unused",
        level=1,
        answer="answer",
        critic_verdict="approve",
    )

    class _Orchestrator:
        audit_events: tuple[()] = ()

        async def run(self, active_task: GaiaTask) -> TaskPrediction:
            return prediction.model_copy(
                update={"task_id": active_task.task_id, "level": active_task.level}
            )

    client = _FakeClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
    )
    runner = EvaluationRunner(
        config=config,
        dataset_loader=lambda _config: (tasks, "revision"),
        orchestrator_factory=lambda _task: _Orchestrator(),
        observability=observability,
    )

    monkeypatch.setattr(runner_module, "sample_stratified", lambda *_args: tasks)

    def score(answer: str | None, reference: str) -> bool:
        assert answer == "answer"
        assert reference == "answer"
        roots = [
            observation for observation in client.observations if observation.name == "gaia-task"
        ]
        assert not roots[-1].ended
        return True

    monkeypatch.setattr(runner_module, "question_scorer", score)

    await runner._run()

    root, vote = client.observations[:2]
    assert root.name == "gaia-task"
    assert root.kwargs["metadata"] == {
        "task_id": "trace-task-1",
        "level": 1,
        "model": "gpt-6.1-sol",
        "reasoning_effort": "low",
        "run_id": "gaia-seed42-n3",
        "attempt_number": 1,
        "evaluation_mode": "test",
    }
    assert vote.name == "vote"
    assert vote.kwargs["metadata"] == {"run_index": 0}
    assert root.ended
    assert root.updates[-1]["output"] == {
        "answer_present": True,
        "evidence_count": 0,
        "critic_verdict": "approve",
        "error_category": None,
        "correct": True,
        "provider_retry_count": 0,
        "tool_success_count": 0,
        "tool_failure_count": 0,
        "provider_latency_ms": 0.0,
        "tool_latency_ms": 0.0,
        "checkpoint_restored": False,
        "attempt_number": 1,
        "recovery_count": 0,
        "recovery_succeeded": False,
        "recovery_source": "not_needed",
        "retry_reason": None,
        "retry_category": None,
        "terminal_route": "submit",
        "reasoning_summary": {
            "route": "submit",
            "recovery": "not_needed",
            "stages": [],
        },
    }


@pytest.mark.asyncio
async def test_runner_traces_resumed_selected_tasks_with_persisted_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gaia_agent import runner as runner_module
    from gaia_agent.config import RunConfig

    tasks = [
        GaiaTask(
            task_id=f"resume-task-{level}", question="private", level=level, final_answer="answer"
        )
        for level in (1, 2, 3)
    ]
    config = RunConfig(
        openai_api_key="sk-test",
        hf_token="hf-test",
        sample_size=3,
        output_dir=tmp_path / "run",
    )

    class _Orchestrator:
        audit_events: tuple[()] = ()

        async def run(self, task: GaiaTask) -> TaskPrediction:
            return TaskPrediction(
                task_id=task.task_id,
                level=task.level,
                answer="answer",
                critic_verdict="approve",
            )

    monkeypatch.setattr(runner_module, "sample_stratified", lambda *_args: tasks)
    monkeypatch.setattr(runner_module, "question_scorer", lambda *_args: True)
    first = EvaluationRunner(
        config=config,
        dataset_loader=lambda _config: (tasks, "revision"),
        orchestrator_factory=lambda _task: _Orchestrator(),
    )
    await first._run()

    client = _FakeClient()
    resumed = EvaluationRunner(
        config=config.model_copy(update={"resume": True}),
        dataset_loader=lambda _config: (tasks, "revision"),
        orchestrator_factory=lambda _task: _Orchestrator(),
        observability=Observability(
            ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
        ),
    )
    await resumed._run()

    roots = [observation for observation in client.observations if observation.name == "gaia-task"]
    assert [root.kwargs["metadata"]["task_id"] for root in roots] == [
        task.task_id for task in tasks
    ]
    assert all(root.ended for root in roots)
    assert all(root.updates[-1]["output"]["correct"] is True for root in roots)


@pytest.mark.asyncio
async def test_runner_root_records_recovered_attempt_and_resume_reuses_that_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gaia_agent import runner as runner_module
    from gaia_agent.config import RunConfig
    from gaia_agent.models import TerminalErrorCategory

    tasks = [
        GaiaTask(
            task_id=f"recover-trace-{level}", question="private", level=level, final_answer="a"
        )
        for level in (1, 2, 3)
    ]
    config = RunConfig(
        openai_api_key="sk-test",
        hf_token="hf-test",
        sample_size=3,
        output_dir=tmp_path / "run",
    )
    outcomes = {
        task.task_id: deque(
            [
                *(
                    [
                        TaskPrediction(
                            task_id=task.task_id,
                            level=task.level,
                            answer=None,
                            critic_verdict="not_run",
                            error_category=TerminalErrorCategory.RETRY_EXHAUSTED,
                            retry_reason="provider_transient_exhausted",
                            retry_category="APIConnectionError",
                            provider_attempt_count=4,
                        )
                    ]
                    if task.level == 1
                    else []
                ),
                TaskPrediction(
                    task_id=task.task_id,
                    level=task.level,
                    answer="a",
                    critic_verdict="approve",
                ),
            ]
        )
        for task in tasks
    }

    class _Orchestrator:
        audit_events: tuple[()] = ()

        def __init__(self, task: GaiaTask) -> None:
            self.task = task

        async def run(self, _task: GaiaTask) -> TaskPrediction:
            return outcomes[self.task.task_id].popleft()

    monkeypatch.setattr(runner_module, "sample_stratified", lambda *_args: tasks)
    monkeypatch.setattr(runner_module, "question_scorer", lambda *_args: True)
    client = _FakeClient()
    await EvaluationRunner(
        config=config,
        dataset_loader=lambda _config: (tasks, "revision"),
        orchestrator_factory=_Orchestrator,
        observability=Observability(
            ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
        ),
        sleep=lambda _delay: __import__("asyncio").sleep(0),
    )._run()

    root = next(item for item in client.observations if item.name == "gaia-task")
    output = root.updates[-1]["output"]
    assert output["attempt_number"] == 2
    assert output["recovery_count"] == 1
    assert output["recovery_succeeded"] is True
    assert output["terminal_route"] == "recovered_submit"
    assert output["reasoning_summary"] == {
        "route": "recovered_submit",
        "recovery": "succeeded",
        "stages": ["recovery"],
    }
    assert root.updates[-1]["metadata"]["attempt_number"] == 2

    resumed_client = _FakeClient()
    await EvaluationRunner(
        config=config.model_copy(update={"resume": True}),
        dataset_loader=lambda _config: (tasks, "revision"),
        orchestrator_factory=_Orchestrator,
        observability=Observability(
            ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
            client=resumed_client,
        ),
    )._run()
    resumed_root = next(item for item in resumed_client.observations if item.name == "gaia-task")
    assert resumed_root.kwargs["metadata"]["attempt_number"] == 2
    assert resumed_root.updates[-1]["output"]["terminal_route"] == "recovered_submit"
    assert resumed_root.updates[-1]["output"]["checkpoint_restored"] is False
    assert resumed_root.updates[-1]["output"]["recovery_source"] == "public_result"


@pytest.mark.asyncio
async def test_runner_records_safe_failure_when_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gaia_agent import runner as runner_module
    from gaia_agent.config import RunConfig

    tasks = [
        GaiaTask(
            task_id=f"persist-task-{level}", question="private", level=level, final_answer="answer"
        )
        for level in (1, 2, 3)
    ]
    config = RunConfig(
        openai_api_key="sk-test",
        hf_token="hf-test",
        sample_size=3,
        output_dir=tmp_path / "run",
    )

    class _Orchestrator:
        audit_events: tuple[()] = ()

        async def run(self, task: GaiaTask) -> TaskPrediction:
            return TaskPrediction(
                task_id=task.task_id,
                level=task.level,
                answer="answer",
                critic_verdict="approve",
            )

    client = _FakeClient()
    monkeypatch.setattr(runner_module, "sample_stratified", lambda *_args: tasks)
    monkeypatch.setattr(runner_module, "question_scorer", lambda *_args: True)
    monkeypatch.setattr(
        runner_module, "_persist_task_artifacts", lambda *_args: (_ for _ in ()).throw(OSError())
    )
    runner = EvaluationRunner(
        config=config,
        dataset_loader=lambda _config: (tasks, "revision"),
        orchestrator_factory=lambda _task: _Orchestrator(),
        observability=Observability(
            ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
        ),
    )

    with pytest.raises(OSError):
        await runner._run()

    outcome = client.observations[0].updates[-1]["output"]
    assert outcome["correct"] is False
    assert outcome["error_category"] == "retry_exhausted"
    assert outcome["terminal_route"] == "terminal_failure"


def test_task_trace_aggregates_provider_and_tool_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    from gaia_agent import observability as observability_module

    ticks = iter((0.0, 10.0, 12.0, 20.0, 25.0))
    monkeypatch.setattr(observability_module.time, "monotonic", lambda: next(ticks))
    client = _FakeClient()
    trace = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"), client=client
    ).start_task(
        "task-latency",
        level=1,
        model="test-model",
        reasoning_effort="max",
        evaluation_mode="test",
    )

    trace.generation("provider-generation").end(status="ok", usage=Usage(input_tokens=1))
    trace.span("mcp-tool").end(status="ok")
    trace.finish(
        TaskTraceOutcome(
            answer_present=False,
            evidence_count=0,
            critic_verdict="not_run",
            error_category="retry_exhausted",
            correct=False,
        )
    )

    output = client.observations[0].updates[-1]["output"]
    assert output["provider_latency_ms"] == 2000.0
    assert output["tool_latency_ms"] == 5000.0


def test_task_trace_outcome_bounds_counts_and_categories() -> None:
    outcome = TaskTraceOutcome(
        answer_present=True,
        evidence_count=-1,
        critic_verdict="private verdict",
        error_category="private category",
        correct=True,
        provider_retry_count=-2,
        tool_success_count=-3,
        tool_failure_count=-4,
        checkpoint_restored=False,
    )

    assert outcome.as_output() == {
        "answer_present": True,
        "evidence_count": 0,
        "critic_verdict": "not_run",
        "error_category": None,
        "correct": True,
        "provider_retry_count": 0,
        "tool_success_count": 0,
        "tool_failure_count": 0,
        "provider_latency_ms": 0.0,
        "tool_latency_ms": 0.0,
        "checkpoint_restored": False,
        "attempt_number": 1,
        "recovery_count": 0,
        "recovery_succeeded": False,
        "recovery_source": "not_needed",
        "retry_reason": None,
        "retry_category": None,
        "terminal_route": "terminal_failure",
    }


def test_restored_checkpoint_failure_keeps_terminal_route_and_recovery_orthogonal() -> None:
    """A restored checkpoint does not turn a later timeout into a successful route."""

    from gaia_agent.models import TerminalErrorCategory
    from gaia_agent.runner import _task_trace_outcome

    output = _task_trace_outcome(
        TaskPrediction(
            task_id="checkpoint-timeout",
            level=2,
            answer=None,
            critic_verdict="not_run",
            error_category=TerminalErrorCategory.TIMEOUT,
        ),
        checkpoint_restored=True,
    ).as_output()

    assert output["terminal_route"] == "terminal_failure"
    assert output["checkpoint_restored"] is True
    assert output["recovery_source"] == "checkpoint"
