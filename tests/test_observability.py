from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

import pytest

from gaia_agent.models import GaiaTask, TaskPrediction, Usage
from gaia_agent.observability import Observability, ObservabilityConfig
from gaia_agent.runner import EvaluationRunner


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


def test_flush_failure_is_swallowed() -> None:
    client = _FakeClient(fail_flush=True)
    config = ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk")
    observability = Observability(config, client=client)

    observability.flush()

    assert client.flush_count == 1


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

    def flush(self) -> None:
        self.flush_count += 1


@pytest.mark.asyncio
async def test_runner_owns_task_trace_lifecycle() -> None:
    task = GaiaTask(task_id="trace-task", question="private", level=1, final_answer="answer")
    config = __import__("gaia_agent.config", fromlist=["RunConfig"]).RunConfig(
        openai_api_key="sk-test",
        hf_token="hf-test",
        sample_size=3,
    )
    prediction = TaskPrediction(
        task_id=task.task_id,
        level=task.level,
        answer="answer",
        critic_verdict="approve",
    )

    class _Orchestrator:
        audit_events: tuple[()] = ()

        async def run(self, active_task: GaiaTask) -> TaskPrediction:
            return prediction

    observability = _RecordingObservability()
    runner = EvaluationRunner(
        config=config,
        dataset_loader=lambda _config: ([task], "revision"),
        orchestrator_factory=lambda _task: _Orchestrator(),
        observability=observability,  # type: ignore[arg-type]
    )

    await runner._run_task(task)

    assert observability.started == [
        {
            "task_id": "trace-task",
            "level": 1,
            "model": "deepseek-v4-flash",
            "reasoning_effort": "medium",
            "run_index": 0,
            "evaluation_mode": "test",
        }
    ]
    assert observability.traces[0].closed == ["ok"]
