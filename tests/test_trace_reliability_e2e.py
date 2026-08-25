from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError

from gaia_agent.agents import LookupAgent, normalize_plan
from gaia_agent.models import AgentPlan, GaiaTask, PlanStep, TaskPrediction
from gaia_agent.observability import Observability, ObservabilityConfig, TaskTraceOutcome
from gaia_agent.persistence import RunManifest, RunStore
from gaia_agent.providers import ModelRequest, ModelResponse, OpenAIProvider, ToolSpec
from gaia_agent.reliability import RetryPolicy
from gaia_agent.reporting import summarize, write_reports
from gaia_agent.research_notebook import ResearchNotebook


class _RecordedObservation:
    def __init__(self, kwargs: dict[str, object]) -> None:
        self.kwargs = kwargs
        self.updates: list[dict[str, object]] = []
        self.closed = False

    def update(self, **kwargs: object) -> None:
        self.updates.append(kwargs)


class _RecordedContext:
    def __init__(self, observation: _RecordedObservation) -> None:
        self._observation = observation

    def __enter__(self) -> _RecordedObservation:
        return self._observation

    def __exit__(self, *_: object) -> None:
        self._observation.closed = True


class _RecordingLangfuse:
    def __init__(self) -> None:
        self.observations: list[_RecordedObservation] = []

    def start_as_current_observation(self, **kwargs: object) -> _RecordedContext:
        observation = _RecordedObservation(dict(kwargs))
        self.observations.append(observation)
        return _RecordedContext(observation)

    def flush(self) -> None:
        return None


class _Response:
    def __init__(self, output: list[object] | None = None) -> None:
        self.id = "offline-response"
        self.output = output or []
        self.output_text = ""
        self.usage = None


class _FunctionCall:
    type = "function_call"
    call_id = "offline-call"
    name = "web_search"
    arguments = '{"query":"offline"}'


class _Executor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((name, arguments))
        return {"sources": []}


class _ResponsesClient:
    def __init__(self, queued: deque[object]) -> None:
        self.responses = self
        self.queued = queued
        self.calls = 0

    def with_options(self, *, max_retries: int) -> _ResponsesClient:
        assert max_retries == 0
        return self

    async def create(self, **_: object) -> _Response:
        self.calls += 1
        outcome = self.queued.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, _Response)
        return outcome


class _ScriptedProvider:
    supports_vision = True

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.calls = 0

    async def complete(
        self,
        request: ModelRequest,
        executor: object | None = None,
        *,
        usage_observer: object | None = None,
    ) -> ModelResponse:
        del request, executor
        self.calls += 1
        response = self._responses.popleft()
        if callable(usage_observer):
            usage_observer(response.usage)
        return response


class _NoToolClient:
    tool_specs = (
        ToolSpec(
            name="web_search",
            description="Offline search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        ToolSpec(
            name="read_webpage",
            description="Offline page read",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}},
        ),
    )


def _task(index: int) -> GaiaTask:
    return GaiaTask(
        task_id=f"offline-{index}",
        question="Who discovered the synthetic element?",
        level=(index % 3) + 1,
        final_answer="unused",
    )


def _request() -> ModelRequest:
    return ModelRequest(
        instructions="Return a deterministic object.",
        input_items=(),
        output_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        allowed_tools=(
            ToolSpec(
                name="web_search",
                description="Offline search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ),
        reasoning_effort="max",
    )


async def _recover_one_continuation_error(observability: Observability) -> tuple[int, int]:
    client = _ResponsesClient(
        deque(
            [
                _Response([_FunctionCall()]),
                APIConnectionError(
                    request=httpx.Request("POST", "https://api.openai.com/v1/responses")
                ),
                _Response(),
            ]
        )
    )
    provider = OpenAIProvider(
        client,
        "gpt-5.6-luna",
        observability=observability,
        retry_policy=RetryPolicy(max_attempts=2),
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda: 0.0,
    )

    executor = _Executor()
    await provider.complete(_request(), executor)
    return client.calls, len(executor.calls)


@pytest.mark.asyncio
async def test_offline_ten_task_trace_reliability_simulation(tmp_path: Path) -> None:
    """Exercise observed recovery categories without a network, dataset, or paid model."""

    recorder = _RecordingLangfuse()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="public", secret_key="secret"),
        client=recorder,
    )
    private_sentinel = "https://private.example/offline-trace-sentinel"
    public_dir = tmp_path / "public-results"
    tasks = [_task(index) for index in range(10)]
    store = RunStore.create(
        public_dir,
        RunManifest(
            selected_task_ids=tuple(task.task_id for task in tasks),
            expected_levels={task.task_id: task.level for task in tasks},
            sample_size=len(tasks),
            seed=42,
            run_id="offline-trace-simulation",
            model="gpt-5.6-luna",
            provider_name="offline-fixture",
            evaluation_mode="test",
            reasoning_effort="max",
            dataset_revision="offline",
        ),
    )
    recovered_calls: list[tuple[int, int]] = []
    repaired_tasks: list[str] = []
    normalized_routes: list[str] = []

    for index, task in enumerate(tasks):
        trace = observability.start_task(
            task.task_id,
            level=task.level,
            model="gpt-5.6-luna",
            reasoning_effort="max",
            evaluation_mode="test",
        )
        notebook = ResearchNotebook(tmp_path / ".offline.private", task.task_id)
        notebook.record("private_search", {"task": task.task_id}, {"sentinel": private_sentinel})
        if index < 6:
            recovered_calls.append(await _recover_one_continuation_error(observability))
        elif index < 8:
            provider = _ScriptedProvider(
                [
                    ModelResponse(
                        parsed={"specialist": "computation", "evidence": [], "candidate_answer": "42"}
                    ),
                    ModelResponse(
                        parsed={"specialist": "lookup", "evidence": [], "candidate_answer": "42"}
                    ),
                ]
            )
            agent = LookupAgent(provider, reasoning_effort="max")
            result = await agent.run(
                task,
                PlanStep(specialist="lookup", instruction="Find observable evidence"),
                (),
                _NoToolClient(),
            )
            assert result.candidate_answer == "42"
            assert agent.last_repair_attempted is True
            repaired_tasks.append(task.task_id)
        else:
            plan = AgentPlan(
                restatement="Inspect a missing image",
                steps=(PlanStep(specialist="visual", instruction="Inspect pixels"),),
            )
            normalized = normalize_plan(plan, task, _ScriptedProvider([]))
            normalized_routes.extend(step.specialist.value for step in normalized.steps)
        critic = trace.span("critic", {"stage": "verify", "verdict": "approve"})
        critic.end(status="ok")
        trace.finish(
                TaskTraceOutcome(
                    answer_present=True,
                    evidence_count=1,
                    critic_verdict="approve",
                    error_category=None,
                    correct=True,
                    terminal_route="submit",
                    stage_codes=("verify", "submit"),
                )
        )
        store.append(
            TaskPrediction(
                task_id=task.task_id,
                level=task.level,
                answer="42",
                correct=True,
                critic_verdict="approve",
            )
        )

    write_reports(store, summarize(store.load_predictions()))
    public_artifacts = "".join(
        path.read_text(encoding="utf-8") for path in public_dir.rglob("*") if path.is_file()
    )
    roots = [
        observation
        for observation in recorder.observations
        if observation.kwargs.get("name") == "gaia-task"
    ]
    critic_stages = [
        observation
        for observation in recorder.observations
        if observation.kwargs.get("name") == "critic"
    ]

    assert recovered_calls == [(3, 1)] * 6
    assert len(repaired_tasks) == 2
    assert normalized_routes == ["lookup", "lookup"]
    assert len(roots) == 10
    assert all(root.closed for root in roots)
    assert {path.name for path in public_dir.iterdir()} >= {
        "manifest.json",
        "predictions.jsonl",
        "summary.json",
        "report.md",
    }
    terminal_outputs = [
        next(update["output"] for update in root.updates if "output" in update)
        for root in roots
    ]
    required_outcome_keys = {
        "answer_present",
        "evidence_count",
        "critic_verdict",
        "error_category",
        "correct",
        "provider_retry_count",
        "tool_success_count",
        "tool_failure_count",
        "provider_latency_ms",
        "tool_latency_ms",
        "checkpoint_restored",
        "attempt_number",
        "recovery_count",
        "recovery_succeeded",
        "retry_reason",
        "retry_category",
        "terminal_route",
        "reasoning_summary",
    }
    assert all(set(output) == required_outcome_keys for output in terminal_outputs)
    assert all(
        output["answer_present"] is True
        and output["evidence_count"] == 1
        and output["critic_verdict"] == "approve"
        and output["error_category"] is None
        and output["correct"] is True
        and output["attempt_number"] == 1
        and output["recovery_count"] == 0
        and output["terminal_route"] == "submit"
        and output["reasoning_summary"]
        == {"route": "submit", "recovery": "not_needed", "stages": ["verify", "submit"]}
        for output in terminal_outputs
    )
    assert len(critic_stages) >= 8
    assert all(stage.closed for stage in critic_stages)
    assert private_sentinel not in public_artifacts
    assert private_sentinel not in repr(
        [(observation.kwargs, observation.updates) for observation in recorder.observations]
    )
