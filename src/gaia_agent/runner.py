"""Evaluation coordination, dependency wiring, and independent result verification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI

from gaia_agent.agents import (
    ComputationAgent,
    CriticAgent,
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
from gaia_agent.orchestrator import Orchestrator, choose_majority
from gaia_agent.persistence import RunManifest, RunStore
from gaia_agent.providers import ModelRequest, ModelResponse, OpenAIProvider
from gaia_agent.reporting import (
    EvaluationSummary,
    LevelMetrics,
    summarize,
    verify_summary,
    write_reports,
)
from gaia_agent.scoring import question_scorer

_PUBLIC_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
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


class TaskOrchestrator(Protocol):
    audit_events: Sequence[AuditEvent]

    async def run(self, task: GaiaTask) -> TaskPrediction: ...


DatasetLoader = Callable[[RunConfig], tuple[list[GaiaTask], str]]
OrchestratorFactory = Callable[[GaiaTask], TaskOrchestrator]
Sleep = Callable[[float], Awaitable[None]]


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
    ) -> None:
        self._config = config
        self._dataset_loader = dataset_loader
        self._orchestrator_factory = orchestrator_factory
        self._sleep = sleep

    async def run(self) -> EvaluationSummary:
        validate_config(self._config)
        tasks, revision = self._dataset_loader(self._config)
        selected = sample_stratified(tasks, self._config.sample_size, self._config.seed)
        store = _open_or_create_store(self._config, selected, revision)

        for task in selected:
            if task.task_id in store.completed_task_ids():
                continue
            started = time.monotonic()
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
            vote = await self._run_vote(task)
            predictions.append(vote.prediction.model_copy(update={"run_index": run_index}))
            audit_events.extend(vote.audit_events)
            first_plan = first_plan or vote.plan
            selected = choose_majority(predictions, self._config.votes)
            if selected.error_category != TerminalErrorCategory.NO_CONSENSUS:
                break

        selected = choose_majority(predictions, self._config.votes)
        usage = _sum_usage(predictions)
        selected = selected.model_copy(update={"usage": usage})
        return _TaskRun(
            prediction=selected,
            audit_events=tuple(audit_events),
            plan=first_plan,
        )

    async def _run_vote(self, task: GaiaTask) -> _TaskRun:
        audit_events: list[AuditEvent] = []
        first_plan: AgentPlan | None = None
        prior_predictions: list[TaskPrediction] = []
        for attempt in range(2):
            orchestrator = self._orchestrator_factory(task)
            try:
                prediction = await orchestrator.run(task)
                _validate_prediction_identity(prediction, task)
            except Exception as error:  # noqa: BLE001 - task isolation boundary
                events = tuple(getattr(orchestrator, "audit_events", ()))
                audit_events.extend(events)
                first_plan = first_plan or _plan_from_audit(events)
                if attempt == 0 and _is_transient_failure(error):
                    await self._sleep(0.05)
                    continue
                category = (
                    TerminalErrorCategory.RETRY_EXHAUSTED
                    if _is_transient_failure(error)
                    else _category_for_exception(error)
                )
                prediction = _failure_prediction(task, category)
                if prior_predictions:
                    prediction = prediction.model_copy(
                        update={"usage": _sum_usage([*prior_predictions, prediction])}
                    )
            else:
                events = tuple(getattr(orchestrator, "audit_events", ()))
                audit_events.extend(events)
                first_plan = first_plan or _plan_from_audit(events)
                if _is_transient_prediction(prediction):
                    prior_predictions.append(prediction)
                    if attempt == 0:
                        await self._sleep(0.05)
                        continue
                    prediction = prediction.model_copy(
                        update={"error_category": TerminalErrorCategory.RETRY_EXHAUSTED}
                    )
                    charged_predictions = prior_predictions
                else:
                    charged_predictions = [*prior_predictions, prediction]
                prediction = prediction.model_copy(
                    update={"usage": _sum_usage(charged_predictions)}
                )
            return _TaskRun(prediction, tuple(audit_events), first_plan)
        raise AssertionError("bounded retry loop did not return")


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


def load_fixture_dataset(path: Path) -> tuple[list[GaiaTask], str]:
    """Load synthetic GAIA-shaped records for the explicitly offline CLI mode."""

    fixture = path.resolve(strict=True)
    raw = fixture.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list):
        raise TypeError("dataset fixture must contain a JSON list")
    tasks: list[GaiaTask] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("dataset fixture records must be objects")
        attachment_path = _fixture_attachment(fixture.parent, record.get("file_path"))
        tasks.append(
            GaiaTask(
                task_id=record["task_id"],
                question=record["Question"],
                level=int(record["Level"]),
                final_answer=record["Final answer"],
                attachment_path=attachment_path,
            )
        )
    return tasks, "fixture-" + hashlib.sha256(raw).hexdigest()[:16]


def build_runner(
    config: RunConfig,
    *,
    dataset_fixture: Path | None = None,
    fake_provider: bool = False,
) -> EvaluationRunner:
    """Compose provider-neutral agents and orchestrators with the real stdio MCP client."""

    validate_config(config)
    if dataset_fixture is None:
        dataset_loader: DatasetLoader = lambda active: load_gaia_validation(
            active.hf_token.get_secret_value()
        )
    else:
        fixture_path = dataset_fixture
        dataset_loader = lambda _active: load_fixture_dataset(fixture_path)

    provider: Any
    if fake_provider:
        provider = _OfflineFixtureProvider()
    else:
        provider = OpenAIProvider(
            AsyncOpenAI(
                api_key=config.openai_api_key.get_secret_value(),
                max_retries=0,
            ),
            model=config.model,
        )
    planner = PlannerAgent(
        provider,
        max_steps=config.max_steps,
        reasoning_effort=config.reasoning_effort,
    )
    specialists = {
        SpecialistName.LOOKUP: LookupAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=config.max_tool_calls,
        ),
        SpecialistName.COMPUTATION: ComputationAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=config.max_tool_calls,
        ),
        SpecialistName.FILE: FileAnalysisAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=config.max_tool_calls,
        ),
        SpecialistName.VISUAL: VisualAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_tool_calls=config.max_tool_calls,
        ),
    }
    critic = CriticAgent(
        provider,
        reasoning_effort=config.reasoning_effort,
        max_tool_calls=config.max_tool_calls,
    )

    def orchestrator_factory(task: GaiaTask) -> Orchestrator:
        return Orchestrator(
            planner=planner,
            specialists=specialists,
            critic=critic,
            tool_client_factory=lambda active_task: _stdio_client(
                active_task,
                config,
                fake_backend=fake_provider,
            ),
            config=config,
        )

    return EvaluationRunner(
        config=config,
        dataset_loader=dataset_loader,
        orchestrator_factory=orchestrator_factory,
    )


def verify_run_directory(root: Path) -> EvaluationSummary:
    """Reload and independently verify every public run artifact."""

    root = root.absolute()
    required = tuple(root / name for name in ("manifest.json", "predictions.jsonl", "summary.json", "report.md"))
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ValueError("run is incomplete: required result artifacts are missing")

    raw_manifest = _load_json_object(root / "manifest.json")
    _require_exact_keys(raw_manifest, RunManifest.model_fields, "manifest")
    _reject_forbidden_keys(raw_manifest)
    manifest = RunManifest.model_validate(raw_manifest)
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
) -> RunStore:
    manifest_path = config.output_dir.absolute() / "manifest.json"
    if manifest_path.exists():
        if not config.resume:
            raise ValueError("output directory already contains a run; pass --resume")
        persisted = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        manifest = _manifest_for(config, selected, revision, started_at=persisted.started_at)
        return RunStore.open(config.output_dir, manifest)
    manifest = _manifest_for(config, selected, revision)
    return RunStore.create(config.output_dir, manifest)


def _manifest_for(
    config: RunConfig,
    selected: Sequence[GaiaTask],
    revision: str,
    *,
    started_at: datetime | None = None,
) -> RunManifest:
    return RunManifest(
        selected_task_ids=tuple(task.task_id for task in selected),
        expected_levels={task.task_id: task.level for task in selected},
        sample_size=config.sample_size,
        seed=config.seed,
        run_id=f"gaia-seed{config.seed}-n{config.sample_size}",
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        dataset_revision=revision,
        limits={
            "max_steps": config.max_steps,
            "max_tool_calls": config.max_tool_calls,
            "timeout_seconds": config.timeout_seconds,
            "votes": config.votes,
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


def _fixture_attachment(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TypeError("fixture file_path must be a string")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("fixture attachment path escapes the fixture directory")
    if not resolved.is_file():
        raise ValueError("fixture attachment does not exist")
    return resolved


@asynccontextmanager
async def _stdio_client(
    task: GaiaTask,
    config: RunConfig,
    *,
    fake_backend: bool,
) -> Any:
    task_root = task.attachment_path.parent if task.attachment_path is not None else Path.cwd()
    attachments = (task.attachment_path,) if task.attachment_path is not None else ()
    previous = os.environ.get("GAIA_MCP_FAKE_BACKEND")
    if fake_backend:
        os.environ["GAIA_MCP_FAKE_BACKEND"] = "1"
    client = MCPToolClient(task_root=task_root, config=config, attachments=attachments)
    try:
        entered = await client.__aenter__()
    finally:
        if fake_backend:
            if previous is None:
                os.environ.pop("GAIA_MCP_FAKE_BACKEND", None)
            else:
                os.environ["GAIA_MCP_FAKE_BACKEND"] = previous
    try:
        yield entered
    finally:
        await client.__aexit__(None, None, None)


class _OfflineFixtureProvider:
    """Deterministic hosted-provider double; MCP transport and server remain real."""

    async def complete(
        self,
        request: ModelRequest,
        executor: Any | None = None,
    ) -> ModelResponse:
        context = _request_context(request)
        if request.instructions == PlannerAgent.instructions:
            return ModelResponse(
                parsed={
                    "restatement": "Gather one observable source and verify the concise answer",
                    "steps": [{"specialist": "lookup", "instruction": "Use the shared tool plane"}],
                }
            )
        if request.instructions == LookupAgent.instructions:
            if executor is None:
                raise RuntimeError("offline lookup requires the MCP executor")
            task_id = str(context.get("task_id", "offline-task"))
            result = await executor.call("web_search", {"query": task_id})
            sources = result.get("sources")
            source_url = (
                sources[0].get("url")
                if isinstance(sources, list) and sources and isinstance(sources[0], Mapping)
                else "https://example.test/source"
            )
            return ModelResponse(
                parsed={
                    "specialist": "lookup",
                    "evidence": [
                        {
                            "evidence_id": f"offline-{task_id}",
                            "kind": "source",
                            "summary": "Deterministic hosted-backend fixture result",
                            "source_url": source_url,
                        }
                    ],
                    "candidate_answer": _offline_answer(str(context.get("question", ""))),
                }
            )
        if request.instructions == CriticAgent.instructions:
            evidence = context.get("evidence")
            evidence_ids = [
                item["evidence_id"]
                for item in evidence
                if isinstance(evidence, list)
                and isinstance(item, Mapping)
                and isinstance(item.get("evidence_id"), str)
            ]
            return ModelResponse(
                parsed={
                    "verdict": "approve",
                    "feedback": "Observable source evidence is present",
                    "checked_evidence_ids": evidence_ids,
                    "corrected_answer": None,
                }
            )
        raise ValueError("offline provider received an unsupported agent request")


def _request_context(request: ModelRequest) -> dict[str, Any]:
    try:
        content = request.input_items[0]["content"]
        text = next(item["text"] for item in content if item.get("type") == "input_text")
        value = json.loads(text)
    except (IndexError, KeyError, StopIteration, TypeError, json.JSONDecodeError) as error:
        raise ValueError("offline provider received malformed input") from error
    if not isinstance(value, dict):
        raise TypeError("offline provider input must be an object")
    return value


def _offline_answer(question: str) -> str:
    prefix = "Synthetic question "
    if question.startswith(prefix):
        return "synthetic gold " + question.removeprefix(prefix)
    return "offline answer"


def _validate_prediction_identity(prediction: TaskPrediction, task: GaiaTask) -> None:
    if prediction.task_id != task.task_id or prediction.level != task.level:
        raise ValueError("orchestrator prediction identity does not match the selected task")


def _is_transient_failure(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return any(base.__name__ in _TRANSIENT_EXCEPTION_NAMES for base in type(error).__mro__)


def _is_transient_prediction(prediction: TaskPrediction) -> bool:
    return prediction.error_category == TerminalErrorCategory.TIMEOUT


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


def _sum_usage(predictions: Sequence[TaskPrediction]) -> Usage:
    return Usage(
        input_tokens=sum(item.usage.input_tokens for item in predictions),
        output_tokens=sum(item.usage.output_tokens for item in predictions),
        reasoning_tokens=sum(item.usage.reasoning_tokens for item in predictions),
        total_tokens=sum(item.usage.total_tokens for item in predictions),
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
    for sequence, event in enumerate(task_run.audit_events, start=first_sequence):
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
