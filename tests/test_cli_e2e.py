from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gaia_agent.cli import app
from gaia_agent.config import RunConfig
from gaia_agent.models import GaiaTask, TaskPrediction, TerminalErrorCategory, Usage
from gaia_agent.runner import EvaluationRunner

cli_runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "gaia_records.json"


def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-credential")
    monkeypatch.setenv("HF_TOKEN", "hf_fake_credential")


def _evaluate_arguments(output_dir: Path, *extra: str) -> list[str]:
    return [
        "evaluate",
        "--dataset-fixture",
        str(FIXTURE),
        "--sample-size",
        "3",
        "--seed",
        "42",
        "--output-dir",
        str(output_dir),
        "--fake-provider",
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


def test_evaluate_cli_runs_four_planes_over_real_stdio_and_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    output_dir = tmp_path / "run"

    result = cli_runner.invoke(app, _evaluate_arguments(output_dir))

    assert result.exit_code == 0, result.output
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "predictions.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    audit = "".join(path.read_text(encoding="utf-8") for path in output_dir.rglob("audit.jsonl"))
    assert '"tool":"web_search"' in audit
    assert '"tool":"submit_answer"' in audit


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "HF_TOKEN"])
def test_evaluate_preflight_names_missing_credentials_without_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _credentials(monkeypatch)
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

    result = cli_runner.invoke(app, _evaluate_arguments(tmp_path / "run", *arguments))

    assert result.exit_code == 2
    assert "configuration error" in result.output.lower()
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_seed_selection_and_task_level_manifest_are_literal_and_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
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
        ConnectionError("temporary"),
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


def test_resume_skips_complete_records_and_plain_append_repairs_incomplete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
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


def test_outputs_stay_contained_and_exclude_secret_and_gated_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
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

    manifest_path.write_text(original_manifest, encoding="utf-8")
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    incomplete = cli_runner.invoke(app, ["verify-results", str(output_dir)])
    assert incomplete.exit_code == 2
