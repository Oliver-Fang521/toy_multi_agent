from __future__ import annotations

import json
from datetime import UTC, datetime

from gaia_agent.models import TaskPrediction, Usage
from gaia_agent.persistence import RunManifest, RunStore
from gaia_agent.reporting import summarize, write_reports


def prediction(
    task_id: str,
    level: int,
    *,
    correct: bool,
    duration: float,
    verdict: str = "approve",
    error: str | None = None,
    usage: Usage | None = None,
) -> TaskPrediction:
    return TaskPrediction(
        task_id=task_id,
        level=level,
        answer="not persisted",
        correct=correct,
        duration_seconds=duration,
        usage=usage or Usage(),
        critic_verdict=verdict,
        error_category=error,
    )


def manifest() -> RunManifest:
    return RunManifest(
        selected_task_ids=("a", "b", "c"),
        sample_size=3,
        seed=42,
        model="test-model",
        reasoning_effort="medium",
        dataset_revision="synthetic-revision",
        started_at=datetime(2026, 8, 2, tzinfo=UTC),
    )


def test_summary_includes_empty_levels_usage_percentiles_and_verifications() -> None:
    summary = summarize(
        [
            prediction(
                "a",
                1,
                correct=True,
                duration=1.0,
                usage=Usage(input_tokens=2, output_tokens=3, reasoning_tokens=5, total_tokens=10),
            ),
            prediction(
                "b",
                1,
                correct=False,
                duration=5.0,
                verdict="revise",
                error="timeout",
                usage=Usage(input_tokens=7, output_tokens=11, reasoning_tokens=13, total_tokens=31),
            ),
            prediction("c", 2, correct=True, duration=100.0, verdict="approve"),
        ]
    )

    assert (summary.total, summary.correct, summary.accuracy) == (3, 2, 2 / 3)
    assert summary.by_level[1].accuracy == 0.5
    assert summary.by_level[2].accuracy == 1.0
    assert summary.by_level[3].accuracy is None
    assert (summary.p50_seconds, summary.p95_seconds) == (5.0, 100.0)
    assert summary.usage == Usage(input_tokens=9, output_tokens=14, reasoning_tokens=18, total_tokens=41)
    assert (
        summary.input_tokens,
        summary.output_tokens,
        summary.reasoning_tokens,
        summary.total_tokens,
    ) == (9, 14, 18, 41)
    assert summary.critic_verdicts == {"approve": 2, "revise": 1}
    assert summary.failure_categories == {"timeout": 1}


def test_summary_recomputes_score_from_predictions() -> None:
    summary = summarize(
        [
            prediction("a", 1, correct=True, duration=1),
            prediction("b", 2, correct=False, duration=1),
        ]
    )

    assert summary.correct == 1
    assert summary.accuracy == 0.5


def test_reports_are_deterministic_private_and_explain_limitations(tmp_path) -> None:
    store = RunStore.create(tmp_path, manifest())
    predictions = [
        prediction("a", 1, correct=True, duration=1.0),
        prediction("b", 2, correct=False, duration=2.0, error="timeout"),
        prediction("c", 3, correct=True, duration=3.0),
    ]
    for item in predictions:
        store.append(item)
    summary_path, report_path = write_reports(store, summarize(predictions))

    payload = json.loads(summary_path.read_text())
    report = report_path.read_text()
    assert payload["total"] == 3
    assert "sample size: 3" in report
    assert "seed: 42" in report
    assert "test-model" in report
    assert "medium" in report
    assert "not comparable to the full GAIA leaderboard" in report
    assert "not persisted" in store.predictions_path.read_text()
    assert "not persisted" not in report
    assert not list(tmp_path.rglob("*.tmp"))


def test_report_writes_are_idempotent(tmp_path) -> None:
    store = RunStore.create(tmp_path, manifest())
    summary = summarize([prediction("a", 1, correct=True, duration=1.0)])
    first = write_reports(store, summary)
    first_contents = tuple(path.read_text() for path in first)
    second = write_reports(store, summary)

    assert first == second
    assert tuple(path.read_text() for path in second) == first_contents
