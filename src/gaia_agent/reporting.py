"""Deterministic aggregate metrics and privacy-safe human-readable reports."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from gaia_agent.models import FrozenModel, TaskPrediction, TerminalErrorCategory, Usage
from gaia_agent.persistence import RunStore


class LevelMetrics(FrozenModel):
    total: int
    correct: int
    accuracy: float | None
    total_seconds: float
    p50_seconds: float | None
    p95_seconds: float | None


class EvaluationSummary(FrozenModel):
    total: int
    correct: int
    accuracy: float | None
    by_level: dict[Literal[1, 2, 3], LevelMetrics]
    usage: Usage
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    total_seconds: float
    p50_seconds: float | None
    p95_seconds: float | None
    critic_verdicts: dict[str, int]
    failure_categories: dict[str, int]

    @model_validator(mode="after")
    def validate_public_report_values(self) -> EvaluationSummary:
        if set(self.by_level) != {1, 2, 3}:
            raise ValueError("by_level must contain all GAIA levels")
        if set(self.critic_verdicts) - {"approve", "revise", "insufficient_evidence", "not_run"}:
            raise ValueError("critic_verdicts contains a non-public verdict")
        if set(self.failure_categories) - {item.value for item in TerminalErrorCategory}:
            raise ValueError("failure_categories contains a non-public category")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for values in (self.critic_verdicts, self.failure_categories)
            for value in values.values()
        ):
            raise ValueError("report counters must be non-negative integers")
        if (
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.total_tokens,
        ) != (
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.usage.reasoning_tokens,
            self.usage.total_tokens,
        ):
            raise ValueError("report token totals must match usage")
        return self


def summarize(predictions: Sequence[TaskPrediction]) -> EvaluationSummary:
    ordered = sorted(predictions, key=lambda prediction: (prediction.level, prediction.task_id))
    levels = {
        level: _level_metrics([prediction for prediction in ordered if prediction.level == level])
        for level in (1, 2, 3)
    }
    total = len(ordered)
    correct = sum(prediction.correct for prediction in ordered)
    durations = [prediction.duration_seconds for prediction in ordered]
    usage = _sum_usage(ordered)
    return EvaluationSummary(
        total=total,
        correct=correct,
        accuracy=correct / total if total else None,
        by_level=levels,
        usage=usage,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        total_tokens=usage.total_tokens,
        total_seconds=sum(durations),
        p50_seconds=_percentile(durations, 0.5),
        p95_seconds=_percentile(durations, 0.95),
        critic_verdicts=dict(sorted(Counter(prediction.critic_verdict for prediction in ordered).items())),
        failure_categories=dict(
            sorted(
                Counter(
                    prediction.error_category.value
                    if prediction.error_category is not None
                    else None
                    for prediction in ordered
                    if prediction.error_category is not None
                ).items()
            )
        ),
    )


def write_reports(store: RunStore, summary: EvaluationSummary) -> tuple[Path, Path]:
    """Write stable JSON and Markdown views of aggregate-only evaluation results."""

    summary_path = store.write_summary(summary)
    report_path = store.write_report(summary)
    return summary_path, report_path


def verify_summary(predictions: Sequence[TaskPrediction], summary: EvaluationSummary) -> None:
    """Raise when persisted summary metrics do not recompute from public predictions."""

    if summarize(predictions) != summary:
        raise ValueError("summary does not match recomputed prediction metrics")


def _level_metrics(predictions: Sequence[TaskPrediction]) -> LevelMetrics:
    total = len(predictions)
    correct = sum(prediction.correct for prediction in predictions)
    durations = [prediction.duration_seconds for prediction in predictions]
    return LevelMetrics(
        total=total,
        correct=correct,
        accuracy=correct / total if total else None,
        total_seconds=sum(durations),
        p50_seconds=_percentile(durations, 0.5),
        p95_seconds=_percentile(durations, 0.95),
    )


def _sum_usage(predictions: Sequence[TaskPrediction]) -> Usage:
    return Usage(
        input_tokens=sum(prediction.usage.input_tokens for prediction in predictions),
        output_tokens=sum(prediction.usage.output_tokens for prediction in predictions),
        reasoning_tokens=sum(prediction.usage.reasoning_tokens for prediction in predictions),
        total_tokens=sum(prediction.usage.total_tokens for prediction in predictions),
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _markdown_report(store: RunStore, summary: EvaluationSummary) -> str:
    manifest = store.manifest
    accuracy = _format_accuracy(summary.accuracy)
    lines = [
        "# GAIA Evaluation Report",
        "",
        "## Run configuration",
        "",
        f"- sample size: {manifest.sample_size}",
        f"- seed: {manifest.seed}",
        f"- model: {manifest.model}",
        f"- reasoning level: {manifest.reasoning_effort}",
        f"- dataset revision: {manifest.dataset_revision}",
        "",
        "This sampled evaluation is not comparable to the full GAIA leaderboard.",
        "",
        "## Correctness",
        "",
        f"- overall: {accuracy} ({summary.correct}/{summary.total})",
        "- per level:",
    ]
    for level in (1, 2, 3):
        metrics = summary.by_level[level]
        lines.append(
            f"  - level {level}: {_format_accuracy(metrics.accuracy)} "
            f"({metrics.correct}/{metrics.total})"
        )
    lines.extend(
        [
            "",
            "## Usage and latency",
            "",
            f"- input tokens: {summary.usage.input_tokens}",
            f"- output tokens: {summary.usage.output_tokens}",
            f"- reasoning tokens: {summary.usage.reasoning_tokens}",
            f"- total tokens: {summary.usage.total_tokens}",
            f"- total duration seconds: {summary.total_seconds:.3f}",
            f"- p50 duration seconds: {_format_seconds(summary.p50_seconds)}",
            f"- p95 duration seconds: {_format_seconds(summary.p95_seconds)}",
            "",
            "## Verification outcomes",
            "",
        ]
    )
    lines.extend(_counter_lines(summary.critic_verdicts))
    lines.extend(["", "## Failure categories", ""])
    lines.extend(_counter_lines(summary.failure_categories))
    return "\n".join(lines) + "\n"


def _counter_lines(values: dict[str, int]) -> list[str]:
    return [f"- {key}: {value}" for key, value in values.items()] or ["- none: 0"]


def _format_accuracy(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _format_seconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
