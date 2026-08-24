"""Atomic, privacy-safe storage for resumable GAIA evaluation runs."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from gaia_agent.config import MAX_STAGE_CONTEXT_BYTES
from gaia_agent.models import (
    AgentPlan,
    AuditEvent,
    FrozenModel,
    SpecialistName,
    TaskPrediction,
    TerminalErrorCategory,
)
from gaia_agent.reliability import TRANSIENT_PROVIDER_ERROR_NAMES

_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_OPAQUE_EVIDENCE_ID_PATTERN = re.compile(r"evidence-[0-9]{4}\Z")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", flags=re.IGNORECASE),
    re.compile(r"\bhf_[A-Za-z0-9_-]{8,}\b", flags=re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{8,}\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|secret|token|password|credential)\s*(?:=|:)\s*[^\s]+",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bauthorization\s*:\s*bearer\s+[^\s]+", flags=re.IGNORECASE),
)
_PUBLIC_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
_PUBLIC_PACKAGE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_PUBLIC_RETRY_REASONS = frozenset(
    {"provider_transient_exhausted", "timeout", "task_transient"}
)
_PUBLIC_RETRY_CATEGORIES = TRANSIENT_PROVIDER_ERROR_NAMES | frozenset(
    {"TimeoutError", "ConnectionError", "provider_transient", "task_transient"}
)
_PUBLIC_TOOL_NAMES = frozenset(
    {
        "web_search",
        "read_webpage",
        "execute_python",
        "read_attachment",
        "transcribe_audio",
        "submit_answer",
    }
)
_PUBLIC_TOOL_ARGUMENT_KEYS = {
    "web_search": frozenset({"query"}),
    "read_webpage": frozenset({"url"}),
    "execute_python": frozenset({"instruction", "code", "paths", "attachment_ids", "file_ids"}),
    "read_attachment": frozenset({"path"}),
    "transcribe_audio": frozenset({"path"}),
    "submit_answer": frozenset({"answer", "evidence_ids"}),
}
_PUBLIC_TOOL_NORMAL_RESULT_SHAPES = {
    "web_search": frozenset({frozenset({"answer", "sources"})}),
    "read_webpage": frozenset(
        {
            frozenset({"url", "status_code", "content_type", "text", "truncated"}),
            frozenset({"url", "answer", "sources", "fallback"}),
        }
    ),
    "execute_python": frozenset({frozenset({"output_text", "outputs"})}),
    "read_attachment": frozenset(
        {
            frozenset({"path", "kind", "text"}),
            frozenset({"path", "kind", "rows"}),
            frozenset({"path", "kind", "data"}),
            frozenset({"path", "kind", "page_count", "text"}),
            frozenset({"path", "kind", "sheets"}),
            frozenset({"path", "kind", "media_type", "size_bytes"}),
        }
    ),
    "transcribe_audio": frozenset({frozenset({"text"})}),
    "submit_answer": frozenset({frozenset({"answer", "evidence_ids"})}),
}
_BOUNDED_RESULT_KEYS = frozenset({"output", "truncated"})
_PUBLIC_AUDIT_EVENT_TYPES = frozenset(
    {
        "tool_call_started",
        "tool_call_completed",
        "tool_call_failed",
        "tool_call_reused",
        "tool_call_restored",
        "terminal_failure",
        "plan_started",
        "plan_completed",
        "act_started",
        "act_completed",
        "correction_act_started",
        "correction_act_completed",
        "verify_started",
        "verify_completed",
        "quality_rejected",
        "correction_started",
        "correction_completed",
        "submit_started",
        "submit_completed",
        "context_budget_pressure",
        "plan_early_stop",
        "soft_step_failure",
        "stage_budget_exhausted",
        "task_recovery",
    }
)
_PUBLIC_AUDIT_EVENT_KEYS = {
    "tool_call_started": frozenset({"tool", "call_number", "argument_keys"}),
    "tool_call_completed": frozenset({"tool", "call_number", "argument_keys", "result_keys"}),
    "tool_call_failed": frozenset({"tool", "call_number", "argument_keys", "category"}),
    "tool_call_reused": frozenset(
        {"tool", "call_number", "argument_keys", "result_keys"}
    ),
    "tool_call_restored": frozenset(
        {"tool", "call_number", "argument_keys", "result_keys"}
    ),
    "terminal_failure": frozenset({"category", "critic_verdict"}),
    "plan_started": frozenset({"task_id"}),
    "plan_completed": frozenset({"task_id", "step_count", "specialists"}),
    "act_started": frozenset({"specialist", "step_number"}),
    "act_completed": frozenset({"specialist", "step_number", "evidence_ids", "has_candidate"}),
    "correction_act_started": frozenset({"specialist", "step_number"}),
    "correction_act_completed": frozenset(
        {"specialist", "step_number", "evidence_ids", "has_candidate"}
    ),
    "verify_started": frozenset({"evidence_ids", "has_candidate"}),
    "verify_completed": frozenset({"verdict", "checked_evidence_ids"}),
    "quality_rejected": frozenset({"category"}),
    "correction_started": frozenset({"verdict"}),
    "correction_completed": frozenset({"evidence_ids", "has_candidate"}),
    "submit_started": frozenset({"evidence_ids"}),
    "submit_completed": frozenset({"evidence_ids"}),
    "context_budget_pressure": frozenset({"bytes", "item_count", "max_bytes"}),
    "plan_early_stop": frozenset({"completed_steps", "remaining_steps"}),
    "soft_step_failure": frozenset({"step_number", "code"}),
    "stage_budget_exhausted": frozenset(
        {"stage", "reason", "timeout_seconds", "max_calls", "logical_calls"}
    ),
    "task_recovery": frozenset(
        {"attempt_number", "retry_reason", "retry_category", "recovery_succeeded"}
    ),
}
_PUBLIC_CRITIC_VERDICTS = frozenset({"approve", "revise", "insufficient_evidence", "not_run"})
_NORMAL_CRITIC_VERDICTS = _PUBLIC_CRITIC_VERDICTS - {"not_run"}
_PREDICTION_KEYS = frozenset(
    {
        "task_id",
        "level",
        "answer",
        "correct",
        "duration_seconds",
        "usage",
        "critic_verdict",
        "evidence_ids",
        "error_category",
        "run_index",
        "vote_count",
        "attempt_number",
        "recovery_count",
        "recovery_succeeded",
        "retry_reason",
        "retry_category",
        "provider_attempt_count",
    }
)


@dataclass(frozen=True)
class _CheckpointState:
    completed: dict[str, dict[str, Any]]
    repairable: dict[str, dict[str, Any]]


class PublicNote(FrozenModel):
    """A deliberately small, non-textual per-task note record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Literal[
        "checkpoint_repaired",
        "checkpoint_skipped",
        "checkpoint_corrupt",
        "checkpoint_incomplete",
    ]
    count: int = Field(ge=0, le=1_000_000)


class RunManifest(FrozenModel):
    """Public, non-gated identity of one reproducible evaluation selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    selected_task_ids: tuple[str, ...]
    expected_levels: dict[str, Literal[1, 2, 3]]
    sample_size: int
    seed: int
    run_id: str = "gaia-run"
    model: str
    provider_name: str = "openai"
    evaluation_mode: Literal["production", "test"] = "production"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    quality_policy: Literal["benchmark", "strict"] = "benchmark"
    dataset_name: str = "gaia-benchmark"
    dataset_revision: str
    dataset_configuration: str = "2023_all"
    dataset_split: str = "validation"
    limits: dict[str, int | float | Decimal] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_selection(self) -> RunManifest:
        if self.schema_version != 1:
            raise ValueError("unsupported manifest schema version")
        if self.sample_size != len(self.selected_task_ids):
            raise ValueError("sample_size must match selected_task_ids")
        if len(set(self.selected_task_ids)) != self.sample_size:
            raise ValueError("selected_task_ids must be unique")
        if set(self.expected_levels) != set(self.selected_task_ids):
            raise ValueError("expected_levels must exactly match selected_task_ids")
        for task_id in self.selected_task_ids:
            _validate_task_id(task_id)
        for value_name, value in (
            ("run_id", self.run_id),
            ("model", self.model),
            ("provider_name", self.provider_name),
            ("dataset_name", self.dataset_name),
            ("dataset_revision", self.dataset_revision),
            ("dataset_configuration", self.dataset_configuration),
            ("dataset_split", self.dataset_split),
        ):
            _validate_public_identifier(value_name, value)
        if set(self.limits) - {
            "max_steps",
            "max_tool_calls",
            "timeout_seconds",
            "planner_timeout_seconds",
            "specialist_timeout_seconds",
            "critic_timeout_seconds",
            "specialist_max_tool_calls",
            "critic_max_tool_calls",
            "votes",
            "provider_max_attempts",
            "provider_backoff_base",
            "private_notebook_max_bytes",
            "stage_context_max_bytes",
            "search_circuit_cooldown",
            "langfuse_input_cost_per_million",
            "langfuse_output_cost_per_million",
        }:
            raise ValueError("limits contains an unsupported public key")
        if any(
            not isinstance(value, (int, float, Decimal)) or isinstance(value, bool)
            for value in self.limits.values()
        ):
            raise ValueError("limits values must be numeric")
        if any(
            not _PUBLIC_PACKAGE_NAME_PATTERN.fullmatch(name)
            or _contains_secret_material(name)
            or not _PUBLIC_IDENTIFIER_PATTERN.fullmatch(version)
            or _contains_secret_material(version)
            for name, version in self.package_versions.items()
        ):
            raise ValueError("package_versions must contain bounded public identifiers")
        return self

    def expected_level(self, task_id: str) -> Literal[1, 2, 3]:
        try:
            return self.expected_levels[task_id]
        except KeyError as error:
            raise ValueError("task ID is not selected by this manifest") from error


class RunStore:
    """A run directory whose manifest and JSONL checkpoints survive interruption."""

    def __init__(self, root: Path, manifest: RunManifest) -> None:
        self.root = root.absolute()
        self.manifest = manifest

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.jsonl"

    @classmethod
    def create(cls, root: Path, manifest: RunManifest) -> RunStore:
        root = root.absolute()
        _ensure_directory(root)
        store = cls(root, manifest)
        if store.manifest_path.exists() or store.manifest_path.is_symlink():
            return cls.open(root, manifest)
        if store.predictions_path.exists() or store.predictions_path.is_symlink():
            raise ValueError("orphaned predictions checkpoint has no manifest")
        _atomic_write(store.manifest_path, _json_text(manifest.model_dump(mode="json")))
        _atomic_write(store.predictions_path, "")
        return store

    @classmethod
    def open(cls, root: Path, manifest: RunManifest) -> RunStore:
        root = root.absolute()
        _ensure_directory(root)
        store = cls(root, manifest)
        if not store.manifest_path.is_file() or store.manifest_path.is_symlink():
            raise ValueError("run manifest is missing or unsafe")
        try:
            persisted = RunManifest.model_validate_json(store.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("run manifest is invalid") from error
        if persisted != manifest:
            raise ValueError("run manifest does not match the requested resume")
        if store.predictions_path.exists() and store.predictions_path.is_symlink():
            raise ValueError("predictions checkpoint is unsafe")
        if not store.predictions_path.exists():
            _atomic_write(store.predictions_path, "")
        return store

    def task_directory(self, task_id: str) -> Path:
        _validate_task_id(task_id)
        if task_id not in self.manifest.selected_task_ids:
            raise ValueError("task ID is not selected by this manifest")
        directory = self.root / "tasks" / task_id
        _ensure_directory(directory, root=self.root)
        return directory

    def append(self, prediction: TaskPrediction) -> None:
        if prediction.task_id not in self.manifest.selected_task_ids:
            raise ValueError("prediction task ID is not selected by this manifest")
        if prediction.level != self.manifest.expected_level(prediction.task_id):
            raise ValueError("prediction level does not match the run manifest")
        checkpoint = self._checkpoint_state()
        if prediction.task_id in checkpoint.completed:
            raise ValueError(f"duplicate prediction for {prediction.task_id}")
        records = checkpoint.repairable
        records[prediction.task_id] = _public_prediction_record(prediction)
        self._write_prediction_records(records)

    def load_predictions(self) -> list[TaskPrediction]:
        return [
            _prediction_from_record(record)
            for record in self._ordered_prediction_records().values()
        ]

    def completed_task_ids(self) -> frozenset[str]:
        return frozenset(self._valid_prediction_records())

    def write_plan(self, task_id: str, plan: AgentPlan) -> Path:
        """Persist a plan's safe shape, never its generated textual instructions."""

        if not isinstance(plan, AgentPlan):
            raise TypeError("plan must be an AgentPlan")
        payload = {
            "steps": [
                {"index": index, "specialist": step.specialist.value}
                for index, step in enumerate(plan.steps, start=1)
            ]
        }
        path = self.task_directory(task_id) / "plan.json"
        _atomic_write(path, _json_text(payload))
        return path

    def append_note(self, task_id: str, note: PublicNote) -> Path:
        if not isinstance(note, PublicNote):
            raise TypeError("note must be a PublicNote")
        return self._append_task_jsonl(task_id, "notes.jsonl", note.model_dump(mode="json"))

    def append_audit(self, task_id: str, event: AuditEvent) -> Path:
        if event.event_type not in _PUBLIC_AUDIT_EVENT_TYPES:
            raise ValueError("audit event_type is not public")
        payload = _public_audit_payload(event.event_type, event.payload)
        record = {
            "sequence": event.sequence,
            "event_type": event.event_type,
            "timestamp": event.timestamp.astimezone(UTC).isoformat(),
            "payload": payload,
        }
        return self._append_task_jsonl(task_id, "audit.jsonl", record)

    def write_summary(self, summary: Any) -> Path:
        from gaia_agent.reporting import EvaluationSummary

        if not isinstance(summary, EvaluationSummary):
            raise TypeError("summary must be an EvaluationSummary")
        path = self.root / "summary.json"
        _assert_contained(path, self.root)
        _atomic_write(path, _json_text(summary.model_dump(mode="json")))
        return path

    def write_report(self, summary: Any) -> Path:
        from gaia_agent.reporting import EvaluationSummary, _markdown_report

        if not isinstance(summary, EvaluationSummary):
            raise TypeError("summary must be an EvaluationSummary")
        path = self.root / "report.md"
        _assert_contained(path, self.root)
        _atomic_write(path, _markdown_report(self, summary))
        return path

    def verify_integrity(self) -> None:
        """Reject unsafe links, malformed checkpoints, and privacy-policy violations."""

        _ensure_directory(self.root)
        persisted = RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))
        if persisted != self.manifest:
            raise ValueError("run manifest does not match the active store")
        raw_lines = _read_jsonl(self.predictions_path)
        valid = self._valid_prediction_records()
        if len(raw_lines) != len(valid):
            raise ValueError("predictions checkpoint contains incomplete or corrupt records")
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise ValueError("run artifacts must not contain symlinks")
            if path.is_file():
                self._verify_artifact(path)

    def _verify_artifact(self, path: Path) -> None:
        relative = path.relative_to(self.root)
        if relative in {Path("manifest.json"), Path("predictions.jsonl")}:
            return
        if relative == Path("summary.json"):
            from gaia_agent.reporting import EvaluationSummary

            EvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))
            return
        if relative == Path("report.md"):
            from gaia_agent.reporting import EvaluationSummary, _markdown_report

            summary = EvaluationSummary.model_validate_json(
                (self.root / "summary.json").read_text(encoding="utf-8")
            )
            if path.read_text(encoding="utf-8") != _markdown_report(self, summary):
                raise ValueError("report does not match the typed public summary")
            return
        parts = relative.parts
        if len(parts) != 3 or parts[0] != "tasks":
            raise ValueError("run artifact is not allowlisted")
        _validate_task_id(parts[1])
        if parts[1] not in self.manifest.selected_task_ids:
            raise ValueError("task artifact is not selected by this manifest")
        if parts[2] == "plan.json":
            _validate_plan_record(json.loads(path.read_text(encoding="utf-8")))
        elif parts[2] == "notes.jsonl":
            for value in _read_jsonl(path):
                PublicNote.model_validate(value)
        elif parts[2] == "audit.jsonl":
            for value in _read_jsonl(path):
                _validate_audit_record(value)
        else:
            raise ValueError("task artifact is not allowlisted")

    def _append_task_jsonl(self, task_id: str, filename: str, payload: Any) -> Path:
        directory = self.task_directory(task_id)
        path = directory / filename
        _assert_contained(path, self.root)
        previous = "" if not path.exists() else path.read_text(encoding="utf-8")
        _atomic_write(path, previous + _json_text(payload))
        return path

    def _ordered_prediction_records(self) -> dict[str, dict[str, Any]]:
        records = self._valid_prediction_records()
        return {
            task_id: records[task_id]
            for task_id in self.manifest.selected_task_ids
            if task_id in records
        }

    def _valid_prediction_records(self) -> dict[str, dict[str, Any]]:
        return self._checkpoint_state().completed

    def _checkpoint_state(self) -> _CheckpointState:
        grouped: dict[str, list[dict[str, Any]]] = {}
        invalid_task_ids: set[str] = set()
        unattributed_corruption = False
        for raw_record in _read_jsonl(self.predictions_path):
            record = _validated_prediction_record(raw_record, self.manifest.expected_levels)
            if record is not None:
                grouped.setdefault(record["task_id"], []).append(record)
                continue
            if isinstance(raw_record, Mapping) and raw_record.get("task_id") in self.manifest.selected_task_ids:
                invalid_task_ids.add(raw_record["task_id"])
            else:
                unattributed_corruption = True
        repairable = {
            task_id: records[0]
            for task_id, records in grouped.items()
            if len(records) == 1 and task_id not in invalid_task_ids
        }
        completed = {} if unattributed_corruption else repairable
        return _CheckpointState(
            completed=completed,
            repairable=repairable,
        )

    def _write_prediction_records(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        ordered = [records[task_id] for task_id in self.manifest.selected_task_ids if task_id in records]
        lines = "".join(_json_text(record) for record in ordered)
        _atomic_write(self.predictions_path, lines)


def _validate_task_id(task_id: str) -> None:
    if not _TASK_ID_PATTERN.fullmatch(task_id) or task_id in {".", ".."} or ".." in task_id:
        raise ValueError("task ID must be a safe filename component")
    if _contains_secret_material(task_id):
        raise ValueError("task ID must not contain secret material")


def _validate_public_identifier(value_name: str, value: str) -> None:
    if not _PUBLIC_IDENTIFIER_PATTERN.fullmatch(value) or _contains_secret_material(value):
        raise ValueError(f"{value_name} must be a bounded public identifier")


def _contains_secret_material(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS) or any(
        secret in value for secret in _known_secret_values()
    )


def _ensure_directory(directory: Path, *, root: Path | None = None) -> None:
    _assert_no_symlink_ancestors(directory)
    if directory.exists() and directory.is_symlink():
        raise ValueError("run artifacts must not contain symlinks")
    if root is not None:
        _assert_contained(directory, root)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("run artifact directory is unsafe")
    if root is not None:
        _assert_contained(directory, root)


def _assert_no_symlink_ancestors(path: Path) -> None:
    current = path.absolute()
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("run artifacts must not traverse symlinks")
        current = current.parent


def _assert_contained(path: Path, root: Path) -> None:
    root = root.absolute()
    path = path.absolute()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("run artifact path escapes its root") from error
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("run artifacts must not contain symlinks")
        current = current.parent
    if root.is_symlink():
        raise ValueError("run artifacts must not contain symlinks")


def _atomic_write(path: Path, text: str) -> None:
    """Replace one artifact only after its complete contents reach the filesystem."""

    if path.is_symlink():
        raise ValueError("refusing to replace a symlinked artifact")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() and temporary.is_symlink():
        raise ValueError("refusing to use a symlinked temporary artifact")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _public_audit_payload(event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != _PUBLIC_AUDIT_EVENT_KEYS[event_type]:
        raise ValueError(f"{event_type} payload keys do not match its public schema")
    public: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "task_id":
            if not isinstance(value, str):
                raise ValueError("audit task_id must be a public identifier")
            _validate_task_id(value)
        elif key == "tool":
            if value not in _PUBLIC_TOOL_NAMES:
                raise ValueError("audit tool is not allowlisted")
        elif key in {"bytes", "max_bytes"}:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= MAX_STAGE_CONTEXT_BYTES
            ):
                raise ValueError("audit context counter must be a bounded integer")
        elif key in {
            "call_number",
            "step_count",
            "step_number",
            "item_count",
            "completed_steps",
            "remaining_steps",
            "attempt_number",
            "max_calls",
            "logical_calls",
        }:
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 1_000_000:
                raise ValueError("audit counter must be a bounded integer")
        elif key == "code":
            if value not in {
                "role_mismatch",
                "evidence_shape",
                "source_url_missing",
                "candidate_format",
                "checked_evidence",
                "schema",
            }:
                raise ValueError("audit code is not allowlisted")
        elif key == "stage":
            if value not in {"planner", "specialist", "critic"}:
                raise ValueError("audit stage is not allowlisted")
        elif key == "reason":
            if value not in {"time", "tool_calls"}:
                raise ValueError("audit stage budget reason is not allowlisted")
        elif key == "timeout_seconds":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1_000_000
            ):
                raise ValueError("audit timeout must be a bounded number")
        elif key == "specialist":
            if value not in {item.value for item in SpecialistName}:
                raise ValueError("audit specialist is not allowlisted")
        elif key == "specialists":
            if not isinstance(value, list) or any(item not in {role.value for role in SpecialistName} for item in value):
                raise ValueError("audit specialists are not allowlisted")
            if len(value) > 32:
                raise ValueError("audit specialists exceed the public limit")
        elif key in {"evidence_ids", "checked_evidence_ids"}:
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not _OPAQUE_EVIDENCE_ID_PATTERN.fullmatch(item)
                for item in value
            ):
                raise ValueError("audit evidence IDs must be trusted opaque IDs")
            if len(value) > 9_999:
                raise ValueError("audit evidence IDs exceed the public limit")
        elif key in {"has_candidate", "recovery_succeeded"} and not isinstance(value, bool):
            raise ValueError("audit boolean field must be boolean")
        elif key == "retry_reason" and value not in _PUBLIC_RETRY_REASONS:
            raise ValueError("audit retry reason is not allowlisted")
        elif key == "retry_category" and value not in _PUBLIC_RETRY_CATEGORIES:
            raise ValueError("audit retry category is not allowlisted")
        public[key] = value
    if event_type in {
        "tool_call_started",
        "tool_call_completed",
        "tool_call_failed",
        "tool_call_reused",
        "tool_call_restored",
    }:
        tool = public.get("tool")
        if tool not in _PUBLIC_TOOL_NAMES:
            raise ValueError("tool audit must include an allowlisted tool")
        argument_keys = public.get("argument_keys")
        if not isinstance(argument_keys, list) or not set(argument_keys) <= _PUBLIC_TOOL_ARGUMENT_KEYS[tool]:
            raise ValueError("tool audit argument keys do not match the tool contract")
        if len(argument_keys) > 32:
            raise ValueError("tool audit argument keys exceed the public limit")
        result_keys = public.get("result_keys")
        if event_type == "tool_call_started" and result_keys is not None:
            raise ValueError("tool start audit cannot include result keys")
        if event_type == "tool_call_failed" and result_keys is not None:
            raise ValueError("tool failure audit cannot include result keys")
        if event_type in {"tool_call_completed", "tool_call_reused", "tool_call_restored"}:
            result_key_set = frozenset(result_keys) if isinstance(result_keys, list) else None
            if result_key_set not in _PUBLIC_TOOL_NORMAL_RESULT_SHAPES[tool] | {_BOUNDED_RESULT_KEYS}:
                raise ValueError("tool audit result keys do not match the tool contract")
            if not isinstance(result_keys, list) or len(result_keys) != len(result_key_set) or len(result_keys) > 32:
                raise ValueError("tool audit result keys exceed the public limit")
    elif "argument_keys" in public or "result_keys" in public or "tool" in public:
        raise ValueError("non-tool audit cannot include tool metadata")
    _validate_audit_value_context(event_type, public)
    return public


def _validate_audit_value_context(event_type: str, payload: Mapping[str, Any]) -> None:
    terminal_categories = {item.value for item in TerminalErrorCategory}
    if event_type == "terminal_failure":
        if payload.get("category") not in terminal_categories:
            raise ValueError("terminal_failure category must be a terminal error category")
        if payload.get("critic_verdict") not in _PUBLIC_CRITIC_VERDICTS:
            raise ValueError("terminal_failure critic_verdict is invalid")
        return
    if event_type == "tool_call_failed":
        if payload.get("category") != "tool_error":
            raise ValueError("tool_call_failed category must be tool_error")
        return
    if "critic_verdict" in payload:
        raise ValueError("critic_verdict is only public on terminal_failure")
    if event_type == "verify_completed":
        if payload.get("verdict") not in _NORMAL_CRITIC_VERDICTS:
            raise ValueError("verify_completed verdict must be a completed critic verdict")
    elif event_type == "correction_started":
        if payload.get("verdict") not in {"revise", "insufficient_evidence"}:
            raise ValueError("correction_started verdict must require correction")
    elif "verdict" in payload:
        raise ValueError("verdict is only public on verification or correction audit events")
    if event_type == "quality_rejected":
        if payload.get("category") != "insufficient_evidence":
            raise ValueError("quality_rejected category must be insufficient_evidence")
    elif "category" in payload:
        raise ValueError("category is only public on terminal_failure or quality_rejected")


def _validate_plan_record(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"steps"} or not isinstance(value["steps"], list):
        raise ValueError("plan artifact is not a public plan shape")
    for index, step in enumerate(value["steps"], start=1):
        if not isinstance(step, dict) or set(step) != {"index", "specialist"}:
            raise ValueError("plan artifact contains an unsupported field")
        if step["index"] != index or step["specialist"] not in {role.value for role in SpecialistName}:
            raise ValueError("plan artifact contains an invalid public value")


def _validate_audit_record(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"sequence", "event_type", "timestamp", "payload"}:
        raise ValueError("audit artifact is not a public audit shape")
    if not isinstance(value["sequence"], int) or value["sequence"] < 1:
        raise ValueError("audit sequence must be a positive integer")
    if value["event_type"] not in _PUBLIC_AUDIT_EVENT_TYPES or not isinstance(value["timestamp"], str):
        raise ValueError("audit event is not public")
    try:
        datetime.fromisoformat(value["timestamp"])
    except ValueError as error:
        raise ValueError("audit timestamp is invalid") from error
    if not isinstance(value["payload"], Mapping):
        raise TypeError("audit payload must be a mapping")
    _public_audit_payload(value["event_type"], value["payload"])


def _public_prediction_record(prediction: TaskPrediction) -> dict[str, Any]:
    validate_public_prediction(prediction)
    evidence_ids = [
        evidence_id
        for evidence_id in prediction.evidence_ids
        if _OPAQUE_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
    ]
    return {
        "task_id": prediction.task_id,
        "level": prediction.level,
        "answer": _redact_answer(prediction.answer),
        "correct": prediction.correct,
        "duration_seconds": prediction.duration_seconds,
        "usage": prediction.usage.model_dump(mode="json"),
        "critic_verdict": prediction.critic_verdict,
        "evidence_ids": evidence_ids,
        "error_category": (
            prediction.error_category.value if prediction.error_category is not None else None
        ),
        "run_index": prediction.run_index,
        "vote_count": prediction.vote_count,
        "attempt_number": prediction.attempt_number,
        "recovery_count": prediction.recovery_count,
        "recovery_succeeded": prediction.recovery_succeeded,
        "retry_reason": prediction.retry_reason,
        "retry_category": prediction.retry_category,
        "provider_attempt_count": prediction.provider_attempt_count,
    }


def _prediction_from_record(record: Mapping[str, Any]) -> TaskPrediction:
    prediction = TaskPrediction.model_validate(record)
    validate_public_prediction(prediction)
    return prediction


def validate_public_prediction(prediction: TaskPrediction) -> None:
    if prediction.critic_verdict not in _PUBLIC_CRITIC_VERDICTS:
        raise ValueError("prediction critic_verdict is not public")
    if prediction.critic_verdict == "not_run" and (
        prediction.answer is not None
        or prediction.correct
        or not isinstance(prediction.error_category, TerminalErrorCategory)
    ):
        raise ValueError("not_run prediction must be an incorrect terminal failure")
    if prediction.retry_reason is not None and prediction.retry_reason not in _PUBLIC_RETRY_REASONS:
        raise ValueError("prediction retry reason is not public")
    if (
        prediction.retry_category is not None
        and prediction.retry_category not in _PUBLIC_RETRY_CATEGORIES
    ):
        raise ValueError("prediction retry category is not public")
    if prediction.recovery_succeeded and prediction.recovery_count < 1:
        raise ValueError("successful recovery requires a recorded recovery")


def _validated_prediction_record(
    record: Any,
    expected_levels: Mapping[str, Literal[1, 2, 3]],
) -> dict[str, Any] | None:
    if not isinstance(record, dict) or set(record) != _PREDICTION_KEYS:
        return None
    task_id = record.get("task_id")
    if task_id not in expected_levels:
        return None
    if record.get("level") != expected_levels[task_id] or not isinstance(record.get("correct"), bool):
        return None
    if not isinstance(record.get("answer"), str | type(None)):
        return None
    if record.get("critic_verdict") not in _PUBLIC_CRITIC_VERDICTS:
        return None
    if not isinstance(record.get("duration_seconds"), int | float) or record["duration_seconds"] < 0:
        return None
    if not isinstance(record.get("run_index"), int) or record["run_index"] < 0:
        return None
    if not isinstance(record.get("vote_count"), int) or record["vote_count"] < 1:
        return None
    if not isinstance(record.get("attempt_number"), int) or isinstance(
        record["attempt_number"], bool
    ) or record["attempt_number"] < 1:
        return None
    if not isinstance(record.get("recovery_count"), int) or isinstance(
        record["recovery_count"], bool
    ) or record["recovery_count"] < 0:
        return None
    if not isinstance(record.get("recovery_succeeded"), bool):
        return None
    if record.get("retry_reason") not in _PUBLIC_RETRY_REASONS | {None}:
        return None
    if record.get("retry_category") not in _PUBLIC_RETRY_CATEGORIES | {None}:
        return None
    if not isinstance(record.get("provider_attempt_count"), int) or isinstance(
        record["provider_attempt_count"], bool
    ) or record["provider_attempt_count"] < 0:
        return None
    usage = record.get("usage")
    if not isinstance(usage, dict) or set(usage) != {
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    }:
        return None
    if any(not isinstance(value, int) or value < 0 for value in usage.values()):
        return None
    evidence_ids = record.get("evidence_ids")
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not _OPAQUE_EVIDENCE_ID_PATTERN.fullmatch(item)
        for item in evidence_ids
    ):
        return None
    if record.get("error_category") is not None and not isinstance(record["error_category"], str):
        return None
    try:
        validate_public_prediction(TaskPrediction.model_validate(record))
    except ValueError:
        return None
    return dict(record)


def _read_jsonl(path: Path) -> list[Any]:
    if not path.exists() or path.is_symlink():
        return []
    records: list[Any] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(None)
    except OSError:
        return []
    return records


def _redact_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    redacted = answer
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    for known_secret in _known_secret_values():
        redacted = redacted.replace(known_secret, "[REDACTED]")
    return redacted


def _known_secret_values() -> tuple[str, ...]:
    return tuple(
        value
        for value in (os.environ.get("OPENAI_API_KEY"), os.environ.get("HF_TOKEN"))
        if value
    )
