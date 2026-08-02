"""Atomic, privacy-safe storage for resumable GAIA evaluation runs."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from gaia_agent.models import AgentPlan, AuditEvent, FrozenModel, TaskPrediction

_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_OPAQUE_EVIDENCE_ID_PATTERN = re.compile(r"evidence-[0-9]{4}\Z")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
    re.compile(r"\bhf_[A-Za-z0-9_-]+\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]+\b", flags=re.IGNORECASE),
)
_BLOCKED_KEY_PARTS = (
    "question",
    "final_answer",
    "reference_answer",
    "gold",
    "attachment",
    "prompt",
    "instruction",
    "restatement",
    "path",
    "content",
    "raw_evidence_id",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
)
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
    }
)


class RunManifest(FrozenModel):
    """Public, non-gated identity of one reproducible evaluation selection."""

    schema_version: int = 1
    selected_task_ids: tuple[str, ...]
    sample_size: int
    seed: int
    model: str
    reasoning_effort: str
    dataset_revision: str
    dataset_configuration: str = "2023_all"
    limits: dict[str, int | float] = Field(default_factory=dict)
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
        for task_id in self.selected_task_ids:
            _validate_task_id(task_id)
        return self


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
        records = self._valid_prediction_records()
        if prediction.task_id in records:
            raise ValueError(f"duplicate prediction for {prediction.task_id}")
        records[prediction.task_id] = _public_prediction_record(prediction)
        self._write_prediction_records(records)

    def load_predictions(self) -> list[TaskPrediction]:
        return [
            _prediction_from_record(record)
            for record in self._ordered_prediction_records().values()
        ]

    def completed_task_ids(self) -> frozenset[str]:
        return frozenset(self._valid_prediction_records())

    def write_plan(self, task_id: str, plan: AgentPlan | Mapping[str, Any]) -> Path:
        """Persist a plan's safe shape, never its generated textual instructions."""

        if isinstance(plan, AgentPlan):
            specialist_values = [step.specialist.value for step in plan.steps]
        else:
            raw_steps = plan.get("steps", ())
            specialist_values = [
                str(step.get("specialist", "unknown"))
                for step in raw_steps
                if isinstance(step, Mapping)
            ]
        payload = {"step_count": len(specialist_values), "specialists": specialist_values}
        path = self.task_directory(task_id) / "plan.json"
        _atomic_write(path, _json_text(payload))
        return path

    def append_note(self, task_id: str, note: Mapping[str, Any]) -> Path:
        return self._append_task_jsonl(task_id, "notes.jsonl", _redact_value(note))

    def append_audit(self, task_id: str, event: AuditEvent) -> Path:
        payload = {
            "sequence": event.sequence,
            "event_type": event.event_type,
            "timestamp": event.timestamp.astimezone(UTC).isoformat(),
            "payload": _redact_value(event.payload),
        }
        return self._append_task_jsonl(task_id, "audit.jsonl", payload)

    def write_public_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        if filename not in {"summary.json"}:
            raise ValueError("unsupported public artifact filename")
        path = self.root / filename
        _assert_contained(path, self.root)
        _atomic_write(path, _json_text(_redact_value(payload)))
        return path

    def write_public_text(self, filename: str, text: str) -> Path:
        if filename not in {"report.md"}:
            raise ValueError("unsupported public artifact filename")
        path = self.root / filename
        _assert_contained(path, self.root)
        _atomic_write(path, text)
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
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                _assert_private_json(path)

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
        raw_records = _read_jsonl(self.predictions_path)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw_record in raw_records:
            record = _validated_prediction_record(raw_record, self.manifest.selected_task_ids)
            if record is not None:
                grouped.setdefault(record["task_id"], []).append(record)
        return {
            task_id: records[0]
            for task_id, records in grouped.items()
            if len(records) == 1
        }

    def _write_prediction_records(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        ordered = [records[task_id] for task_id in self.manifest.selected_task_ids if task_id in records]
        lines = "".join(_json_text(record) for record in ordered)
        _atomic_write(self.predictions_path, lines)


def _validate_task_id(task_id: str) -> None:
    if not _TASK_ID_PATTERN.fullmatch(task_id) or task_id in {".", ".."} or ".." in task_id:
        raise ValueError("task ID must be a safe filename component")


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


def _public_prediction_record(prediction: TaskPrediction) -> dict[str, Any]:
    evidence_ids = [
        evidence_id
        for evidence_id in prediction.evidence_ids
        if _OPAQUE_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
    ]
    return {
        "task_id": prediction.task_id,
        "level": prediction.level,
        "answer": _redact_value(prediction.answer),
        "correct": prediction.correct,
        "duration_seconds": prediction.duration_seconds,
        "usage": prediction.usage.model_dump(mode="json"),
        "critic_verdict": _redact_value(prediction.critic_verdict),
        "evidence_ids": evidence_ids,
        "error_category": (
            prediction.error_category.value if prediction.error_category is not None else None
        ),
        "run_index": prediction.run_index,
        "vote_count": prediction.vote_count,
    }


def _prediction_from_record(record: Mapping[str, Any]) -> TaskPrediction:
    return TaskPrediction.model_validate(record)


def _validated_prediction_record(
    record: Any,
    selected_task_ids: Sequence[str],
) -> dict[str, Any] | None:
    if not isinstance(record, dict) or set(record) != _PREDICTION_KEYS:
        return None
    if record.get("task_id") not in selected_task_ids:
        return None
    if record.get("level") not in {1, 2, 3} or not isinstance(record.get("correct"), bool):
        return None
    if not isinstance(record.get("answer"), str | type(None)):
        return None
    if not isinstance(record.get("critic_verdict"), str) or not record["critic_verdict"]:
        return None
    if not isinstance(record.get("duration_seconds"), int | float) or record["duration_seconds"] < 0:
        return None
    if not isinstance(record.get("run_index"), int) or record["run_index"] < 0:
        return None
    if not isinstance(record.get("vote_count"), int) or record["vote_count"] < 1:
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
        TaskPrediction.model_validate(record)
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


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            string_key = str(key)
            if _is_blocked_key(string_key):
                continue
            if string_key == "evidence_ids":
                safe[string_key] = [
                    evidence_id
                    for evidence_id in item
                    if isinstance(evidence_id, str)
                    and _OPAQUE_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
                ] if isinstance(item, list | tuple) else []
            else:
                safe[string_key] = _redact_value(item)
        return safe
    if isinstance(value, list | tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_VALUE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        for known_secret in _known_secret_values():
            redacted = redacted.replace(known_secret, "[REDACTED]")
        return redacted
    return value


def _known_secret_values() -> tuple[str, ...]:
    return tuple(
        value
        for value in (os.environ.get("OPENAI_API_KEY"), os.environ.get("HF_TOKEN"))
        if value
    )


def _is_blocked_key(key: str) -> bool:
    lowered = key.casefold()
    return (
        lowered == "evidence_id"
        or lowered == "token"
        or lowered.endswith("_token")
        or any(part in lowered for part in _BLOCKED_KEY_PARTS)
    )


def _assert_private_json(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("run artifacts must not contain symlinks")
    if path.suffix == ".jsonl":
        values = _read_jsonl(path)
    else:
        try:
            values = [json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("public JSON artifact is malformed") from error
    for value in values:
        if _contains_private_value(value):
            raise ValueError("public artifact contains gated or secret material")


def _contains_private_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _is_blocked_key(str(key))
            or (
                str(key) == "evidence_ids"
                and (
                    not isinstance(item, list)
                    or any(
                        not isinstance(evidence_id, str)
                        or not _OPAQUE_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
                        for evidence_id in item
                    )
                )
            )
            or _contains_private_value(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_private_value(item) for item in value)
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS) or any(
            secret in value for secret in _known_secret_values()
        )
    return False
