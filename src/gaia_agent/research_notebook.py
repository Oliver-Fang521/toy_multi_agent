"""Private, durable checkpoints for successful tool research."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NotebookEntry:
    signature: str
    tool: str
    result: dict[str, Any]
    created_at: datetime


class ResearchNotebook:
    """Store task-scoped tool results outside the public evaluation run."""

    def __init__(self, root: Path, task_id: str, max_entry_bytes: int = 65_536) -> None:
        if max_entry_bytes < 1:
            raise ValueError("max_entry_bytes must be positive")
        if not _is_contained_task_id(task_id):
            raise ValueError("task_id must be a single path component")
        self.root = Path(root)
        self.task_id = task_id
        self.max_entry_bytes = max_entry_bytes
        self._create_private_directory(self.root)
        self._task_root = self.root / task_id
        self._create_private_directory(self._task_root)

    def lookup(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
        signature = _signature(tool, arguments)
        try:
            raw_entry = json.loads(self._entry_path(signature).read_text(encoding="utf-8"))
            entry = _parse_entry(raw_entry)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if entry.signature != signature or entry.tool != tool:
            return None
        return copy.deepcopy(entry.result)

    def record(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        signature = _signature(tool, arguments)
        entry = NotebookEntry(
            signature=signature,
            tool=tool,
            result=copy.deepcopy(dict(result)),
            created_at=datetime.now(UTC),
        )
        payload = _encode_entry(entry)
        if len(payload) > self.max_entry_bytes:
            raise ValueError("notebook entry exceeds max_entry_bytes")
        self._atomic_write(self._entry_path(signature), payload)

    @staticmethod
    def _create_private_directory(path: Path) -> None:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)

    def _entry_path(self, signature: str) -> Path:
        return self._task_root / f"{signature}.json"

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".entry-", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def _is_contained_task_id(task_id: str) -> bool:
    path = Path(task_id)
    return bool(task_id) and path.name == task_id and "/" not in task_id and "\\" not in task_id


def _signature(tool: str, arguments: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"tool": tool, "arguments": dict(arguments)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _encode_entry(entry: NotebookEntry) -> bytes:
    return json.dumps(
        {
            "signature": entry.signature,
            "tool": entry.tool,
            "result": entry.result,
            "created_at": entry.created_at.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _parse_entry(value: object) -> NotebookEntry:
    if not isinstance(value, dict):
        raise TypeError("notebook entry must be an object")
    signature = value.get("signature")
    tool = value.get("tool")
    result = value.get("result")
    created_at = value.get("created_at")
    if not isinstance(signature, str) or not isinstance(tool, str) or not isinstance(result, dict):
        raise TypeError("notebook entry has invalid fields")
    if not isinstance(created_at, str):
        raise TypeError("notebook entry has invalid timestamp")
    return NotebookEntry(
        signature=signature,
        tool=tool,
        result=result,
        created_at=datetime.fromisoformat(created_at),
    )
