"""Private, durable checkpoints for successful tool research."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_MAX_CARD_TITLE_CHARS = 256
_MAX_CARD_DOMAIN_CHARS = 255
_MAX_CARD_EXCERPT_CHARS = 1_536


@dataclass(frozen=True)
class NotebookEntry:
    signature: str
    tool: str
    result: dict[str, Any]
    created_at: datetime


def select_evidence_cards(
    entries: Sequence[NotebookEntry],
    *,
    max_items: int = 12,
    max_json_bytes: int = 24_576,
) -> tuple[dict[str, Any], ...]:
    """Select a small, deterministic model-safe view of private tool results.

    Cards intentionally retain only source metadata and a short excerpt.  The
    notebook's raw results may contain full pages or tool output and must never
    be copied directly into a model request.
    """
    if max_items < 0:
        raise ValueError("max_items must be non-negative")
    if max_json_bytes < 0:
        raise ValueError("max_json_bytes must be non-negative")

    candidates = [_evidence_card(entry, index) for index, entry in enumerate(entries)]
    unique: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for score, index, card, duplicate_key in sorted(
        candidates,
        key=lambda item: (-item[0], item[1]),
    ):
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        unique.append((score, index, card))

    selected: list[dict[str, Any]] = []
    for _, _, card in unique:
        if len(selected) >= max_items:
            break
        candidate = [*selected, card]
        if _json_size_bytes(candidate) > max_json_bytes:
            continue
        selected.append(card)
    return tuple(selected)


def _evidence_card(entry: NotebookEntry, index: int) -> tuple[int, int, dict[str, Any], tuple[str, str]]:
    url, title, excerpt = _card_fields(entry.result)
    domain = urlparse(url).hostname if url else None
    card = {
        "entry_hash": hashlib.sha256(entry.signature.encode("utf-8")).hexdigest(),
        "tool_kind": _bound_text(entry.tool, _MAX_CARD_TITLE_CHARS),
        "title": _bound_text(title, _MAX_CARD_TITLE_CHARS),
        "domain": _bound_text(domain, _MAX_CARD_DOMAIN_CHARS),
        "excerpt": _bound_text(excerpt, _MAX_CARD_EXCERPT_CHARS),
    }
    evidence_id = entry.result.get("evidence_id")
    if isinstance(evidence_id, str) and evidence_id.strip():
        card["evidence_id"] = _bound_text(evidence_id, _MAX_CARD_TITLE_CHARS)
    card = {key: value for key, value in card.items() if value}
    content_hash = hashlib.sha256(
        _canonical_card_content(title, excerpt).encode("utf-8")
    ).hexdigest()
    duplicate_key = ("url", url) if url else ("content", content_hash)
    has_source = bool(url or title)
    has_excerpt = bool(excerpt)
    has_error = _is_error_result(entry.result)
    score = (1_000 if has_source else 0) + (100 if has_excerpt else 0) - (500 if has_error else 0)
    return score, index, card, duplicate_key


def _card_fields(result: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    source: Mapping[str, Any] = result
    sources = result.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, Mapping):
                source = {**result, **item}
                break
    url = _first_text(source, "url", "source_url")
    title = _first_text(source, "title", "name")
    excerpt = _first_text(
        source,
        "snippet",
        "answer",
        "summary",
        "text",
        "content",
        "logs",
        "output",
        "result",
        "value",
        "data",
    )
    return url, title, excerpt


def _first_text(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _bound_text(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.split())
    if not normalized:
        return None
    return normalized[:limit]


def _canonical_card_content(title: str | None, excerpt: str | None) -> str:
    return "\n".join(
        value.casefold().strip() for value in (title, excerpt) if value and value.strip()
    )


def _is_error_result(result: Mapping[str, Any]) -> bool:
    return bool(result.get("error")) and not any(
        _first_text(result, key)
        for key in ("url", "source_url", "title", "snippet", "answer", "text", "content")
    )


def _json_size_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


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
        entry_path = self._entry_path(signature)
        try:
            self._assert_private_directory(self.root)
            self._assert_private_directory(self._task_root)
            payload = _read_bounded_entry(entry_path, self.max_entry_bytes)
            if payload is None:
                return None
            raw_entry = json.loads(payload)
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
        self._assert_private_directory(self.root)
        self._assert_private_directory(self._task_root)
        self._atomic_write(self._entry_path(signature), payload)

    @staticmethod
    def _create_private_directory(path: Path) -> None:
        _reject_symlink_components(path)
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        ResearchNotebook._assert_private_directory(path)
        path.chmod(0o700)

    @staticmethod
    def _assert_private_directory(path: Path) -> None:
        _reject_symlink_components(path)
        if not path.is_dir():
            raise ValueError("notebook storage path must be a directory")

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


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("notebook storage must not traverse symlinks")


def _read_bounded_entry(path: Path, max_entry_bytes: int) -> bytes | None:
    try:
        initial_state = os.lstat(path)
        if not stat.S_ISREG(initial_state.st_mode) or initial_state.st_size > max_entry_bytes:
            return None
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        opened_state = os.fstat(descriptor)
        if not stat.S_ISREG(opened_state.st_mode) or opened_state.st_size > max_entry_bytes:
            return None
        payload = os.read(descriptor, max_entry_bytes + 1)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    return payload if len(payload) <= max_entry_bytes else None


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
