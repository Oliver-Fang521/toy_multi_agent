"""Private, durable checkpoints for successful tool research."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
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
_ENTRY_SCHEMA_VERSION = 1
_ENTRY_CONTRACT = "gaia.research-notebook.entry"
_TRUST_ANCHOR_SCHEMA_VERSION = 1
_TRUST_ANCHOR_CONTRACT = "gaia.research-notebook.trust-anchor"
_TRUST_ANCHOR_FILENAME = ".trust-anchor"
_MAX_TRUST_ANCHOR_BYTES = 4_096
_HMAC_KEY_BYTES = 32
_ENTRY_BODY_KEYS = frozenset(
    {
        "schema_version",
        "contract",
        "notebook_scope",
        "task_id",
        "signature",
        "tool",
        "result",
        "created_at",
    }
)
_ENTRY_KEYS = _ENTRY_BODY_KEYS | {"integrity_hmac_sha256"}


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
    title = _bound_text(title, _MAX_CARD_TITLE_CHARS)
    excerpt = _bound_text(excerpt, _MAX_CARD_EXCERPT_CHARS)
    domain = urlparse(url).hostname if url else None
    card = {
        "entry_hash": hashlib.sha256(entry.signature.encode("utf-8")).hexdigest(),
        "tool_kind": _bound_text(entry.tool, _MAX_CARD_TITLE_CHARS),
        "title": title,
        "domain": _bound_text(domain, _MAX_CARD_DOMAIN_CHARS),
        "excerpt": excerpt,
    }
    evidence_id = entry.result.get("evidence_id")
    if isinstance(evidence_id, str) and evidence_id.strip():
        card["evidence_id"] = evidence_id
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

    def __init__(
        self,
        root: Path,
        task_id: str,
        max_entry_bytes: int = 65_536,
        *,
        run_generation: str = "standalone",
    ) -> None:
        if max_entry_bytes < 1:
            raise ValueError("max_entry_bytes must be positive")
        if not _is_contained_task_id(task_id):
            raise ValueError("task_id must be a single path component")
        if not run_generation or len(run_generation.encode("utf-8")) > 512:
            raise ValueError("run_generation must be non-empty and bounded")
        self.root = Path(root)
        self.task_id = task_id
        self.max_entry_bytes = max_entry_bytes
        self._create_private_directory(self.root)
        self._integrity_key = self._load_or_rotate_trust_anchor(run_generation)
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
            entry = _parse_entry(
                raw_entry,
                integrity_key=self._integrity_key,
                expected_notebook_scope=_notebook_scope(self.root),
                expected_task_id=self.task_id,
            )
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
        payload = _encode_entry(
            entry,
            integrity_key=self._integrity_key,
            notebook_scope=_notebook_scope(self.root),
            task_id=self.task_id,
        )
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

    def _load_or_rotate_trust_anchor(self, run_generation: str) -> bytes:
        path = self.root / _TRUST_ANCHOR_FILENAME
        if path.is_symlink():
            raise ValueError("notebook trust anchor must not be a symlink")
        if path.exists():
            state = os.lstat(path)
            if stat.S_IMODE(state.st_mode) != 0o600:
                raise ValueError("notebook trust anchor permissions must be 0600")
            payload = _read_bounded_entry(
                path,
                _MAX_TRUST_ANCHOR_BYTES,
                required_mode=0o600,
            )
            if payload is None:
                raise ValueError("notebook trust anchor is invalid or oversized")
            persisted_generation, integrity_key = _parse_trust_anchor(payload)
            if persisted_generation == run_generation:
                return integrity_key
        integrity_key = secrets.token_bytes(_HMAC_KEY_BYTES)
        payload = _encode_trust_anchor(run_generation, integrity_key)
        if len(payload) > _MAX_TRUST_ANCHOR_BYTES:
            raise ValueError("notebook trust anchor is oversized")
        self._atomic_write(path, payload)
        return integrity_key

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


def _read_bounded_entry(
    path: Path,
    max_entry_bytes: int,
    *,
    required_mode: int | None = None,
) -> bytes | None:
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
        if required_mode is not None and stat.S_IMODE(opened_state.st_mode) != required_mode:
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


def _notebook_scope(root: Path) -> str:
    return hashlib.sha256(str(root.absolute()).encode("utf-8")).hexdigest()


def _entry_body(
    entry: NotebookEntry,
    *,
    notebook_scope: str,
    task_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": _ENTRY_SCHEMA_VERSION,
        "contract": _ENTRY_CONTRACT,
        "notebook_scope": notebook_scope,
        "task_id": task_id,
        "signature": entry.signature,
        "tool": entry.tool,
        "result": entry.result,
        "created_at": entry.created_at.isoformat(),
    }


def _integrity_digest(body: Mapping[str, Any], integrity_key: bytes) -> str:
    canonical = json.dumps(
        dict(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hmac.new(integrity_key, canonical, hashlib.sha256).hexdigest()


def _encode_entry(
    entry: NotebookEntry,
    *,
    integrity_key: bytes,
    notebook_scope: str,
    task_id: str,
) -> bytes:
    body = _entry_body(entry, notebook_scope=notebook_scope, task_id=task_id)
    return json.dumps(
        {
            **body,
            "integrity_hmac_sha256": _integrity_digest(body, integrity_key),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _parse_entry(
    value: object,
    *,
    integrity_key: bytes,
    expected_notebook_scope: str,
    expected_task_id: str,
) -> NotebookEntry:
    if not isinstance(value, dict) or set(value) != _ENTRY_KEYS:
        raise TypeError("notebook entry must be an object")
    schema_version = value.get("schema_version")
    contract = value.get("contract")
    notebook_scope = value.get("notebook_scope")
    task_id = value.get("task_id")
    signature = value.get("signature")
    tool = value.get("tool")
    result = value.get("result")
    created_at = value.get("created_at")
    integrity_hmac_sha256 = value.get("integrity_hmac_sha256")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _ENTRY_SCHEMA_VERSION
    ):
        raise ValueError("notebook entry schema is not supported")
    if contract != _ENTRY_CONTRACT:
        raise ValueError("notebook entry contract is not supported")
    if notebook_scope != expected_notebook_scope:
        raise ValueError("notebook entry run scope does not match")
    if task_id != expected_task_id:
        raise ValueError("notebook entry task scope does not match")
    if not isinstance(signature, str) or not isinstance(tool, str) or not isinstance(result, dict):
        raise TypeError("notebook entry has invalid fields")
    if not isinstance(created_at, str) or not isinstance(integrity_hmac_sha256, str):
        raise TypeError("notebook entry has invalid timestamp")
    body = {key: value[key] for key in _ENTRY_BODY_KEYS}
    if not hmac.compare_digest(
        integrity_hmac_sha256,
        _integrity_digest(body, integrity_key),
    ):
        raise ValueError("notebook entry failed integrity validation")
    return NotebookEntry(
        signature=signature,
        tool=tool,
        result=result,
        created_at=datetime.fromisoformat(created_at),
    )


def _encode_trust_anchor(run_generation: str, integrity_key: bytes) -> bytes:
    return json.dumps(
        {
            "schema_version": _TRUST_ANCHOR_SCHEMA_VERSION,
            "contract": _TRUST_ANCHOR_CONTRACT,
            "run_generation": run_generation,
            "hmac_key_hex": integrity_key.hex(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _parse_trust_anchor(payload: bytes) -> tuple[str, bytes]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("notebook trust anchor is malformed") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "contract",
        "run_generation",
        "hmac_key_hex",
    }:
        raise ValueError("notebook trust anchor contract is invalid")
    if value.get("schema_version") != _TRUST_ANCHOR_SCHEMA_VERSION:
        raise ValueError("notebook trust anchor schema is unsupported")
    if value.get("contract") != _TRUST_ANCHOR_CONTRACT:
        raise ValueError("notebook trust anchor contract is unsupported")
    run_generation = value.get("run_generation")
    key_hex = value.get("hmac_key_hex")
    if not isinstance(run_generation, str) or not isinstance(key_hex, str):
        raise TypeError("notebook trust anchor fields are invalid")
    try:
        integrity_key = bytes.fromhex(key_hex)
    except ValueError as error:
        raise ValueError("notebook trust anchor key is invalid") from error
    if len(integrity_key) != _HMAC_KEY_BYTES:
        raise ValueError("notebook trust anchor key is invalid")
    return run_generation, integrity_key
