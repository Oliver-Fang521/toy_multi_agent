from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaia_agent.research_notebook import (
    NotebookEntry,
    ResearchNotebook,
    select_evidence_cards,
)


def _rewrite_with_test_digest(path: Path, **updates: object) -> None:
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry.update(updates)
    integrity_field = (
        "integrity_hmac_sha256"
        if "integrity_hmac_sha256" in entry
        else "integrity_sha256"
    )
    unsigned = {
        key: value
        for key, value in entry.items()
        if key not in {"integrity_sha256", "integrity_hmac_sha256"}
    }
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    entry[integrity_field] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(entry), encoding="utf-8")


def test_notebook_persists_private_bounded_result(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1", max_entry_bytes=1024)

    notebook.record("web_search", {"query": "q"}, {"answer": "a", "sources": []})

    assert notebook.lookup("web_search", {"query": "q"}) == {
        "answer": "a",
        "sources": [],
    }
    assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
    assert stat.S_IMODE(next((tmp_path / "private").rglob("*.json")).stat().st_mode) == 0o600


def test_notebook_normalizes_arguments_and_returns_independent_copies(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    notebook.record(
        "web_search",
        {"query": "q", "options": {"language": "en", "limit": 1}},
        {"sources": [{"url": "https://private.example"}]},
    )

    restored = notebook.lookup(
        "web_search",
        {"options": {"limit": 1, "language": "en"}, "query": "q"},
    )
    assert restored == {"sources": [{"url": "https://private.example"}]}
    assert restored is not None
    restored["sources"][0]["url"] = "caller mutation"
    assert notebook.lookup(
        "web_search", {"query": "q", "options": {"limit": 1, "language": "en"}}
    ) == {"sources": [{"url": "https://private.example"}]}


def test_notebook_treats_corrupt_entry_as_a_cache_miss(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    notebook.record("web_search", {"query": "q"}, {"answer": "a"})
    entry = next((tmp_path / "private").rglob("*.json"))
    entry.write_text("not json", encoding="utf-8")

    assert notebook.lookup("web_search", {"query": "q"}) is None


def test_notebook_treats_parseable_result_tampering_as_a_cache_miss(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    arguments = {"query": "q"}
    notebook.record("web_search", arguments, {"answer": "trusted"})
    entry = next((tmp_path / "private").rglob("*.json"))
    raw = json.loads(entry.read_text(encoding="utf-8"))
    raw["result"] = {"answer": "fabricated"}
    entry.write_text(json.dumps(raw), encoding="utf-8")

    assert notebook.lookup("web_search", arguments) is None


def test_notebook_rejects_result_forged_with_a_recomputed_public_digest(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    arguments = {"query": "q"}
    notebook.record("web_search", arguments, {"answer": "trusted"})
    entry = next((tmp_path / "private").rglob("*.json"))
    _rewrite_with_test_digest(entry, result={"answer": "forged"})

    assert notebook.lookup("web_search", arguments) is None


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        pytest.param("schema_version", 999, id="unknown-schema"),
        pytest.param("schema_version", 1.0, id="invalid-schema-type"),
        pytest.param("contract", "gaia.research-notebook.stale", id="unknown-contract"),
        pytest.param("task_id", "task-2", id="wrong-task-scope"),
    ],
)
def test_notebook_rejects_rehashed_stale_contracts(
    tmp_path: Path,
    field: str,
    stale_value: object,
) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    arguments = {"query": "q"}
    notebook.record("web_search", arguments, {"answer": "trusted"})
    entry = next((tmp_path / "private").rglob("*.json"))
    _rewrite_with_test_digest(entry, **{field: stale_value})

    assert notebook.lookup("web_search", arguments) is None


def test_notebook_does_not_trust_legacy_unsigned_entries(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    arguments = {"query": "q"}
    notebook.record("web_search", arguments, {"answer": "trusted"})
    entry = next((tmp_path / "private").rglob("*.json"))
    raw = json.loads(entry.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "contract",
        "notebook_scope",
        "task_id",
        "integrity_sha256",
        "integrity_hmac_sha256",
    ):
        raw.pop(key, None)
    entry.write_text(json.dumps(raw), encoding="utf-8")

    assert notebook.lookup("web_search", arguments) is None


def test_notebook_does_not_replay_entry_copied_from_another_run_root(tmp_path: Path) -> None:
    arguments = {"query": "q"}
    first = ResearchNotebook(tmp_path / "run-one.private", "task-1")
    first.record("web_search", arguments, {"answer": "first run"})
    source = next(first.root.rglob("*.json"))
    second = ResearchNotebook(tmp_path / "run-two.private", "task-1")
    destination = second.root / "task-1" / source.name
    destination.write_bytes(source.read_bytes())

    assert second.lookup("web_search", arguments) is None


def test_notebook_rejects_cross_run_scope_forged_with_public_digest(tmp_path: Path) -> None:
    arguments = {"query": "q"}
    first = ResearchNotebook(tmp_path / "run-one.private", "task-1")
    first.record("web_search", arguments, {"answer": "first run"})
    source = next(first.root.rglob("*.json"))
    second = ResearchNotebook(tmp_path / "run-two.private", "task-1")
    destination = second.root / "task-1" / source.name
    destination.write_bytes(source.read_bytes())
    destination_scope = hashlib.sha256(
        str(second.root.absolute()).encode("utf-8")
    ).hexdigest()
    _rewrite_with_test_digest(
        destination,
        notebook_scope=destination_scope,
        result={"answer": "forged cross-run"},
    )

    assert second.lookup("web_search", arguments) is None


def test_notebook_creates_private_per_run_trust_anchor(tmp_path: Path) -> None:
    root = tmp_path / "private"
    ResearchNotebook(root, "task-1")
    anchor = root / ".trust-anchor"

    assert anchor.is_file()
    assert stat.S_IMODE(anchor.stat().st_mode) == 0o600


def test_notebook_rejects_symlinked_trust_anchor(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_text("do not overwrite", encoding="utf-8")
    (root / ".trust-anchor").symlink_to(target)

    with pytest.raises(ValueError, match="anchor.*symlink"):
        ResearchNotebook(root, "task-1")

    assert target.read_text(encoding="utf-8") == "do not overwrite"


def test_notebook_rejects_overpermissive_trust_anchor(tmp_path: Path) -> None:
    root = tmp_path / "private"
    ResearchNotebook(root, "task-1")
    anchor = root / ".trust-anchor"
    assert anchor.is_file()
    anchor.chmod(0o644)

    with pytest.raises(ValueError, match="anchor.*permissions"):
        ResearchNotebook(root, "task-2")


def test_notebook_rejects_oversized_trust_anchor(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    anchor = root / ".trust-anchor"
    anchor.write_bytes(b"x" * 10_000)
    anchor.chmod(0o600)

    with pytest.raises(ValueError, match="anchor"):
        ResearchNotebook(root, "task-1")


def test_notebook_rejects_task_ids_that_escape_its_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task_id"):
        ResearchNotebook(tmp_path / "private", "../outside")


def test_notebook_rejects_oversized_entries_without_writing_them(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1", max_entry_bytes=48)

    with pytest.raises(ValueError, match="max_entry_bytes"):
        notebook.record("web_search", {"query": "q"}, {"answer": "x" * 100})

    assert list((tmp_path / "private").rglob("*.json")) == []


def test_notebook_rejects_a_private_root_symlink(tmp_path: Path) -> None:
    public_output = tmp_path / "public-output"
    public_output.mkdir()
    private_root = tmp_path / "private"
    private_root.symlink_to(public_output, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        ResearchNotebook(private_root, "task-1")

    assert list(public_output.iterdir()) == []


def test_notebook_rejects_a_task_directory_symlink(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (private_root / "task-1").symlink_to(escaped, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        ResearchNotebook(private_root, "task-1")

    assert list(escaped.iterdir()) == []


def test_notebook_treats_existing_oversized_entry_as_a_cache_miss(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1", max_entry_bytes=512)
    arguments = {"query": "q"}
    notebook.record("web_search", arguments, {"answer": "a"})
    entry = next((tmp_path / "private").rglob("*.json"))
    raw = json.loads(entry.read_text(encoding="utf-8"))
    raw["result"] = {"answer": "x" * 1_000}
    entry.write_text(json.dumps(raw), encoding="utf-8")

    assert entry.stat().st_size > notebook.max_entry_bytes
    assert notebook.lookup("web_search", arguments) is None


def test_notebook_treats_a_nonregular_entry_as_a_cache_miss(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    arguments = {"query": "q"}
    notebook.record("web_search", arguments, {"answer": "a"})
    entry = next((tmp_path / "private").rglob("*.json"))
    entry_payload = entry.read_bytes()
    entry.unlink()
    os.mkfifo(entry)
    reader = os.open(entry, os.O_RDONLY | os.O_NONBLOCK)
    def write_payload() -> None:
        with entry.open("wb") as writer:
            writer.write(entry_payload)
            writer.flush()
            time.sleep(0.05)

    writer = threading.Thread(target=write_payload)
    writer.start()
    try:
        assert notebook.lookup("web_search", arguments) is None
    finally:
        writer.join()
        os.close(reader)


def _notebook_entry(tool: str, result: dict[str, object], signature: str) -> NotebookEntry:
    return NotebookEntry(
        signature=signature,
        tool=tool,
        result=result,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def test_evidence_cards_deduplicate_sources_prioritize_them_and_keep_ties_stable() -> None:
    entries = (
        _notebook_entry(
            "read_webpage",
            {"url": "https://example.test/fact", "title": "Primary fact", "text": "A" * 500},
            "first-source",
        ),
        _notebook_entry(
            "web_search",
            {"sources": [{"url": "https://example.test/fact", "title": "Duplicate", "snippet": "A" * 500}]},
            "duplicate-url",
        ),
        _notebook_entry("execute_python", {"error": "tool failed"}, "error"),
        _notebook_entry("execute_python", {"text": "same score one"}, "tie-one"),
        _notebook_entry("execute_python", {"text": "same score two"}, "tie-two"),
    )

    cards = select_evidence_cards(entries, max_items=5)

    assert len(cards) == 4
    assert cards[0]["domain"] == "example.test"
    assert cards[0]["tool_kind"] == "read_webpage"
    assert [card["excerpt"] for card in cards[1:3]] == [
        "same score one",
        "same score two",
    ]
    assert cards[-1]["tool_kind"] != "read_webpage"


def test_evidence_cards_never_exceed_json_budget_or_include_full_webpage_bodies() -> None:
    cards = select_evidence_cards(
        tuple(
            _notebook_entry(
                "read_webpage",
                {
                    "url": f"https://source-{index}.example.test/page",
                    "title": "Title " + "x" * 1_000,
                    "text": "private full webpage body " + "y" * 10_000,
                },
                f"entry-{index}",
            )
            for index in range(20)
        ),
        max_json_bytes=2_000,
    )

    encoded = json.dumps(cards, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 2_000
    assert all(len(str(card.get("excerpt", ""))) < 10_000 for card in cards)
    assert all("y" * 10_000 not in json.dumps(card) for card in cards)


def test_evidence_cards_collapse_normalized_whitespace_content_duplicates() -> None:
    cards = select_evidence_cards(
        (
            _notebook_entry(
                "execute_python",
                {"title": "  Computed\n value", "text": "Answer\t is 42"},
                "first",
            ),
            _notebook_entry(
                "execute_python",
                {"title": "Computed value", "text": "Answer is   42"},
                "duplicate",
            ),
        )
    )

    assert len(cards) == 1
