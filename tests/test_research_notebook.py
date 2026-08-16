from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from gaia_agent.research_notebook import ResearchNotebook


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
