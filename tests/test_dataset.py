import json
from pathlib import Path

import pytest

from gaia_agent.dataset import load_gaia_validation, sample_stratified
from gaia_agent.models import GaiaTask


@pytest.fixture
def synthetic_tasks() -> list[GaiaTask]:
    fixture_path = Path(__file__).parent / "fixtures" / "gaia_records.json"
    records = json.loads(fixture_path.read_text())
    return [
        GaiaTask(
            task_id=record["task_id"],
            question=record["Question"],
            level=int(record["Level"]),
            final_answer=record["Final answer"],
        )
        for record in records
    ]


def test_sample_is_balanced_stable_and_gold_independent(synthetic_tasks: list[GaiaTask]) -> None:
    first = sample_stratified(synthetic_tasks, sample_size=6, seed=42)
    changed_gold = [task.model_copy(update={"final_answer": "changed"}) for task in synthetic_tasks]
    second = sample_stratified(list(reversed(changed_gold)), sample_size=6, seed=42)

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert [task.level for task in first].count(1) == 2
    assert [task.level for task in first].count(2) == 2
    assert [task.level for task in first].count(3) == 2


@pytest.mark.parametrize("sample_size", [0, 2, 4])
def test_sample_requires_at_least_one_task_per_level(sample_size: int, synthetic_tasks: list[GaiaTask]) -> None:
    with pytest.raises(ValueError, match="at least 3 and divisible by 3"):
        sample_stratified(synthetic_tasks, sample_size=sample_size, seed=42)


def test_sample_rejects_a_level_with_too_few_tasks(synthetic_tasks: list[GaiaTask]) -> None:
    with pytest.raises(ValueError, match="level 3 has 1 tasks; need 2"):
        sample_stratified(synthetic_tasks[:-2], sample_size=6, seed=42)


def test_load_validation_converts_records_and_contains_attachments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "hub" / "snapshots" / "revision-abc"
    attachment = snapshot / "files" / "sample.txt"
    attachment.parent.mkdir(parents=True)
    attachment.touch()
    records = [
        {
            "task_id": "synthetic-load",
            "Question": "Synthetic load question",
            "Level": "2",
            "Final answer": "synthetic gold",
            "file_path": "files/sample.txt",
        }
    ]

    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = kwargs.get("local_dir")
        if local_dir is None:
            return str(snapshot)
        materialized = Path(str(local_dir))
        materialized_attachment = materialized / "files" / "sample.txt"
        materialized_attachment.parent.mkdir(parents=True)
        materialized_attachment.write_bytes(attachment.read_bytes())
        return str(materialized)

    monkeypatch.setattr("gaia_agent.dataset.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("gaia_agent.dataset.load_dataset", lambda *_args, **_kwargs: records)

    cache_dir = tmp_path / "cache"
    tasks, revision = load_gaia_validation("hf-test-token", cache_dir=cache_dir)
    materialized_attachment = (
        cache_dir / "gaia-materialized" / "revision-abc" / "files" / "sample.txt"
    ).resolve()

    assert revision == "revision-abc"
    assert materialized_attachment.resolve().is_relative_to(cache_dir.resolve())
    assert tasks == [
        GaiaTask(
            task_id="synthetic-load",
            question="Synthetic load question",
            level=2,
            final_answer="synthetic gold",
            attachment_path=materialized_attachment,
        )
    ]
    assert "final_answer" not in tasks[0].model_dump(mode="json")


def test_load_validation_materializes_hf_snapshot_symlink_as_regular_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    repo_cache = cache_dir / "datasets--gaia-benchmark--GAIA"
    snapshot = repo_cache / "snapshots" / "revision-symlink"
    blob = repo_cache / "blobs" / "blob-hash"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"authorized attachment")
    snapshot_attachment = snapshot / "files" / "sample.ext"
    snapshot_attachment.parent.mkdir(parents=True)
    snapshot_attachment.symlink_to(blob)
    records = [
        {
            "task_id": "synthetic-symlink",
            "Question": "Synthetic symlink question",
            "Level": "3",
            "Final answer": "synthetic gold",
            "file_path": "files/sample.ext",
        }
    ]

    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = kwargs.get("local_dir")
        if local_dir is None:
            return str(snapshot)
        assert kwargs.get("revision") == "revision-symlink"
        materialized = Path(str(local_dir))
        attachment = materialized / "files" / "sample.ext"
        attachment.parent.mkdir(parents=True)
        attachment.write_bytes(blob.read_bytes())
        return str(materialized)

    monkeypatch.setattr("gaia_agent.dataset.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("gaia_agent.dataset.load_dataset", lambda *_args, **_kwargs: records)

    tasks, revision = load_gaia_validation("hf-test-token", cache_dir=cache_dir)

    attachment = tasks[0].attachment_path
    assert revision == "revision-symlink"
    assert attachment is not None
    assert attachment.is_file()
    assert not attachment.is_symlink()
    assert attachment.resolve().is_relative_to(cache_dir.resolve())
    assert attachment.suffix == ".ext"
    assert attachment.stat().st_size == len(b"authorized attachment")


def test_load_validation_rejects_external_materialized_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    repo_cache = cache_dir / "datasets--gaia-benchmark--GAIA"
    snapshot = repo_cache / "snapshots" / "revision-external"
    blob = repo_cache / "blobs" / "blob-hash"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"snapshot blob")
    snapshot_attachment = snapshot / "files" / "sample.txt"
    snapshot_attachment.parent.mkdir(parents=True)
    snapshot_attachment.symlink_to(blob)
    unauthorized = tmp_path / "unauthorized.txt"
    unauthorized.write_bytes(b"unauthorized")
    records = [
        {
            "task_id": "synthetic-external",
            "Question": "Synthetic external question",
            "Level": "1",
            "Final answer": "synthetic gold",
            "file_path": "files/sample.txt",
        }
    ]

    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = kwargs.get("local_dir")
        if local_dir is None:
            return str(snapshot)
        materialized = Path(str(local_dir))
        attachment = materialized / "files" / "sample.txt"
        attachment.parent.mkdir(parents=True)
        attachment.symlink_to(unauthorized)
        return str(materialized)

    monkeypatch.setattr("gaia_agent.dataset.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("gaia_agent.dataset.load_dataset", lambda *_args, **_kwargs: records)

    with pytest.raises(ValueError, match="escapes dataset snapshot"):
        load_gaia_validation("hf-test-token", cache_dir=cache_dir)


def test_load_validation_rejects_oversized_materialized_attachment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    repo_cache = cache_dir / "datasets--gaia-benchmark--GAIA"
    snapshot = repo_cache / "snapshots" / "revision-large"
    blob = repo_cache / "blobs" / "blob-hash"
    blob.parent.mkdir(parents=True)
    with blob.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    snapshot_attachment = snapshot / "files" / "sample.bin"
    snapshot_attachment.parent.mkdir(parents=True)
    snapshot_attachment.symlink_to(blob)
    records = [
        {
            "task_id": "synthetic-large",
            "Question": "Synthetic large question",
            "Level": "2",
            "Final answer": "synthetic gold",
            "file_path": "files/sample.bin",
        }
    ]

    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = kwargs.get("local_dir")
        if local_dir is None:
            return str(snapshot)
        materialized = Path(str(local_dir))
        attachment = materialized / "files" / "sample.bin"
        attachment.parent.mkdir(parents=True)
        with attachment.open("wb") as stream:
            stream.truncate(32 * 1024 * 1024 + 1)
        return str(materialized)

    monkeypatch.setattr("gaia_agent.dataset.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("gaia_agent.dataset.load_dataset", lambda *_args, **_kwargs: records)

    with pytest.raises(ValueError, match="size limit"):
        load_gaia_validation("hf-test-token", cache_dir=cache_dir)


def test_load_validation_rejects_attachment_paths_outside_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshots" / "revision-safe"
    records = [
        {
            "task_id": "synthetic-unsafe",
            "Question": "Synthetic unsafe question",
            "Level": "1",
            "Final answer": "synthetic gold",
            "file_path": "../outside.txt",
        }
    ]
    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = kwargs.get("local_dir")
        return str(snapshot if local_dir is None else local_dir)

    monkeypatch.setattr("gaia_agent.dataset.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("gaia_agent.dataset.load_dataset", lambda *_args, **_kwargs: records)

    with pytest.raises(ValueError, match="escapes dataset snapshot"):
        load_gaia_validation("hf-test-token")
