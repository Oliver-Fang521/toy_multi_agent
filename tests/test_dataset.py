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
    monkeypatch.setattr("gaia_agent.dataset.snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr("gaia_agent.dataset.load_dataset", lambda *_args, **_kwargs: records)

    tasks, revision = load_gaia_validation("hf-test-token", cache_dir=tmp_path / "cache")

    assert revision == "revision-abc"
    assert tasks == [
        GaiaTask(
            task_id="synthetic-load",
            question="Synthetic load question",
            level=2,
            final_answer="synthetic gold",
            attachment_path=attachment.resolve(),
        )
    ]
    assert "final_answer" not in tasks[0].model_dump(mode="json")


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
    monkeypatch.setattr("gaia_agent.dataset.snapshot_download", lambda **_: str(snapshot))
    monkeypatch.setattr("gaia_agent.dataset.load_dataset", lambda *_args, **_kwargs: records)

    with pytest.raises(ValueError, match="escapes dataset snapshot"):
        load_gaia_validation("hf-test-token")
