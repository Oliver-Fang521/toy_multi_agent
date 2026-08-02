from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from gaia_agent.models import AuditEvent, TaskPrediction, Usage
from gaia_agent.persistence import RunManifest, RunStore


def manifest_for(task_ids: list[str]) -> RunManifest:
    return RunManifest(
        selected_task_ids=tuple(task_ids),
        sample_size=len(task_ids),
        seed=42,
        model="test-model",
        reasoning_effort="medium",
        dataset_revision="synthetic-revision",
        started_at=datetime(2026, 8, 2, tzinfo=UTC),
    )


@pytest.fixture
def prediction() -> TaskPrediction:
    return TaskPrediction(
        task_id="task-1",
        level=1,
        answer="a model answer that must not be persisted",
        correct=True,
        duration_seconds=2.5,
        usage=Usage(input_tokens=2, output_tokens=3, reasoning_tokens=4, total_tokens=9),
        critic_verdict="approve",
        evidence_ids=("raw-evidence-id",),
        vote_count=3,
    )


def test_checkpoint_round_trip_is_atomic_and_resumable(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))

    store.append(prediction)

    assert store.completed_task_ids() == frozenset({prediction.task_id})
    assert [item.task_id for item in store.load_predictions()] == [prediction.task_id]
    assert not list(tmp_path.rglob("*.tmp"))


def test_public_artifacts_exclude_gated_content(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append(prediction)
    store.write_plan(
        prediction.task_id,
        {"restatement": "question text", "steps": [{"instruction": "model prompt"}]},
    )
    store.append_note(
        prediction.task_id,
        {
            "question": "gated question",
            "attachment_path": "/private/attachment.pdf",
            "secret": "sk-secret-value",
            "evidence_ids": ["raw-evidence-id"],
        },
    )

    combined = "".join(path.read_text() for path in tmp_path.rglob("*.json*"))
    for forbidden in (
        '"question"',
        '"final_answer"',
        '"attachment_path"',
        "model prompt",
        "sk-secret-value",
        "raw-evidence-id",
    ):
        assert forbidden not in combined


def test_resume_rejects_a_different_manifest(tmp_path) -> None:
    RunStore.create(tmp_path, manifest_for(["task-1"]))

    with pytest.raises(ValueError, match="manifest"):
        RunStore.open(tmp_path, manifest_for(["task-2"]))


def test_duplicate_prediction_is_rejected(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append(prediction)

    with pytest.raises(ValueError, match="duplicate"):
        store.append(prediction)


def test_corrupt_or_incomplete_checkpoint_records_are_not_resumed(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id, "task-2"]))
    store.append(prediction)
    store.predictions_path.write_text(
        store.predictions_path.read_text()
        + '{"task_id":"task-2","level":2}\n'
        + "not-json\n",
        encoding="utf-8",
    )

    assert store.completed_task_ids() == frozenset({prediction.task_id})


def test_audit_append_is_jsonl_and_redacts_secret_values(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append_audit(
        prediction.task_id,
        AuditEvent(
            sequence=1,
            event_type="tool_called",
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            payload={"api_key": "sk-secret", "query": "safe aggregate", "path": "/secret"},
        ),
    )

    audit = (store.task_directory(prediction.task_id) / "audit.jsonl").read_text()
    assert audit.count("\n") == 1
    assert "sk-secret" not in audit
    assert '"api_key"' not in audit
    assert '"path"' not in audit
    assert "safe aggregate" in audit


def test_task_paths_are_contained_and_symlinks_are_rejected(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))

    with pytest.raises(ValueError, match="task ID"):
        store.task_directory("../escape")

    task_directory = store.task_directory(prediction.task_id)
    task_directory.rmdir()
    task_directory.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.append_note(prediction.task_id, {"safe": True})


def test_manifest_and_prediction_json_are_deterministically_sorted(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append(prediction)

    payload = json.loads(store.predictions_path.read_text())
    assert list(payload) == sorted(payload)


def test_integrity_accepts_task_six_opaque_evidence_ids(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append(prediction.model_copy(update={"evidence_ids": ("evidence-0001",)}))

    store.verify_integrity()


def test_public_prediction_redacts_a_secret_embedded_in_model_output(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append(prediction.model_copy(update={"answer": "result sk-secret-value"}))

    assert "sk-secret-value" not in store.predictions_path.read_text()


def test_create_refuses_a_root_that_traverses_a_symlink(tmp_path, prediction) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        RunStore.create(linked_directory / "run", manifest_for([prediction.task_id]))


def test_create_refuses_an_orphaned_checkpoint(tmp_path, prediction) -> None:
    (tmp_path / "predictions.jsonl").write_text("partial checkpoint\n", encoding="utf-8")

    with pytest.raises(ValueError, match="orphaned"):
        RunStore.create(tmp_path, manifest_for([prediction.task_id]))
