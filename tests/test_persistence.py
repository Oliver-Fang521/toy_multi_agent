from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gaia_agent.models import (
    AgentPlan,
    AuditEvent,
    PlanStep,
    SpecialistName,
    TaskPrediction,
    TerminalErrorCategory,
    Usage,
)
from gaia_agent.persistence import PublicNote, RunManifest, RunStore


def manifest_for(task_ids: list[str]) -> RunManifest:
    return RunManifest(
        selected_task_ids=tuple(task_ids),
        expected_levels={task_id: index % 3 + 1 for index, task_id in enumerate(task_ids)},
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
        AgentPlan(
            restatement="question text",
            steps=(PlanStep(specialist=SpecialistName.LOOKUP, instruction="model prompt"),),
        ),
    )
    store.append_note(
        prediction.task_id,
        PublicNote(category="checkpoint_skipped", count=1),
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

    assert store.completed_task_ids() == frozenset()


def test_audit_append_is_jsonl_and_redacts_secret_values(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append_audit(
        prediction.task_id,
        AuditEvent(
            sequence=1,
            event_type="tool_call_started",
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            payload={"tool": "web_search", "call_number": 1, "argument_keys": ["query"]},
        ),
    )

    audit = (store.task_directory(prediction.task_id) / "audit.jsonl").read_text()
    assert audit.count("\n") == 1
    assert '"tool":"web_search"' in audit


def test_task_paths_are_contained_and_symlinks_are_rejected(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))

    with pytest.raises(ValueError, match="task ID"):
        store.task_directory("../escape")

    task_directory = store.task_directory(prediction.task_id)
    task_directory.rmdir()
    task_directory.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.append_note(prediction.task_id, PublicNote(category="checkpoint_skipped", count=1))


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


def test_parseable_invalid_duplicate_makes_a_task_incomplete_until_next_append_repairs(
    tmp_path, prediction
) -> None:
    store = RunStore.create(tmp_path, manifest_for(["task-1", "task-2"]))
    second = prediction.model_copy(update={"task_id": "task-2", "level": 2})
    store.append(prediction)
    store.append(second)
    store.predictions_path.write_text(
        store.predictions_path.read_text() + '{"task_id":"task-1","level":1}\n',
        encoding="utf-8",
    )

    assert store.completed_task_ids() == frozenset({"task-2"})

    store.append(prediction.model_copy(update={"answer": "repaired"}))

    assert store.completed_task_ids() == frozenset({"task-1", "task-2"})
    assert [item.task_id for item in store.load_predictions()] == ["task-1", "task-2"]
    assert store.predictions_path.read_text().count("\n") == 2


def test_unparseable_checkpoint_line_fails_safe_until_next_append_repairs(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for(["task-1", "task-2"]))
    second = prediction.model_copy(update={"task_id": "task-2", "level": 2})
    store.append(prediction)
    store.append(second)
    store.predictions_path.write_text(
        store.predictions_path.read_text() + "not-json\n",
        encoding="utf-8",
    )

    assert store.completed_task_ids() == frozenset()

    store.append(prediction.model_copy(update={"answer": "repaired"}))

    assert store.completed_task_ids() == frozenset({"task-1", "task-2"})


def test_manifest_requires_a_complete_typed_task_level_mapping() -> None:
    manifest = manifest_for(["task-1", "task-2"])

    assert manifest.expected_levels == {"task-1": 1, "task-2": 2}
    with pytest.raises(ValidationError, match="expected_levels"):
        RunManifest(
            selected_task_ids=("task-1", "task-2"),
            expected_levels={"task-1": 1},
            sample_size=2,
            seed=42,
            model="test-model",
            reasoning_effort="medium",
            dataset_revision="synthetic-revision",
        )


def test_prediction_level_must_match_manifest_on_append_resume_and_integrity(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    wrong_level = prediction.model_copy(update={"level": 2})

    with pytest.raises(ValueError, match="level"):
        store.append(wrong_level)

    store.append(prediction)
    persisted = json.loads(store.predictions_path.read_text())
    persisted["level"] = 2
    store.predictions_path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")

    assert store.completed_task_ids() == frozenset()
    with pytest.raises(ValueError, match="corrupt"):
        store.verify_integrity()


def test_manifest_level_mapping_participates_in_resume_identity(tmp_path) -> None:
    manifest = manifest_for(["task-1"])
    RunStore.create(tmp_path, manifest)
    changed_levels = manifest.model_copy(update={"expected_levels": {"task-1": 2}})

    with pytest.raises(ValueError, match="manifest"):
        RunStore.open(tmp_path, changed_levels)


def test_artifact_inputs_reject_unknown_keys_and_nested_arbitrary_text(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))

    with pytest.raises(TypeError, match="AgentPlan"):
        store.write_plan(prediction.task_id, {"steps": []})
    with pytest.raises(TypeError, match="PublicNote"):
        store.append_note(prediction.task_id, {"category": "timeout", "count": 1})
    with pytest.raises(ValueError, match="payload"):
        store.append_audit(
            prediction.task_id,
            AuditEvent(
                sequence=1,
                event_type="tool_call_started",
                timestamp=datetime(2026, 8, 2, tzinfo=UTC),
                payload={"tool": "web_search", "metadata": {"innocuous": "gated question"}},
            ),
        )


def test_manifest_rejects_markdown_injection_in_public_strings() -> None:
    with pytest.raises(ValidationError, match="model"):
        RunManifest(
            selected_task_ids=("task-1",),
            expected_levels={"task-1": 1},
            sample_size=1,
            seed=42,
            model="test-model\n# injected",
            reasoning_effort="medium",
            dataset_revision="synthetic-revision",
        )


def test_task_six_terminal_not_run_prediction_and_audit_are_publicly_reportable(tmp_path, prediction) -> None:
    from gaia_agent.reporting import summarize, write_reports

    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    terminal = prediction.model_copy(
        update={
            "answer": None,
            "correct": False,
            "critic_verdict": "not_run",
            "error_category": TerminalErrorCategory.TIMEOUT,
        }
    )
    store.append(terminal)
    store.append_audit(
        prediction.task_id,
        AuditEvent(
            sequence=1,
            event_type="terminal_failure",
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            payload={"category": "timeout", "critic_verdict": "not_run"},
        ),
    )

    _, report_path = write_reports(store, summarize(store.load_predictions()))

    assert '"critic_verdict":"not_run"' in store.predictions_path.read_text()
    assert "- not_run: 1" in report_path.read_text()


def test_real_task_six_tool_audit_shapes_persist_only_key_names(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    shapes = (
        ("web_search", ["query"], ["answer", "sources"]),
        ("read_webpage", ["url"], ["url", "status_code", "content_type", "text", "truncated"]),
        ("execute_python", ["code", "paths"], ["output_text", "outputs"]),
        ("read_attachment", ["path"], ["path", "kind", "text", "page_count", "rows", "data", "sheets", "media_type", "size_bytes"]),
        ("submit_answer", ["answer", "evidence_ids"], ["answer", "evidence_ids"]),
    )
    sequence = 1
    for tool, argument_keys, result_keys in shapes:
        store.append_audit(
            prediction.task_id,
            AuditEvent(
                sequence=sequence,
                event_type="tool_call_started",
                timestamp=datetime(2026, 8, 2, tzinfo=UTC),
                payload={"tool": tool, "call_number": sequence, "argument_keys": argument_keys},
            ),
        )
        sequence += 1
        store.append_audit(
            prediction.task_id,
            AuditEvent(
                sequence=sequence,
                event_type="tool_call_completed",
                timestamp=datetime(2026, 8, 2, tzinfo=UTC),
                payload={
                    "tool": tool,
                    "call_number": sequence - 1,
                    "argument_keys": argument_keys,
                    "result_keys": result_keys,
                },
            ),
        )
        sequence += 1

    audit = (store.task_directory(prediction.task_id) / "audit.jsonl").read_text()
    assert '"result_keys":["answer","sources"]' in audit
    assert "gated question" not in audit


def test_manifest_rejects_known_environment_secrets_in_all_public_identifiers(tmp_path, monkeypatch) -> None:
    secret = "env-private-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    values = {
        "run_id": secret,
        "model": secret,
        "dataset_name": secret,
        "dataset_revision": secret,
        "dataset_split": secret,
        "package_versions": {"gaia_agent": secret},
        "selected_task_ids": ("task-1",),
        "expected_levels": {"task-1": 1},
    }
    with pytest.raises(ValidationError, match="public"):
        RunManifest(
            sample_size=1,
            seed=42,
            reasoning_effort="medium",
            **values,
        )
    safe = RunStore.create(tmp_path, manifest_for(["task-1"]))
    assert secret not in "".join(path.read_text() for path in tmp_path.rglob("*"))
    assert safe.manifest_path.exists()
