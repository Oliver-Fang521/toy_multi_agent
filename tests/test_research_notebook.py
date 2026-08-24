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

from gaia_agent.models import (
    AgentPlan,
    CriticDecision,
    Evidence,
    PlanStep,
    SpecialistName,
    StageCheckpoint,
    StageToolCallCounts,
    Usage,
)
from gaia_agent.research_notebook import (
    NotebookEntry,
    ResearchNotebook,
    _is_safe_source_url,
    _open_directory,
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


@pytest.mark.parametrize("invalid_schema_version", [True, 1.0])
def test_notebook_rejects_non_integer_trust_anchor_schema_versions(
    tmp_path: Path,
    invalid_schema_version: object,
) -> None:
    root = tmp_path / "private"
    ResearchNotebook(root, "task-1")
    anchor = root / ".trust-anchor"
    payload = json.loads(anchor.read_text(encoding="utf-8"))
    payload["schema_version"] = invalid_schema_version
    anchor.write_text(json.dumps(payload), encoding="utf-8")
    anchor.chmod(0o600)

    with pytest.raises(ValueError, match="anchor schema"):
        ResearchNotebook(root, "task-2")


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


@pytest.mark.parametrize("task_id", [".", ".."])
def test_notebook_rejects_navigation_task_ids(tmp_path: Path, task_id: str) -> None:
    with pytest.raises(ValueError, match="task_id"):
        ResearchNotebook(tmp_path / "private", task_id)


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


def _stage_checkpoint(
    *,
    cursor: str = "verify",
    candidate: str = "answer",
    decision: CriticDecision | None = None,
) -> StageCheckpoint:
    raw_evidence_id = "provider-evidence-1"
    return StageCheckpoint(
        cursor=cursor,
        plan=AgentPlan(
            restatement="private restatement",
            steps=(
                PlanStep(
                    specialist=SpecialistName.LOOKUP,
                    instruction="private instruction",
                ),
            ),
        ),
        evidence=(
            Evidence(
                evidence_id="evidence-0001",
                kind="source",
                summary="private summary",
                source_url="https://private.example/source",
            ),
        ),
        evidence_origins={"evidence-0001": 1},
        raw_evidence_id_digests=(
            hashlib.sha256(raw_evidence_id.encode("utf-8")).hexdigest(),
        ),
        candidate_origins={1: candidate},
        candidate=candidate,
        usage=Usage(input_tokens=4, output_tokens=6, total_tokens=10),
        step_count=1,
        verify_count=0,
        correction_count=0,
        next_step_index=1,
        soft_step_failures=0,
        logical_tool_calls=2,
        stage_logical_tool_calls=StageToolCallCounts(specialist=2),
        critic_verdict="not_run" if decision is None else decision.verdict,
        decision=decision,
    )


def test_stage_checkpoint_round_trip_is_private_authenticated_and_bounded(
    tmp_path: Path,
) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    checkpoint = _stage_checkpoint()

    notebook.save_stage_checkpoint(checkpoint)

    path = notebook.root / "task-1" / ".stage-checkpoint.json"
    assert notebook.load_stage_checkpoint() == checkpoint
    assert path.is_file()
    assert path.stat().st_size <= notebook.max_entry_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_stage_checkpoint_preserves_a_validated_critic_decision(tmp_path: Path) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    decision = CriticDecision(
        verdict="approve",
        feedback="private approval",
        checked_evidence_ids=("evidence-0001",),
    )
    checkpoint = _stage_checkpoint(cursor="submit", decision=decision)

    notebook.save_stage_checkpoint(checkpoint)

    assert notebook.load_stage_checkpoint() == checkpoint
    assert notebook.load_stage_checkpoint().decision == decision  # type: ignore[union-attr]


@pytest.mark.parametrize("keep_evidence", [False, True], ids=["no-evidence", "unchecked"])
def test_submit_checkpoint_requires_checked_known_evidence(keep_evidence: bool) -> None:
    checkpoint = _stage_checkpoint()
    payload = checkpoint.model_dump(mode="python")
    payload.update(
        {
            "cursor": "submit",
            "critic_verdict": "approve",
            "decision": CriticDecision(
                verdict="approve",
                feedback="unsafe approval",
                checked_evidence_ids=(),
            ),
        }
    )
    if not keep_evidence:
        payload.update(
            {
                "evidence": (),
                "evidence_origins": {},
                "raw_evidence_id_digests": (),
            }
        )

    with pytest.raises(ValueError, match="submit.*evidence"):
        StageCheckpoint.model_validate(payload)


def test_stage_checkpoint_requires_a_nonempty_plan() -> None:
    checkpoint = _stage_checkpoint()
    payload = checkpoint.model_dump(mode="python")
    payload["plan"] = checkpoint.plan.model_copy(update={"steps": ()})
    payload["next_step_index"] = 0

    with pytest.raises(ValueError, match="plan"):
        StageCheckpoint.model_validate(payload)


def test_act_checkpoint_requires_an_executable_next_step() -> None:
    checkpoint = _stage_checkpoint()
    payload = checkpoint.model_dump(mode="python")
    payload.update({"cursor": "act", "next_step_index": len(checkpoint.plan.steps)})

    with pytest.raises(ValueError, match="act.*next step"):
        StageCheckpoint.model_validate(payload)


@pytest.mark.parametrize(
    ("cursor", "decision"),
    [
        pytest.param("evidence_repair", None, id="evidence-repair"),
        pytest.param("verify", None, id="verify"),
        pytest.param(
            "correct",
            CriticDecision(
                verdict="revise",
                feedback="revise",
                checked_evidence_ids=("evidence-0001",),
                corrected_answer="corrected",
            ),
            id="correct",
        ),
        pytest.param(
            "submit",
            CriticDecision(
                verdict="approve",
                feedback="approve",
                checked_evidence_ids=("evidence-0001",),
            ),
            id="submit",
        ),
    ],
)
def test_non_act_checkpoint_allows_a_valid_early_index(
    cursor: str,
    decision: CriticDecision | None,
) -> None:
    checkpoint = _stage_checkpoint()
    payload = checkpoint.model_dump(mode="python")
    payload.update(
        {
            "cursor": cursor,
            "next_step_index": 0,
            "critic_verdict": "not_run" if decision is None else decision.verdict,
            "decision": decision,
        }
    )

    assert StageCheckpoint.model_validate(payload).next_step_index == 0


def test_submit_checkpoint_rejects_unknown_checked_evidence() -> None:
    checkpoint = _stage_checkpoint()
    payload = checkpoint.model_dump(mode="python")
    payload.update(
        {
            "cursor": "submit",
            "critic_verdict": "approve",
            "decision": CriticDecision(
                verdict="approve",
                feedback="unsafe approval",
                checked_evidence_ids=("evidence-9999",),
            ),
        }
    )

    with pytest.raises(ValueError, match="invalid evidence"):
        StageCheckpoint.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("contract", "gaia.stage-checkpoint.stale", id="contract"),
        pytest.param("task_id", "task-2", id="task"),
        pytest.param("unexpected", "field", id="unknown-key"),
    ],
)
def test_stage_checkpoint_tampering_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    notebook.save_stage_checkpoint(_stage_checkpoint())
    path = notebook.root / "task-1" / ".stage-checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert notebook.load_stage_checkpoint() is None


def test_stage_checkpoint_nested_unknown_keys_fail_closed(tmp_path: Path) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    notebook.save_stage_checkpoint(_stage_checkpoint())
    path = notebook.root / "task-1" / ".stage-checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint"]["evidence"][0]["private_extra"] = "not contracted"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert notebook.load_stage_checkpoint() is None


def test_stage_checkpoint_from_wrong_run_generation_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "private"
    first = ResearchNotebook(root, "task-1", run_generation="run-1")
    first.save_stage_checkpoint(_stage_checkpoint())

    second = ResearchNotebook(root, "task-1", run_generation="run-2")

    assert second.load_stage_checkpoint() is None


def test_stage_checkpoint_from_another_notebook_fails_closed(tmp_path: Path) -> None:
    first = ResearchNotebook(
        tmp_path / "first.private",
        "task-1",
        run_generation="run-1",
    )
    first.save_stage_checkpoint(_stage_checkpoint())
    second = ResearchNotebook(
        tmp_path / "second.private",
        "task-1",
        run_generation="run-1",
    )
    source = first.root / "task-1" / ".stage-checkpoint.json"
    destination = second.root / "task-1" / ".stage-checkpoint.json"
    destination.write_bytes(source.read_bytes())
    destination.chmod(0o600)

    assert second.load_stage_checkpoint() is None


def test_stage_checkpoint_symlink_fails_closed_without_reading_target(tmp_path: Path) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    target = tmp_path / "outside.json"
    target.write_text('{"candidate":"outside private value"}', encoding="utf-8")
    path = notebook.root / "task-1" / ".stage-checkpoint.json"
    path.symlink_to(target)

    assert notebook.load_stage_checkpoint() is None
    assert target.read_text(encoding="utf-8") == '{"candidate":"outside private value"}'


def test_stage_checkpoint_oversized_payload_fails_closed(tmp_path: Path) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        max_entry_bytes=1_024,
        run_generation="run-1",
    )
    path = notebook.root / "task-1" / ".stage-checkpoint.json"
    path.write_bytes(b"x" * 1_025)
    path.chmod(0o600)

    assert notebook.load_stage_checkpoint() is None


def test_stage_checkpoint_oversized_save_keeps_last_atomic_checkpoint(tmp_path: Path) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        max_entry_bytes=2_048,
        run_generation="run-1",
    )
    checkpoint = _stage_checkpoint()
    notebook.save_stage_checkpoint(checkpoint)
    oversized = checkpoint.model_copy(
        update={"candidate": "x" * 4_096, "candidate_origins": {1: "x" * 4_096}}
    )

    with pytest.raises(ValueError, match="max_entry_bytes"):
        notebook.save_stage_checkpoint(oversized)

    assert notebook.load_stage_checkpoint() == checkpoint


def test_stage_checkpoint_with_nonprivate_mode_fails_closed(tmp_path: Path) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    notebook.save_stage_checkpoint(_stage_checkpoint())
    path = notebook.root / "task-1" / ".stage-checkpoint.json"
    path.chmod(0o644)

    assert notebook.load_stage_checkpoint() is None


def test_stage_checkpoint_read_cannot_follow_a_swapped_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    checkpoint = _stage_checkpoint()
    notebook.save_stage_checkpoint(checkpoint)
    task_root = notebook.root / "task-1"
    detached = notebook.root / "task-1.detached"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    outside_checkpoint = outside / ".stage-checkpoint.json"
    outside_checkpoint.write_bytes((task_root / ".stage-checkpoint.json").read_bytes())
    outside_checkpoint.chmod(0o600)
    original_assert = ResearchNotebook._assert_private_directory
    swapped = False

    def swap_after_validation(path: Path) -> None:
        nonlocal swapped
        original_assert(path)
        if path == task_root and not swapped:
            swapped = True
            task_root.rename(detached)
            task_root.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(swap_after_validation),
    )

    assert notebook.load_stage_checkpoint() is None

    task_root.unlink()
    detached.rename(task_root)
    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(original_assert),
    )
    assert notebook.load_stage_checkpoint() == checkpoint


def test_stage_checkpoint_write_cannot_follow_a_swapped_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = ResearchNotebook(
        tmp_path / "private",
        "task-1",
        run_generation="run-1",
    )
    checkpoint = _stage_checkpoint()
    notebook.save_stage_checkpoint(checkpoint)
    replacement = checkpoint.model_copy(
        update={"candidate": "replacement", "candidate_origins": {1: "replacement"}}
    )
    task_root = notebook.root / "task-1"
    detached = notebook.root / "task-1.detached"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    original_assert = ResearchNotebook._assert_private_directory
    swapped = False

    def swap_after_validation(path: Path) -> None:
        nonlocal swapped
        original_assert(path)
        if path == task_root and not swapped:
            swapped = True
            task_root.rename(detached)
            task_root.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(swap_after_validation),
    )

    with pytest.raises((OSError, ValueError)):
        notebook.save_stage_checkpoint(replacement)

    assert not (outside / ".stage-checkpoint.json").exists()
    task_root.unlink()
    detached.rename(task_root)
    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(original_assert),
    )
    assert notebook.load_stage_checkpoint() == checkpoint


def test_stage_checkpoint_read_cannot_follow_a_swapped_intermediate_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "ancestor"
    notebook = ResearchNotebook(
        ancestor / "private",
        "task-1",
        run_generation="run-1",
    )
    checkpoint = _stage_checkpoint()
    notebook.save_stage_checkpoint(checkpoint)
    task_root = notebook.root / "task-1"
    detached = tmp_path / "ancestor.detached"
    outside_ancestor = tmp_path / "outside-ancestor"
    outside_task = outside_ancestor / "private" / "task-1"
    outside_task.mkdir(mode=0o700, parents=True)
    outside_checkpoint = outside_task / ".stage-checkpoint.json"
    outside_checkpoint.write_bytes((task_root / ".stage-checkpoint.json").read_bytes())
    outside_checkpoint.chmod(0o600)
    original_assert = ResearchNotebook._assert_private_directory
    swapped = False

    def swap_after_validation(path: Path) -> None:
        nonlocal swapped
        original_assert(path)
        if path == task_root and not swapped:
            swapped = True
            ancestor.rename(detached)
            ancestor.symlink_to(outside_ancestor, target_is_directory=True)

    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(swap_after_validation),
    )

    assert notebook.load_stage_checkpoint() is None

    ancestor.unlink()
    detached.rename(ancestor)
    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(original_assert),
    )
    assert notebook.load_stage_checkpoint() == checkpoint


def test_stage_checkpoint_write_cannot_follow_a_swapped_intermediate_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "ancestor"
    notebook = ResearchNotebook(
        ancestor / "private",
        "task-1",
        run_generation="run-1",
    )
    checkpoint = _stage_checkpoint()
    notebook.save_stage_checkpoint(checkpoint)
    replacement = checkpoint.model_copy(
        update={"candidate": "replacement", "candidate_origins": {1: "replacement"}}
    )
    task_root = notebook.root / "task-1"
    detached = tmp_path / "ancestor.detached"
    outside_ancestor = tmp_path / "outside-ancestor"
    outside_task = outside_ancestor / "private" / "task-1"
    outside_task.mkdir(mode=0o700, parents=True)
    original_assert = ResearchNotebook._assert_private_directory
    swapped = False

    def swap_after_validation(path: Path) -> None:
        nonlocal swapped
        original_assert(path)
        if path == task_root and not swapped:
            swapped = True
            ancestor.rename(detached)
            ancestor.symlink_to(outside_ancestor, target_is_directory=True)

    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(swap_after_validation),
    )

    with pytest.raises((OSError, ValueError)):
        notebook.save_stage_checkpoint(replacement)

    assert not (outside_task / ".stage-checkpoint.json").exists()
    ancestor.unlink()
    detached.rename(ancestor)
    monkeypatch.setattr(
        ResearchNotebook,
        "_assert_private_directory",
        staticmethod(original_assert),
    )
    assert notebook.load_stage_checkpoint() == checkpoint


def test_stage_checkpoint_root_dotdot_cannot_be_relocated_to_an_outside_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    moving = parent / "moving"
    moving.mkdir(parents=True)
    notebook = ResearchNotebook(
        moving / ".." / "private",
        "task-1",
        run_generation="run-1",
    )
    assert notebook.root == parent / "private"
    checkpoint = _stage_checkpoint()
    redirected = checkpoint.model_copy(
        update={"candidate": "redirected", "candidate_origins": {1: "redirected"}}
    )
    notebook.save_stage_checkpoint(redirected)
    redirected_payload = (
        notebook.root / "task-1" / ".stage-checkpoint.json"
    ).read_bytes()
    notebook.save_stage_checkpoint(checkpoint)
    outside_parent = tmp_path / "outside-parent"
    outside_task = outside_parent / "private" / "task-1"
    outside_task.mkdir(mode=0o700, parents=True)
    outside_checkpoint = outside_task / ".stage-checkpoint.json"
    outside_checkpoint.write_bytes(redirected_payload)
    outside_checkpoint.chmod(0o600)
    real_open = os.open
    relocated = False

    def relocate_before_dotdot(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal relocated
        if path == ".." and not relocated:
            relocated = True
            moving.rename(outside_parent / "moving")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", relocate_before_dotdot)

    assert notebook.load_stage_checkpoint() == checkpoint
    assert not relocated


def test_checkpoint_directory_walker_rejects_navigation_components(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    descriptor = -1
    try:
        with pytest.raises(ValueError, match="navigation"):
            descriptor = _open_directory(child / "..")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


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


def test_evidence_link_cards_expose_only_bounded_authenticated_metadata(
    tmp_path: Path,
) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    notebook.record(
        "web_search",
        {"query": "fact"},
        {
            "answer": "Authenticated answer " + "x" * 5_000,
            "sources": [
                {
                    "url": "https://example.test/private/path?view=summary",
                    "title": "Primary title " + "t" * 1_000,
                }
            ],
            "credential": "never expose",
            "diagnostic": "private diagnostic",
        },
    )

    cards = notebook.linkable_evidence_cards()

    assert len(cards) == 1
    assert set(cards[0]) <= {"entry_hash", "tool_kind", "title", "domain", "excerpt"}
    assert cards[0]["tool_kind"] == "web_search"
    assert cards[0]["domain"] == "example.test"
    assert len(cards[0]["title"]) <= 256
    assert len(cards[0]["excerpt"]) <= 1_536
    encoded = json.dumps(cards)
    assert "https://example.test/private/path?view=summary" not in encoded
    assert "password" not in encoded
    assert "credential" not in encoded
    assert "private diagnostic" not in encoded


def test_evidence_link_cards_ignore_failed_unusable_and_unauthenticated_entries(
    tmp_path: Path,
) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    notebook.record("execute_python", {"instruction": "fail"}, {"error": "traceback"})
    notebook.record("web_search", {"query": "no source"}, {"answer": "no URL"})
    notebook.record(
        "read_webpage",
        {"url": "https://example.test/fact"},
        {"url": "https://example.test/fact", "text": "authenticated fact"},
    )
    entry = next(
        path
        for path in (tmp_path / "private" / "task-1").glob("*.json")
        if "authenticated fact" in path.read_text(encoding="utf-8")
    )
    raw = json.loads(entry.read_text(encoding="utf-8"))
    raw["result"]["text"] = "tampered fact"
    entry.write_text(json.dumps(raw), encoding="utf-8")

    assert notebook.linkable_evidence_cards() == ()


def test_evidence_for_hashes_resolves_authenticated_source_url_not_model_fields(
    tmp_path: Path,
) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    trusted_url = "https://primary.example/fact"
    notebook.record(
        "web_search",
        {"query": "fact"},
        {
            "answer": "Supported answer",
            "sources": [{"url": trusted_url, "title": "Primary source"}],
        },
    )
    entry_hash = notebook.linkable_evidence_cards()[0]["entry_hash"]

    evidence = notebook.evidence_for_hashes((entry_hash, "unknown", entry_hash))

    assert evidence == (
        Evidence(
            evidence_id=entry_hash,
            kind="source",
            summary="Supported answer",
            source_url=trusted_url,
        ),
    )


def test_evidence_for_hashes_maps_only_usable_typed_non_source_results(
    tmp_path: Path,
) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1")
    notebook.record(
        "execute_python",
        {"instruction": "calculate"},
        {"output_text": "The computed value is 42", "outputs": []},
    )
    notebook.record(
        "transcribe_audio",
        {"path": "audio.wav"},
        {"text": "The speaker said forty-two"},
    )
    notebook.record(
        "read_attachment",
        {"path": "private/image.png"},
        {"path": "private/image.png", "kind": "image", "media_type": "image/png"},
    )
    cards = notebook.linkable_evidence_cards()
    hashes_by_tool = {card["tool_kind"]: card["entry_hash"] for card in cards}

    evidence = notebook.evidence_for_hashes(
        (hashes_by_tool["execute_python"], hashes_by_tool["transcribe_audio"])
    )

    assert [item.kind for item in evidence] == ["computation", "file"]
    assert [item.summary for item in evidence] == [
        "The computed value is 42",
        "The speaker said forty-two",
    ]
    assert all(item.source_url is None for item in evidence)
    assert "read_attachment" not in hashes_by_tool


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/fact?q=public#section-two",
        "https://example。com/fact",
        "http://8.8.8.8/fact",
        "http://8。8。8。8/fact",
        "https://[2606:4700:4700::1111]/fact",
    ],
)
def test_safe_source_url_accepts_public_hosts_without_credentials(url: str) -> None:
    assert _is_safe_source_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://user:password@example.com/fact", id="userinfo"),
        pytest.param("http://localhost/fact", id="localhost"),
        pytest.param("http://api.localhost./fact", id="localhost-subdomain"),
        pytest.param("http://localhost。/fact", id="ideographic-dot-localhost"),
        pytest.param("http://localhost．/fact", id="fullwidth-dot-localhost"),
        pytest.param("http://localhost｡/fact", id="halfwidth-dot-localhost"),
        pytest.param("http://api.localhost。/fact", id="unicode-dot-localhost-subdomain"),
        pytest.param("http://127.0.0.1/fact", id="ipv4-loopback"),
        pytest.param("http://127.1/fact", id="short-ipv4-loopback"),
        pytest.param("http://127.0.1/fact", id="three-part-ipv4-loopback"),
        pytest.param("http://127.0.0.01/fact", id="leading-zero-ipv4-loopback"),
        pytest.param("http://2130706433/fact", id="decimal-integer-ipv4-loopback"),
        pytest.param("http://0x7f000001/fact", id="hex-integer-ipv4-loopback"),
        pytest.param("http://017700000001/fact", id="octal-integer-ipv4-loopback"),
        pytest.param("http://0177.0.0.1/fact", id="octal-part-ipv4-loopback"),
        pytest.param("http://0x7f.1/fact", id="hex-part-ipv4-loopback"),
        pytest.param("http://8.8.2056/fact", id="ambiguous-public-ipv4-parts"),
        pytest.param("http://134744072/fact", id="ambiguous-public-ipv4-integer"),
        pytest.param("http://10.0.0.1/fact", id="ipv4-private"),
        pytest.param("http://169.254.10.20/fact", id="ipv4-link-local"),
        pytest.param("http://224.0.0.1/fact", id="ipv4-multicast"),
        pytest.param("http://192.0.2.1/fact", id="ipv4-reserved"),
        pytest.param("http://0.0.0.0/fact", id="ipv4-unspecified"),
        pytest.param("http://[::1]/fact", id="ipv6-loopback"),
        pytest.param("http://[fc00::1]/fact", id="ipv6-private"),
        pytest.param("http://[fe80::1]/fact", id="ipv6-link-local"),
        pytest.param("http://[ff00::1]/fact", id="ipv6-multicast"),
        pytest.param("http://[2001:db8::1]/fact", id="ipv6-reserved"),
        pytest.param("http://[::]/fact", id="ipv6-unspecified"),
        pytest.param("https://example.com/fact?token=value", id="query-token"),
        pytest.param("https://example.com/fact?api%5Fkey=value", id="encoded-query-key"),
        pytest.param(
            "https://example.com/fact?X-Amz-Signature=value",
            id="signature-variant",
        ),
        pytest.param("https://example.com/fact?apikey=value", id="compact-key-variant"),
        pytest.param(
            "https://example.com/fact?accessToken=value",
            id="camel-token-variant",
        ),
        pytest.param(
            "https://example.com/fact?clientSecret=value",
            id="camel-secret-variant",
        ),
        pytest.param(
            "https://example.com/fact?oauthSignature=value",
            id="camel-signature-variant",
        ),
        pytest.param("https://example.com/fact#access_token=value", id="fragment-token"),
        pytest.param("https://example.com/fact#auth%3Avalue", id="encoded-fragment-auth"),
    ],
)
def test_safe_source_url_rejects_nonpublic_or_credential_bearing_urls(url: str) -> None:
    assert _is_safe_source_url(url) is False
