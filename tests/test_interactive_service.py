import asyncio

import pytest

from gaia_agent.interactive.models import InteractiveRequest, InteractiveResult
from gaia_agent.interactive.service import RunService
from gaia_agent.interactive.store import ConversationStore, ServiceError
from gaia_agent.models import Usage


def test_store_denies_cross_user_and_persists(tmp_path):
    store = ConversationStore(tmp_path / "db.sqlite")
    conversation = store.create("alice")
    with pytest.raises(ServiceError):
        store.get("bob", conversation["id"])
    assert (
        ConversationStore(tmp_path / "db.sqlite").get("alice", conversation["id"])["id"]
        == conversation["id"]
    )


@pytest.mark.asyncio
async def test_duplicate_submission_does_not_execute_twice(tmp_path):
    store = ConversationStore(tmp_path / "db.sqlite")
    calls = []

    async def execute(request, workspace, history, on_progress, attachment_path):
        calls.append(request.question)
        return InteractiveResult(answer="answer", evidence=[], usage=Usage())

    service = RunService(store, tmp_path / "runs", execute)
    cid = store.create("alice")["id"]
    request = InteractiveRequest(question="question")
    first = service.start("alice", cid, request, "key")
    second = service.start("alice", cid, request, "key")
    assert first["id"] == second["id"]
    await service.tasks[first["id"]]
    assert calls == ["question"]
    assert service.get("alice", first["id"])["status"] == "completed"
    with pytest.raises(ServiceError):
        service.get("bob", first["id"])


@pytest.mark.asyncio
async def test_active_limit_cancellation_and_hourly_quota(tmp_path):
    store = ConversationStore(tmp_path / "db.sqlite")
    gate = asyncio.Event()

    async def execute(*args):
        await gate.wait()

    service = RunService(store, tmp_path / "runs", execute)
    cid = store.create("alice")["id"]
    req = InteractiveRequest(question="q")
    first = service.start("alice", cid, req, "1")
    with pytest.raises(ServiceError) as active:
        service.start("alice", cid, req, "2")
    assert active.value.status_code == 429
    await service.cancel("alice", first["id"])
    for key in ("2", "3"):
        run = service.start("alice", cid, req, key)
        await service.cancel("alice", run["id"])
    with pytest.raises(ServiceError) as quota:
        service.start("alice", cid, req, "4")
    assert quota.value.status_code == 429


def test_restart_marks_unfinished_runs_interrupted(tmp_path):
    store = ConversationStore(tmp_path / "db.sqlite")
    cid = store.create("alice")["id"]
    run = store.reserve("alice", cid, InteractiveRequest(question="q"), "key")
    store.interrupt_unfinished()
    assert store.run("alice", run["id"])["status"] == "interrupted"
