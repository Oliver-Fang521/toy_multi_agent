import pytest
from fastapi.testclient import TestClient

from gaia_agent.interactive.api import RequestBoundary, create_app


@pytest.mark.asyncio
async def test_unauthorized_upload_never_reads_or_parses_body():
    async def forbidden(*args):
        raise AssertionError("unauthorized body consumed")

    responses = []

    async def send(message):
        responses.append(message)

    await RequestBoundary(forbidden, "test-service-token")(
        {"type": "http", "path": "/v1/conversations/x/uploads", "method": "POST", "headers": []},
        forbidden, send,
    )
    assert responses[0]["status"] == 401


def test_upload_rejects_multiple_files_and_oversized_chunked_body(tmp_path):
    with TestClient(create_app(tmp_path, "test-service-token", execute=lambda *args: None)) as client:
        headers = {"authorization": "Bearer test-service-token", "x-agent-user": "alice"}
        cid = client.post("/v1/conversations", headers=headers).json()["id"]
        path = f"/v1/conversations/{cid}/uploads"
        assert client.post(path, headers=headers, files=[
            ("file", ("a.txt", b"one", "text/plain")),
            ("file", ("b.txt", b"two", "text/plain")),
        ]).status_code == 422
        assert client.post(path, headers=headers, content=iter([b"x" * (34 * 1024 * 1024)] )).status_code == 413


def test_conversation_preserves_all_answers_and_evidence(tmp_path):
    with TestClient(create_app(tmp_path, "test-service-token", execute=lambda *args: None)) as client:
        headers = {"authorization": "Bearer test-service-token", "x-agent-user": "alice"}
        cid = client.post("/v1/conversations", headers=headers).json()["id"]
        store = client.app.state.service.store
        for i in range(6):
            from gaia_agent.interactive.models import InteractiveRequest
            run = store.reserve("alice", cid, InteractiveRequest(question=f"q{i}"), str(i))
            store.update(run["id"], status="completed", result={"answer": f"a{i}", "evidence": [{"summary": "source"}]})
            store.db.execute("UPDATE runs SET created=0 WHERE id=?", (run["id"],))
            store.db.commit()
        data = client.get(f"/v1/conversations/{cid}", headers=headers).json()
        assert len(data["messages"]) == 12
        assert data["messages"][1]["evidence"] == [{"summary": "source"}]


def test_api_authentication_ownership_and_upload_validation(tmp_path):
    with TestClient(
        create_app(tmp_path, "test-service-token", execute=lambda *args: None)
    ) as client:
        assert client.post("/v1/conversations").status_code == 401
        headers = {"authorization": "Bearer test-service-token", "x-agent-user": "alice"}
        conversation = client.post("/v1/conversations", headers=headers).json()
        identifier = conversation["id"]
        bob = {**headers, "x-agent-user": "bob"}
        assert client.get(f"/v1/conversations/{identifier}", headers=bob).status_code == 404
        assert (
            client.post(
                f"/v1/conversations/{identifier}/uploads",
                headers=headers,
                files={"file": ("payload.exe", b"evil", "application/octet-stream")},
            ).status_code
            == 415
        )
        assert (
            client.post(
                f"/v1/conversations/{identifier}/uploads",
                headers=headers,
                files={"file": ("note.txt", b"hello", "text/plain")},
            ).status_code
            == 200
        )
        assert (
            client.post(
                f"/v1/conversations/{identifier}/runs",
                headers={**headers, "idempotency-key": "x"},
                json={"question": "   "},
            ).status_code
            == 422
        )
