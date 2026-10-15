import pytest
from fastapi.testclient import TestClient

from gaia_agent.interactive.api import create_app
from gaia_agent.interactive.credential_crypto import KeyCipher
from gaia_agent.interactive.credential_vault import CredentialError

HEADERS = {"authorization": "Bearer test-service-token", "x-agent-user": "alice"}
FAKE_KEY = "synthetic-api-private-key-347891"


@pytest.fixture
def client(tmp_path):
    app = create_app(
        tmp_path,
        "test-service-token",
        execute=lambda *args: None,
        cipher=KeyCipher({"v1": b"a" * 32}, "v1"),
        provider_hostname="models.example",
    )
    with TestClient(app) as client:
        yield client


def test_safe_status_consent_replace_delete_and_ownership(client, caplog):
    assert client.get("/v1/credentials").status_code == 401
    assert (
        client.get("/v1/credentials", headers=HEADERS).json()["provider_hostname"]
        == "models.example"
    )
    payload = {"provider": "model", "key": FAKE_KEY}
    assert client.post("/v1/credentials", headers=HEADERS, json=payload).status_code == 422
    payload["consent"] = True
    assert (
        client.post(
            "/v1/credentials", headers=HEADERS, json={**payload, "owner": "bob"}
        ).status_code
        == 422
    )
    saved = client.post("/v1/credentials", headers=HEADERS, json=payload)
    assert saved.status_code == 200 and FAKE_KEY not in saved.text
    statuses = client.get("/v1/credentials", headers=HEADERS).json()["statuses"]
    assert statuses[0]["last_four"] == "7891"
    assert not client.get("/v1/credentials", headers={**HEADERS, "x-agent-user": "bob"}).json()[
        "statuses"
    ][0]["configured"]
    deleted = client.post("/v1/credentials/delete", headers=HEADERS, json={"provider": "model"})
    assert deleted.status_code == 200
    assert not client.get("/v1/credentials", headers=HEADERS).json()["statuses"][0]["configured"]
    assert FAKE_KEY not in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        {"provider": "model", "key": {"nested": FAKE_KEY}, "consent": True},
        {"provider": "model", "key": FAKE_KEY + "\n", "consent": True},
        {"provider": "model", "key": FAKE_KEY, "consent": True, "base_url": "https://evil.example"},
    ],
)
def test_validation_never_echoes_secrets(client, body):
    response = client.post("/v1/credentials", headers=HEADERS, json=body)
    assert response.status_code == 422
    assert FAKE_KEY not in response.text


def test_temporary_handles_owner_bound_invalidated_on_saved_mutation(client):
    response = client.post("/v1/credential-sessions", headers=HEADERS, json={"model_key": FAKE_KEY})
    assert response.status_code == 200
    assert FAKE_KEY not in response.text
    sessions = client.app.state.credential_sessions
    handle = response.json()["handle"]
    with pytest.raises(CredentialError):
        sessions.reference("bob", handle)
    ref = sessions.reference("alice", handle)
    client.post(
        "/v1/credentials",
        headers=HEADERS,
        json={"provider": "model", "key": FAKE_KEY, "consent": True},
    )
    with pytest.raises(CredentialError, match="credential_revoked"):
        sessions.resolve(ref)


def test_body_limit_and_deletion_waits_for_cancellation(client):
    response = client.post("/v1/credential-sessions", headers=HEADERS, content=b"x" * 65537)
    assert response.status_code == 413
    cancelled = []

    async def cancel(owner):
        import asyncio

        await asyncio.sleep(0)
        cancelled.append(owner)

    client.app.state.service.cancel_owner = cancel
    assert (
        client.post(
            "/v1/credentials/delete", headers=HEADERS, json={"provider": "model"}
        ).status_code
        == 200
    )
    assert cancelled == ["alice"]


@pytest.mark.parametrize("consent", [1, "true", False])
def test_save_requires_literal_boolean_true_not_coercion(client, consent):
    response = client.post(
        "/v1/credentials",
        headers=HEADERS,
        json={"provider": "model", "key": FAKE_KEY, "consent": consent},
    )
    assert response.status_code == 422
    assert not client.get("/v1/credentials", headers=HEADERS).json()["statuses"][0]["configured"]
