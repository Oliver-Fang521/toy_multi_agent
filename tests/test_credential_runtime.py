import asyncio
import os

import pytest
from pydantic import SecretStr

from gaia_agent.interactive.api import create_app
from gaia_agent.interactive.credential_crypto import KeyCipher
from gaia_agent.interactive.credential_runtime import (
    InteractiveRuntimeSettings,
    SafePublicProvider,
    UserRuntime,
)
from gaia_agent.interactive.credential_sessions import CredentialSessions
from gaia_agent.interactive.credential_vault import (
    CredentialError,
    CredentialMaterial,
    CredentialVault,
)
from gaia_agent.interactive.models import InteractiveConfig, InteractiveRequest, InteractiveResult
from gaia_agent.interactive.service import RunService
from gaia_agent.interactive.store import ConversationStore
from gaia_agent.mcp_client import MCPToolClient
from gaia_agent.models import Usage


def setup_runtime(tmp_path, clock=lambda: 0):
    store = ConversationStore(tmp_path / "state.sqlite")
    vault = CredentialVault(store.db, KeyCipher({"v1": b"a" * 32}, "v1"), store.lock)
    return store, CredentialSessions(vault, clock)


def test_mcp_explicit_environment_excludes_owner_keys(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "SERPER_API_KEY", "SERPAPI_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(key, "synthetic-owner-secret-9812")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://owner.example/v1")
    before = dict(os.environ)
    env = {
        "OPENAI_API_KEY": "synthetic-alice-model-8134",
        "OPENAI_BASE_URL": "https://models.example/v1",
    }
    client = MCPToolClient(
        task_root=tmp_path,
        config=InteractiveConfig(openai_api_key=env["OPENAI_API_KEY"]),
        runtime_env=env,
    )
    child = client._minimal_server_env()
    assert child["OPENAI_API_KEY"] == env["OPENAI_API_KEY"]
    assert child["OPENAI_BASE_URL"] == "https://models.example/v1"
    assert not any(k in child for k in ("SERPER_API_KEY", "SERPAPI_API_KEY", "HF_TOKEN"))
    assert "synthetic-owner-secret-9812" not in str(child)
    assert client._redact(env["OPENAI_API_KEY"]) == "[REDACTED]"
    assert dict(os.environ) == before


def test_public_tool_redaction_covers_nested_dictionary_keys(tmp_path):
    secret = "synthetic-nested-private-274918"
    config = InteractiveConfig(openai_api_key=secret)
    client = MCPToolClient(
        task_root=tmp_path, config=config, runtime_env={"OPENAI_API_KEY": secret}
    )
    assert client._redact_value({"nested": [{secret: secret}]}) == {
        "nested": [{"[REDACTED]": "[REDACTED]"}]
    }


@pytest.mark.asyncio
async def test_runtime_concurrent_parent_and_child_user_isolation(tmp_path, monkeypatch):
    import gaia_agent.interactive.credential_runtime as module

    _store, sessions = setup_runtime(tmp_path)
    settings = InteractiveRuntimeSettings(base_url="https://models.example/v1")
    clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            clients.append(self)

        def with_options(self, **kwargs):
            return self

        async def close(self):
            self.closed = True

    observed = []

    async def execute(request, config, workspace, history, on_progress, **kwargs):
        task = type("Task", (), {"attachment_path": None})()
        child = kwargs["tools_factory"](task)._minimal_server_env()
        observed.append((config.openai_api_key.get_secret_value(), child))
        await asyncio.sleep(0)
        return InteractiveResult(answer="ok", evidence=[], usage=Usage())

    monkeypatch.setattr(module, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(module, "execute_interactive", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-owner-private-9123")
    before = dict(os.environ)
    refs = []
    for owner in ("alice", "bob"):
        material = CredentialMaterial(
            SecretStr(f"synthetic-{owner}-private-4471"),
            SecretStr(f"synthetic-{owner}-search-3871") if owner == "alice" else None,
        )
        refs.append(sessions.reference(owner, sessions.issue(owner, material)))
    runtime = UserRuntime(settings, sessions)
    await asyncio.gather(
        *(
            runtime.execute(
                ref,
                InteractiveRequest(question="q"),
                tmp_path / ref.owner,
                [],
                lambda *args: None,
                None,
            )
            for ref in refs
        )
    )
    assert {c.options["api_key"] for c in clients} == {
        "synthetic-alice-private-4471",
        "synthetic-bob-private-4471",
    }
    assert all(c.options["base_url"] == "https://models.example/v1" and c.closed for c in clients)
    for key, child in observed:
        assert child["OPENAI_API_KEY"] == key
        assert "synthetic-owner-private-9123" not in str(child)
        assert ("SERPER_API_KEY" in child) == ("alice" in key)
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_queued_expiry_and_idempotent_completed_retry_without_credentials(tmp_path):
    now = [0]
    store, sessions = setup_runtime(tmp_path, lambda: now[0])
    calls = []

    async def execute(ref, *args):
        sessions.resolve(ref)
        calls.append(ref.owner)
        return InteractiveResult(answer="paid answer", evidence=[], usage=Usage())

    service = RunService(store, tmp_path / "runs", execute, credential_sessions=sessions)
    cid = store.create("alice")["id"]
    req = InteractiveRequest(question="q")
    handle = sessions.issue("alice", CredentialMaterial(SecretStr("synthetic-queue-private-1239")))
    await service.slots.acquire()
    await service.slots.acquire()
    run = service.start(
        "alice", cid, req, "expired", credential_ref=sessions.reference("alice", handle)
    )
    now[0] = 1800
    service.slots.release()
    await service.tasks[run["id"]]
    assert not calls
    assert service.get("alice", run["id"])["error"] == "credential_expired"
    handle = sessions.issue("alice", CredentialMaterial(SecretStr("synthetic-queue-private-1239")))
    run = service.start(
        "alice", cid, req, "completed", credential_ref=sessions.reference("alice", handle)
    )
    await service.tasks[run["id"]]
    sessions.invalidate("alice")
    retry = service.start("alice", cid, req, "completed")
    assert retry["result"]["answer"] == "paid answer" and calls == ["alice"]
    with pytest.raises(CredentialError):
        service.start("alice", cid, req, "missing")
    assert store.db.execute("SELECT count(*) FROM runs").fetchone()[0] == 2
    await service.shutdown()


@pytest.mark.asyncio
async def test_provider_errors_are_safe_before_notebook_or_orchestrator(caplog):
    class UnsafeProvider:
        supports_vision = True

        async def complete(self, *args, **kwargs):
            raise RuntimeError({"nested": "synthetic-upstream-private-7318"})

    provider = SafePublicProvider(UnsafeProvider())
    with pytest.raises(CredentialError) as error:
        await provider.complete(None)
    assert str(error.value) == "provider_unavailable"
    assert "synthetic-upstream-private-7318" not in caplog.text


def test_public_startup_without_owner_model_key_and_no_key_run_rejected(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(
        tmp_path,
        "test-service-token",
        cipher=KeyCipher({"v1": b"a" * 32}, "v1"),
        runtime_settings=InteractiveRuntimeSettings(base_url="https://models.example/v1"),
    )
    with TestClient(app) as client:
        headers = {
            "authorization": "Bearer test-service-token",
            "x-agent-user": "alice",
            "idempotency-key": "new",
        }
        cid = client.post("/v1/conversations", headers=headers).json()["id"]
        response = client.post(
            f"/v1/conversations/{cid}/runs", headers=headers, json={"question": "q"}
        )
        assert response.status_code == 422 and response.json()["error"] == "credential_required"
        assert (
            client.get("/v1/credentials", headers=headers).json()["provider_hostname"]
            == "models.example"
        )


@pytest.mark.asyncio
async def test_delete_cancels_active_and_queued_credentials_without_new_calls(tmp_path):
    store, sessions = setup_runtime(tmp_path)
    started = asyncio.Event()
    calls = []

    async def execute(ref, *args):
        calls.append(ref.owner)
        started.set()
        await asyncio.Event().wait()

    service = RunService(store, tmp_path / "runs", execute, credential_sessions=sessions)
    req = InteractiveRequest(question="q")
    cid = store.create("alice")["id"]
    handle = sessions.issue("alice", CredentialMaterial(SecretStr("synthetic-cancel-private-7481")))
    ref = sessions.reference("alice", handle)
    run = service.start("alice", cid, req, "first", credential_ref=ref)
    await started.wait()
    sessions.invalidate("alice")
    await service.cancel_owner("alice")
    assert service.get("alice", run["id"])["status"] == "cancelled"
    await service.slots.acquire()
    await service.slots.acquire()
    handle = sessions.issue("alice", CredentialMaterial(SecretStr("synthetic-cancel-private-7481")))
    run = service.start(
        "alice", cid, req, "second", credential_ref=sessions.reference("alice", handle)
    )
    sessions.invalidate("alice")
    await service.cancel_owner("alice")
    service.slots.release()
    assert service.get("alice", run["id"])["status"] == "cancelled" and calls == ["alice"]
    await service.shutdown()
