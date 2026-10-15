import base64
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from gaia_agent.interactive.api import create_app
from gaia_agent.interactive.credential_crypto import cipher_from_environment
from gaia_agent.interactive.credential_vault import CredentialError


def infrastructure():
    return {
        "AGENT_SERVICE_TOKEN": "synthetic-infrastructure-token-17482",
        "AGENT_CREDENTIAL_KEYRING": json.dumps({"v1": base64.b64encode(b"a" * 32).decode()}),
        "AGENT_CREDENTIAL_ACTIVE_VERSION": "v1",
        "AGENT_MODEL_BASE_URL": "https://models.example/v1",
    }


def test_deployment_payload_accepts_only_infrastructure_and_never_reads_env_file(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "deploy/configure_runtime.py"
    spec = importlib.util.spec_from_file_location("configure_runtime", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(__import__("sys").modules, "modal", Mock())
    spec.loader.exec_module(module)
    payload = infrastructure()
    assert module.infrastructure_values(payload) == payload
    for extra in ("OPENAI_API_KEY", "SERPER_API_KEY", "SERPAPI_API_KEY", "env_path"):
        with pytest.raises(CredentialError):
            module.infrastructure_values({**payload, extra: "synthetic-owner-private-78231"})


@pytest.mark.parametrize("keyring", [None, "invalid-json", "{}", '{"v1":"invalid-base64"}', "[]"])
def test_missing_or_malformed_encryption_keys_fail_closed(monkeypatch, keyring):
    monkeypatch.setenv("AGENT_CREDENTIAL_ACTIVE_VERSION", "v1")
    monkeypatch.delenv("AGENT_CREDENTIAL_KEYRING", raising=False)
    if keyring is not None:
        monkeypatch.setenv("AGENT_CREDENTIAL_KEYRING", keyring)
    with pytest.raises(CredentialError, match="credential_configuration"):
        cipher_from_environment()


def test_default_public_runtime_restart_ignores_owner_keys(tmp_path, monkeypatch, caplog):
    for name, value in infrastructure().items():
        monkeypatch.setenv(name, value)
    for name in ("OPENAI_API_KEY", "SERPER_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.setenv(name, "synthetic-owner-private-78231")
    headers = {
        "authorization": "Bearer synthetic-infrastructure-token-17482",
        "x-agent-user": "alice",
    }
    secret = "synthetic-restart-private-871932"
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.post(
                "/v1/credentials",
                headers=headers,
                json={"provider": "model", "key": secret, "consent": True},
            ).status_code
            == 200
        )
    with TestClient(create_app(tmp_path)) as client:
        metadata = client.get("/v1/credentials", headers=headers).json()
        assert metadata["statuses"][0]["configured"]
        sessions = client.app.state.credential_sessions
        assert (
            sessions.resolve(sessions.reference("alice", None)).model.get_secret_value() == secret
        )
        with pytest.raises(CredentialError):
            sessions.reference("bob", None)
        assert secret not in str(metadata)
    assert secret not in caplog.text
    for path in tmp_path.iterdir():
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
