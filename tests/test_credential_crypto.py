from dataclasses import replace

import pytest
from pydantic import SecretStr

from gaia_agent.interactive.credential_crypto import KeyCipher
from gaia_agent.interactive.credential_vault import CredentialError


def test_aead_random_nonce_and_identity_integrity():
    cipher = KeyCipher({"v1": b"a" * 32, "v2": b"b" * 32}, "v1")
    secret = SecretStr("synthetic-private-model-credential-17492")
    envelope = cipher.encrypt("alice", "model", secret)
    another = cipher.encrypt("alice", "model", secret)
    assert envelope.nonce != another.nonce
    assert envelope.ciphertext != another.ciphertext
    assert cipher.decrypt("alice", "model", envelope) == secret
    assert secret.get_secret_value() not in repr(envelope)
    for owner, provider, value in [
        ("bob", "model", envelope), ("alice", "serper", envelope),
        ("alice", "model", replace(envelope, version="v2")),
        ("alice", "model", replace(envelope, ciphertext=envelope.ciphertext[:-1] + bytes([envelope.ciphertext[-1] ^ 1]))),
    ]:
        with pytest.raises(CredentialError, match="credential_integrity"):
            cipher.decrypt(owner, provider, value)


@pytest.mark.parametrize("keys,active", [({}, "v1"), ({"v1": b"a" * 31}, "v1"), ({"v1": b"a" * 32}, "missing")])
def test_master_keys_fail_closed(keys, active):
    with pytest.raises(CredentialError, match="credential_configuration"):
        KeyCipher(keys, active)
