import json
import secrets
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from gaia_agent.interactive.credential_vault import CredentialError


@dataclass(frozen=True, repr=False)
class EncryptedCredential:
    version: str
    nonce: bytes
    ciphertext: bytes


class KeyCipher:
    def __init__(self, keyring, active_version):
        if not isinstance(keyring, dict) or active_version not in keyring or not all(
            isinstance(version, str) and version and isinstance(key, bytes) and len(key) == 32
            for version, key in keyring.items()
        ):
            raise CredentialError("credential_configuration", 503)
        self._keys = dict(keyring)
        self.active_version = active_version

    @staticmethod
    def _aad(owner, provider, version):
        return json.dumps(["toy-agent-credential-v1", owner, provider, version], separators=(",", ":"), ensure_ascii=False).encode()

    def encrypt(self, owner, provider, plaintext: SecretStr) -> EncryptedCredential:
        nonce = secrets.token_bytes(12)
        version = self.active_version
        ciphertext = AESGCM(self._keys[version]).encrypt(nonce, plaintext.get_secret_value().encode(), self._aad(owner, provider, version))
        return EncryptedCredential(version, nonce, ciphertext)

    def decrypt(self, owner, provider, envelope) -> SecretStr:
        try:
            value = AESGCM(self._keys[envelope.version]).decrypt(envelope.nonce, envelope.ciphertext, self._aad(owner, provider, envelope.version))
            return SecretStr(value.decode())
        except (InvalidTag, ValueError, KeyError, TypeError, UnicodeDecodeError):
            raise CredentialError("credential_integrity", 422) from None
