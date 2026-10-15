"""Owner-scoped encrypted storage. This module never consults process credentials."""
import sqlite3
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import SecretStr

ProviderName = Literal["model", "serper", "serpapi"]
PROVIDERS = ("model", "serper", "serpapi")
SAFE_CODES = frozenset({"credential_configuration", "credential_integrity", "credential_storage",
    "credential_required", "consent_required", "invalid_credential", "credential_expired",
    "credential_revoked", "credential_sessions_busy", "provider_authentication",
    "provider_unavailable", "provider_model_unavailable", "execution_failed"})


class CredentialError(RuntimeError):
    def __init__(self, code: str, status_code: int = 400):
        self.code = code if code in SAFE_CODES else "execution_failed"
        self.status_code = status_code
        super().__init__(self.code)


@dataclass(frozen=True)
class CredentialStatus:
    provider: ProviderName
    configured: bool
    last_four: str | None = None
    updated_at: float | None = None
    consent_at: float | None = None


@dataclass(frozen=True, repr=False)
class CredentialMaterial:
    model: SecretStr
    serper: SecretStr | None = None
    serpapi: SecretStr | None = None


@dataclass(frozen=True, repr=False)
class CredentialRef:
    owner: str
    generation: int
    handle: str | None = None


def validate_secret(secret: SecretStr) -> None:
    value = secret.get_secret_value()
    if not value.strip() or len(value) > 4096 or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise CredentialError("invalid_credential", 422)


class CredentialVault:
    def __init__(self, connection, cipher, lock):
        self.db = connection
        self.cipher = cipher
        self.lock = lock
        with self.lock, self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS credentials(
                owner TEXT, provider TEXT, version TEXT, nonce BLOB, ciphertext BLOB,
                last_four TEXT, updated_at REAL, consent_at REAL, PRIMARY KEY(owner,provider))""")

    def put(self, owner, provider, secret: SecretStr, *, consent: bool) -> CredentialStatus:
        if consent is not True:
            raise CredentialError("consent_required", 422)
        if provider not in PROVIDERS:
            raise CredentialError("invalid_credential", 422)
        validate_secret(secret)
        envelope = self.cipher.encrypt(owner, provider, secret)
        now = time.time()
        try:
            with self.lock, self.db:
                self.db.execute("""INSERT INTO credentials VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(owner,provider) DO UPDATE SET version=excluded.version,
                    nonce=excluded.nonce,ciphertext=excluded.ciphertext,last_four=excluded.last_four,
                    updated_at=excluded.updated_at,consent_at=excluded.consent_at""",
                    (owner, provider, envelope.version, envelope.nonce, envelope.ciphertext,
                     secret.get_secret_value()[-4:], now, now))
        except sqlite3.Error:
            raise CredentialError("credential_storage", 503) from None
        return CredentialStatus(provider, True, secret.get_secret_value()[-4:], now, now)

    def status(self, owner) -> list[CredentialStatus]:
        try:
            with self.lock:
                rows = self.db.execute("SELECT provider,last_four,updated_at,consent_at FROM credentials WHERE owner=?", (owner,)).fetchall()
            found = {row[0]: CredentialStatus(row[0], True, row[1], row[2], row[3]) for row in rows}
            return [found.get(p, CredentialStatus(p, False)) for p in PROVIDERS]
        except sqlite3.Error:
            raise CredentialError("credential_storage", 503) from None

    def read(self, owner, provider) -> SecretStr:
        from gaia_agent.interactive.credential_crypto import EncryptedCredential

        try:
            with self.lock:
                row = self.db.execute("SELECT version,nonce,ciphertext FROM credentials WHERE owner=? AND provider=?", (owner, provider)).fetchone()
        except sqlite3.Error:
            raise CredentialError("credential_storage", 503) from None
        if row is None:
            raise CredentialError("credential_required", 422)
        return self.cipher.decrypt(owner, provider, EncryptedCredential(*row))

    def delete(self, owner, provider) -> None:
        try:
            with self.lock, self.db:
                self.db.execute("DELETE FROM credentials WHERE owner=? AND provider=?", (owner, provider))
        except sqlite3.Error:
            raise CredentialError("credential_storage", 503) from None

    def rotate_to_active(self, owner, provider) -> None:
        with self.lock, self.db:
            secret = self.read(owner, provider)
            envelope = self.cipher.encrypt(owner, provider, secret)
            self.db.execute("UPDATE credentials SET version=?,nonce=?,ciphertext=? WHERE owner=? AND provider=?",
                (envelope.version, envelope.nonce, envelope.ciphertext, owner, provider))
