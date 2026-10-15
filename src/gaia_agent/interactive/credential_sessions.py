import secrets
import time
from dataclasses import dataclass

from gaia_agent.interactive.credential_vault import (
    CredentialError, CredentialMaterial, CredentialRef, validate_secret,
)


@dataclass(repr=False)
class _Session:
    handle: str
    expires_at: float
    material: CredentialMaterial


class CredentialSessions:
    """Event-loop-owned ephemeral references; never serialize this object."""

    def __init__(self, vault, clock=time.time):
        self.vault = vault
        self.clock = clock
        self._sessions: dict[str, _Session] = {}
        self._generations: dict[str, int] = {}

    def issue(self, owner, material: CredentialMaterial) -> str:
        for value in (material.model, material.serper, material.serpapi):
            if value is not None:
                validate_secret(value)
        now = self.clock()
        self._sessions = {o: s for o, s in self._sessions.items() if s.expires_at > now}
        if owner not in self._sessions and len(self._sessions) >= 100:
            raise CredentialError("credential_sessions_busy", 503)
        self.invalidate(owner)
        handle = secrets.token_urlsafe(32)
        self._sessions[owner] = _Session(handle, now + 1800, material)
        return handle

    def reference(self, owner, handle: str | None) -> CredentialRef:
        ref = CredentialRef(owner, self._generations.get(owner, 0), handle)
        self.resolve(ref)
        return ref

    def resolve(self, ref: CredentialRef) -> CredentialMaterial:
        if ref.generation != self._generations.get(ref.owner, 0):
            raise CredentialError("credential_revoked", 422)
        if ref.handle is not None:
            entry = self._sessions.get(ref.owner)
            if entry is None or not secrets.compare_digest(ref.handle, entry.handle):
                raise CredentialError("credential_required", 422)
            if self.clock() >= entry.expires_at:
                self._sessions.pop(ref.owner, None)
                raise CredentialError("credential_expired", 422)
            return entry.material
        model = self.vault.read(ref.owner, "model")
        optional = {}
        for provider in ("serper", "serpapi"):
            try:
                optional[provider] = self.vault.read(ref.owner, provider)
            except CredentialError as error:
                if error.code != "credential_required":
                    raise
        return CredentialMaterial(model, **optional)

    def invalidate(self, owner) -> None:
        self._generations[owner] = self._generations.get(owner, 0) + 1
        self._sessions.pop(owner, None)

    def close(self) -> None:
        for owner in self._sessions:
            self._generations[owner] = self._generations.get(owner, 0) + 1
        self._sessions.clear()
