from dataclasses import asdict
from typing import Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from gaia_agent.interactive.credential_vault import (
    CredentialError, CredentialMaterial, ProviderName, validate_secret,
)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class SaveCredential(_Body):
    provider: ProviderName
    key: SecretStr
    consent: Literal[True]

    @field_validator("key")
    @classmethod
    def valid_key(cls, value):
        validate_secret(value)
        return value


class DeleteCredential(_Body):
    provider: ProviderName


class TemporaryCredentials(_Body):
    model_key: SecretStr
    serper_key: SecretStr | None = None
    serpapi_key: SecretStr | None = None

    @field_validator("model_key", "serper_key", "serpapi_key")
    @classmethod
    def valid_key(cls, value):
        if value is not None:
            validate_secret(value)
        return value


def mount_credential_routes(app, *, owner_dependency, vault, sessions, cancel_owner, provider_hostname):
    @app.exception_handler(CredentialError)
    async def credential_error(request, error):
        return JSONResponse({"error": error.code}, status_code=error.status_code)

    @app.get("/v1/credentials")
    async def status(user=owner_dependency):
        return {"provider_hostname": provider_hostname, "statuses": [asdict(s) for s in vault.status(user)]}

    async def mutate(user, action):
        if user in sessions.mutating:
            raise CredentialError("credential_revoked", 409)
        sessions.mutating.add(user)
        try:
            sessions.invalidate(user)
            result = action()
            await cancel_owner(user)
            return result
        finally:
            sessions.mutating.discard(user)

    @app.post("/v1/credentials")
    async def save(body: SaveCredential, user=owner_dependency):
        return await mutate(user, lambda: asdict(vault.put(user, body.provider, body.key, consent=body.consent)))

    @app.post("/v1/credentials/delete")
    async def delete(body: DeleteCredential, user=owner_dependency):
        def action():
            vault.delete(user, body.provider)
            return {"deleted": True}
        return await mutate(user, action)

    @app.post("/v1/credential-sessions")
    async def temporary(body: TemporaryCredentials, user=owner_dependency):
        return await mutate(user, lambda: {"handle": sessions.issue(user,
            CredentialMaterial(body.model_key, body.serper_key, body.serpapi_key), _mutation=True), "expires_in": 1800})
