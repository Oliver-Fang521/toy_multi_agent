"""Public execution: explicit per-user clients, never a process credential fallback."""

import asyncio
from urllib.parse import urlsplit

import httpx
from openai import APIStatusError, AsyncOpenAI, AuthenticationError, PermissionDeniedError
from pydantic import BaseModel, ConfigDict, field_validator

from gaia_agent.interactive.composition import execute_interactive
from gaia_agent.interactive.credential_vault import CredentialError
from gaia_agent.interactive.models import InteractiveConfig
from gaia_agent.mcp_client import MCPToolClient
from gaia_agent.providers import OpenAIProvider
from gaia_agent.reliability import RetryPolicy


class InteractiveRuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_url: str

    @field_validator("base_url")
    @classmethod
    def https_endpoint(cls, value):
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or any(c.isspace() for c in value)
        ):
            raise ValueError("invalid_model_endpoint")
        return value.rstrip("/")

    @property
    def provider_hostname(self):
        return urlsplit(self.base_url).hostname


def provider_error_code(error):
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return "provider_authentication"
    if isinstance(error, APIStatusError) and error.status_code in {400, 404, 422}:
        return "provider_model_unavailable"
    return "provider_unavailable"


class SafePublicProvider:
    def __init__(self, provider, secret_values=()):
        self.provider = provider
        self.supports_vision = provider.supports_vision
        self._secrets = tuple(value for value in secret_values if value)
        self.failure_code = None

    def _sanitize(self, value):
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self._sanitize(k): self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(v) for v in value]
        return value

    async def complete(self, request, executor=None, *, usage_observer=None):
        try:
            result = await self.provider.complete(request, executor, usage_observer=usage_observer)
            return type(result).model_validate(self._sanitize(result.model_dump()))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - raw provider exceptions never reach notebooks
            self.failure_code = provider_error_code(error)
            raise CredentialError(self.failure_code, 502) from None


class UserRuntime:
    def __init__(self, settings: InteractiveRuntimeSettings, sessions):
        self.settings = settings
        self.sessions = sessions

    async def execute(self, ref, request, workspace, history, on_progress, attachment_path):
        material = self.sessions.resolve(ref)
        runtime_env = {
            "OPENAI_API_KEY": material.model.get_secret_value(),
            "OPENAI_BASE_URL": self.settings.base_url,
        }
        for provider, name in (
            (material.serper, "SERPER_API_KEY"),
            (material.serpapi, "SERPAPI_API_KEY"),
        ):
            if provider is not None:
                runtime_env[name] = provider.get_secret_value()
        config = InteractiveConfig(openai_api_key=material.model, output_dir=workspace)
        client = None
        transport = httpx.AsyncClient(follow_redirects=False, trust_env=False)
        try:
            client = AsyncOpenAI(
                api_key=material.model.get_secret_value(),
                base_url=self.settings.base_url,
                organization="",
                project="",
                max_retries=0,
                http_client=transport,
            )
            provider = SafePublicProvider(
                OpenAIProvider(
                    client,
                    model=config.model,
                    retry_policy=RetryPolicy(max_attempts=config.provider_max_attempts),
                ),
                [value for key, value in runtime_env.items() if key.endswith("API_KEY")],
            )

            def tools_factory(task):
                return MCPToolClient(
                    config=config,
                    task_root=workspace,
                    runtime_env=runtime_env,
                    attachments=(task.attachment_path,) if task.attachment_path else (),
                )

            async with asyncio.timeout(500):
                try:
                    return await execute_interactive(
                        request,
                        config,
                        workspace,
                        history,
                        on_progress,
                        provider=provider,
                        tools_factory=tools_factory,
                        attachment_path=attachment_path,
                    )
                except Exception:
                    if provider.failure_code:
                        raise CredentialError(provider.failure_code, 502) from None
                    raise
        except (asyncio.CancelledError, TimeoutError, CredentialError):
            raise
        except Exception:  # noqa: BLE001 - never expose original exceptions or nested payloads
            raise CredentialError("execution_failed", 502) from None
        finally:
            try:
                if client is not None:
                    await client.close()
            finally:
                await transport.aclose()
