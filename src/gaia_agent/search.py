"""Bounded, provider-neutral public web search adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from gaia_agent.providers.base import ToolCallError
from gaia_agent.reliability import RetryPolicy, is_transient_provider_error, retry_delay


class SearchProvider(Protocol):
    provider: str

    async def search(self, query: str) -> dict[str, Any]: ...


class SearchProviderError(ToolCallError):
    def __init__(
        self,
        *,
        provider: str,
        category: str,
        retryable: bool,
        http_status: int | None = None,
        attempt_count: int = 1,
    ) -> None:
        super().__init__(category)
        self.provider = provider
        self.category = category
        self.retryable = retryable
        self.http_status = http_status
        self.attempt_count = attempt_count


class SearchUnavailableError(ToolCallError):
    def __init__(self, attempt_count: int) -> None:
        super().__init__("search_unavailable")
        self.provider = None
        self.category = "search_unavailable"
        self.retryable = True
        self.http_status = None
        self.attempt_count = max(1, attempt_count)


def _source(entry: Any) -> dict[str, str | None] | None:
    if not isinstance(entry, dict):
        return None
    url = entry.get("link") or entry.get("url")
    if not isinstance(url, str) or not url:
        return None
    title = entry.get("title")
    return {"url": url, "title": title if isinstance(title, str) else None}


def _answer(payload: dict[str, Any], key: str) -> str:
    box = payload.get(key)
    if not isinstance(box, dict):
        return ""
    for field in ("answer", "snippet", "result", "title"):
        value = box.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _normalize(payload: Any, *, provider: str, answer_key: str, results_key: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SearchProviderError(provider=provider, category="parse_failure", retryable=False)
    raw_results = payload.get(results_key, [])
    if not isinstance(raw_results, list):
        raise SearchProviderError(provider=provider, category="parse_failure", retryable=False)
    sources = [source for item in raw_results if (source := _source(item)) is not None]
    answer = _answer(payload, answer_key)
    if not answer and not sources:
        raise SearchProviderError(provider=provider, category="empty_content", retryable=False)
    return {"answer": answer, "sources": sources}


def _http_error(provider: str, status: int) -> SearchProviderError:
    return SearchProviderError(
        provider=provider,
        category="http_status",
        retryable=status == 408 or status == 429 or status >= 500,
        http_status=status,
    )


class SerperSearchProvider:
    provider = "serper"

    def __init__(self, api_key: str, client: httpx.AsyncClient, *, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._client = client
        self._timeout_seconds = timeout_seconds

    async def search(self, query: str) -> dict[str, Any]:
        try:
            response = await self._client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self._api_key},
                json={"q": query},
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException:
            raise SearchProviderError(
                provider=self.provider, category="timeout", retryable=True
            ) from None
        except httpx.TransportError:
            raise SearchProviderError(
                provider=self.provider, category="transport", retryable=True
            ) from None
        if response.status_code >= 400:
            raise _http_error(self.provider, response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise SearchProviderError(
                provider=self.provider, category="parse_failure", retryable=False
            ) from None
        return _normalize(payload, provider=self.provider, answer_key="answerBox", results_key="organic")


class SerpAPISearchProvider:
    provider = "serpapi"

    def __init__(self, api_key: str, client: httpx.AsyncClient, *, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._client = client
        self._timeout_seconds = timeout_seconds

    async def search(self, query: str) -> dict[str, Any]:
        try:
            response = await self._client.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": self._api_key},
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException:
            raise SearchProviderError(
                provider=self.provider, category="timeout", retryable=True
            ) from None
        except httpx.TransportError:
            raise SearchProviderError(
                provider=self.provider, category="transport", retryable=True
            ) from None
        if response.status_code >= 400:
            raise _http_error(self.provider, response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise SearchProviderError(
                provider=self.provider, category="parse_failure", retryable=False
            ) from None
        return _normalize(
            payload, provider=self.provider, answer_key="answer_box", results_key="organic_results"
        )


class OpenAISearchProvider:
    """Classify hosted OpenAI search failures before chain retry decisions."""

    provider = "openai"

    def __init__(self, backend: SearchProvider) -> None:
        self._backend = backend

    async def search(self, query: str) -> dict[str, Any]:
        try:
            return await self._backend.search(query)
        except SearchProviderError:
            raise
        except Exception as error:  # noqa: BLE001 - normalize opaque SDK failures at this boundary.
            status = getattr(error, "status_code", None)
            if isinstance(status, int):
                raise _http_error(self.provider, status) from None
            if is_transient_provider_error(error):
                category = "timeout" if "Timeout" in type(error).__name__ else "transport"
                raise SearchProviderError(
                    provider=self.provider, category=category, retryable=True
                ) from None
            raise SearchProviderError(
                provider=self.provider, category="provider_error", retryable=False
            ) from None


@dataclass
class _Circuit:
    failures: int = 0
    open_until: float = 0.0


class SearchProviderChain:
    """Retries each configured provider before falling through to the next one."""

    def __init__(
        self,
        providers: Sequence[SearchProvider],
        retry_policy: RetryPolicy,
        clock: Callable[[], float],
        *,
        circuit_failure_threshold: int = 2,
        circuit_cooldown_seconds: float = 30.0,
    ) -> None:
        self._providers = tuple(providers)
        self._retry_policy = retry_policy
        self._clock = clock
        self._threshold = circuit_failure_threshold
        self._cooldown = circuit_cooldown_seconds
        self._circuits = {provider.provider: _Circuit() for provider in self._providers}

    async def search(self, query: str) -> dict[str, Any]:
        total_attempts = 0
        for provider in self._providers:
            circuit = self._circuits[provider.provider]
            if circuit.open_until > self._clock():
                continue
            for attempt in range(1, self._retry_policy.max_attempts + 1):
                total_attempts += 1
                try:
                    result = await provider.search(query)
                except SearchProviderError as error:
                    if not error.retryable:
                        raise
                    if attempt < self._retry_policy.max_attempts:
                        await asyncio.sleep(retry_delay(self._retry_policy, attempt, lambda: 0.0))
                        continue
                    circuit.failures += 1
                    if circuit.failures >= self._threshold:
                        circuit.open_until = self._clock() + self._cooldown
                    break
                else:
                    circuit.failures = 0
                    circuit.open_until = 0.0
                    return result
        raise SearchUnavailableError(total_attempts)
