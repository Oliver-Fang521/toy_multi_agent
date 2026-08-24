"""Bounded, provider-neutral public web search adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
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
    def __init__(
        self,
        attempt_count: int,
        *,
        circuit_open_count: int = 0,
        fallback_attempted: bool = False,
    ) -> None:
        super().__init__("search_unavailable")
        self.provider = None
        self.category = "search_unavailable"
        self.retryable = True
        self.http_status = None
        self.attempt_count = max(1, attempt_count)
        self.circuit_open_count = max(0, circuit_open_count)
        self.fallback_attempted = fallback_attempted


class SearchDeadlineExceeded(RuntimeError):
    pass


class SearchProviderResult(dict[str, Any]):
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        provider: str,
        attempt_count: int,
        circuit_open_count: int,
        fallback_used: bool,
    ) -> None:
        super().__init__(payload)
        self.provider = provider
        self.attempt_count = attempt_count
        self.circuit_open_count = circuit_open_count
        self.fallback_used = fallback_used


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


def _validate_normalized_result(payload: Any, *, provider: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SearchProviderError(provider=provider, category="parse_failure", retryable=False)
    answer = payload.get("answer")
    sources = payload.get("sources")
    if not isinstance(answer, str) or not isinstance(sources, list):
        raise SearchProviderError(provider=provider, category="parse_failure", retryable=False)
    for source in sources:
        if not isinstance(source, dict):
            raise SearchProviderError(
                provider=provider, category="parse_failure", retryable=False
            )
        url = source.get("url")
        title = source.get("title")
        if (
            not isinstance(url, str)
            or not url
            or title is not None
            and not isinstance(title, str)
        ):
            raise SearchProviderError(
                provider=provider, category="parse_failure", retryable=False
            )
    if not answer and not sources:
        raise SearchProviderError(provider=provider, category="empty_content", retryable=False)
    return dict(payload)


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
            result = await self._backend.search(query)
        except SearchProviderError:
            raise
        except TimeoutError:
            raise SearchProviderError(
                provider=self.provider,
                category="timeout",
                retryable=True,
            ) from None
        except Exception as error:
            status = getattr(error, "status_code", None)
            if isinstance(status, int):
                raise _http_error(self.provider, status) from None
            if is_transient_provider_error(error):
                category = "timeout" if "Timeout" in type(error).__name__ else "transport"
                raise SearchProviderError(
                    provider=self.provider, category=category, retryable=True
                ) from None
            raise
        return _validate_normalized_result(result, provider=self.provider)


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
        operation_timeout_seconds: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._providers = tuple(providers)
        self._retry_policy = retry_policy
        self._clock = clock
        self._threshold = circuit_failure_threshold
        self._cooldown = circuit_cooldown_seconds
        self._operation_timeout_seconds = operation_timeout_seconds
        self._sleep = sleep
        self._circuits = {provider.provider: _Circuit() for provider in self._providers}

    def _can_wait(self, deadline: float | None, delay: float) -> bool:
        return deadline is None or deadline - self._clock() > delay

    async def _search_with_deadline(
        self,
        provider: SearchProvider,
        query: str,
        deadline: float | None,
    ) -> dict[str, Any]:
        if deadline is None:
            return await provider.search(query)
        remaining_seconds = deadline - self._clock()
        if remaining_seconds <= 0:
            raise SearchDeadlineExceeded
        timeout = asyncio.timeout(remaining_seconds)
        try:
            async with timeout:
                result = await provider.search(query)
        except TimeoutError:
            if timeout.expired():
                raise SearchDeadlineExceeded from None
            raise
        if timeout.expired():
            raise SearchDeadlineExceeded
        return result

    async def search(self, query: str) -> dict[str, Any]:
        deadline = (
            None
            if self._operation_timeout_seconds is None
            else self._clock() + self._operation_timeout_seconds
        )
        total_attempts = 0
        circuit_open_count = 0
        fallback_attempted = False
        for provider_index, provider in enumerate(self._providers):
            if not self._can_wait(deadline, 0.0):
                raise SearchDeadlineExceeded
            circuit = self._circuits[provider.provider]
            if circuit.open_until > self._clock():
                circuit_open_count += 1
                continue
            if provider_index > 0:
                fallback_attempted = True
            for attempt in range(1, self._retry_policy.max_attempts + 1):
                if not self._can_wait(deadline, 0.0):
                    raise SearchDeadlineExceeded
                total_attempts += 1
                try:
                    result = await self._search_with_deadline(provider, query, deadline)
                except SearchProviderError as error:
                    if not error.retryable:
                        circuit.failures = 0
                        circuit.open_until = 0.0
                        break
                    if attempt < self._retry_policy.max_attempts:
                        delay = retry_delay(self._retry_policy, attempt, lambda: 0.0)
                        if not self._can_wait(deadline, delay):
                            raise SearchDeadlineExceeded
                        await self._sleep(delay)
                        continue
                    circuit.failures += 1
                    if circuit.failures >= self._threshold:
                        circuit.open_until = self._clock() + self._cooldown
                    break
                else:
                    circuit.failures = 0
                    circuit.open_until = 0.0
                    return SearchProviderResult(
                        result,
                        provider=provider.provider,
                        attempt_count=total_attempts,
                        circuit_open_count=circuit_open_count,
                        fallback_used=provider_index > 0,
                    )
        raise SearchUnavailableError(
            total_attempts,
            circuit_open_count=circuit_open_count,
            fallback_attempted=fallback_attempted,
        )
