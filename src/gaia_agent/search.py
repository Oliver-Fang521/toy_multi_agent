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


class SearchDeadlineExceeded(ToolCallError):
    """Safe typed operation expiry, including bounded routing metadata."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        attempt_count: int = 1,
        circuit_open_count: int = 0,
        fallback_attempted: bool = False,
        fallback_used: bool = False,
    ) -> None:
        super().__init__("search_timeout")
        self.provider = provider
        self.category = "timeout"
        self.retryable = True
        self.http_status = None
        self.attempt_count = max(1, attempt_count)
        self.circuit_open_count = max(0, circuit_open_count)
        self.fallback_attempted = fallback_attempted
        self.fallback_used = fallback_used


class _AttemptDeadlineExceeded(RuntimeError):
    pass


class _OperationDeadlineExceeded(RuntimeError):
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
        category: str | None = None,
    ) -> None:
        super().__init__(payload)
        self.provider = provider
        self.attempt_count = attempt_count
        self.circuit_open_count = circuit_open_count
        self.fallback_used = fallback_used
        self.category = category


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
    """Retry providers while reserving one equal time slice per later fallback.

    Each current-provider retry and each remaining fallback provider owns one
    dynamic share of the operation time left at attempt entry. A retry can use
    only its share; downstream providers retain theirs regardless of provider
    count, while every call and backoff remains under the one operation cutoff.
    """

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
        attempt_deadline: float | None,
        operation_deadline: float | None,
    ) -> dict[str, Any]:
        if attempt_deadline is None:
            return await provider.search(query)
        remaining_seconds = attempt_deadline - self._clock()
        if remaining_seconds <= 0:
            if operation_deadline is not None and attempt_deadline >= operation_deadline:
                raise _OperationDeadlineExceeded
            raise _AttemptDeadlineExceeded
        timeout = asyncio.timeout(remaining_seconds)
        try:
            async with timeout:
                result = await provider.search(query)
        except TimeoutError:
            if timeout.expired():
                if operation_deadline is not None and attempt_deadline >= operation_deadline:
                    raise _OperationDeadlineExceeded from None
                raise _AttemptDeadlineExceeded from None
            raise
        except Exception:
            if timeout.expired() or self._clock() >= attempt_deadline:
                if (
                    operation_deadline is not None
                    and self._clock() >= operation_deadline
                ):
                    raise _OperationDeadlineExceeded from None
                raise _AttemptDeadlineExceeded from None
            raise
        if operation_deadline is not None and self._clock() >= operation_deadline:
            raise _OperationDeadlineExceeded
        if timeout.expired() or self._clock() >= attempt_deadline:
            raise _AttemptDeadlineExceeded
        return result

    def _attempt_deadline(
        self,
        operation_deadline: float | None,
        *,
        provider_index: int,
        attempt: int,
    ) -> float | None:
        if operation_deadline is None:
            return None
        remaining = operation_deadline - self._clock()
        if remaining <= 0:
            raise _OperationDeadlineExceeded
        current_attempt_slots = self._retry_policy.max_attempts - attempt + 1
        fallback_slots = len(self._providers) - provider_index - 1
        slot_count = max(1, current_attempt_slots + fallback_slots)
        return self._clock() + remaining / slot_count

    def _retry_preserves_fallback(
        self,
        operation_deadline: float | None,
        *,
        provider_index: int,
        attempt: int,
        delay: float,
    ) -> bool:
        if operation_deadline is None:
            return True
        remaining = operation_deadline - self._clock()
        if remaining <= delay:
            return False
        fallback_slots = len(self._providers) - provider_index - 1
        current_retry_slots = self._retry_policy.max_attempts - attempt
        slot_count = current_retry_slots + fallback_slots
        if fallback_slots <= 0 or slot_count <= 0:
            return True
        fallback_reserve = remaining * fallback_slots / slot_count
        return remaining - delay > fallback_reserve

    async def _sleep_with_deadline(
        self,
        delay: float,
        operation_deadline: float | None,
    ) -> None:
        if operation_deadline is None:
            await self._sleep(delay)
            return
        remaining = operation_deadline - self._clock()
        if remaining <= delay:
            raise _OperationDeadlineExceeded
        timeout = asyncio.timeout(remaining)
        try:
            async with timeout:
                await self._sleep(delay)
        except TimeoutError:
            if timeout.expired():
                raise _OperationDeadlineExceeded from None
            raise
        if timeout.expired() or self._clock() >= operation_deadline:
            raise _OperationDeadlineExceeded

    def _deadline_error(
        self,
        *,
        provider: str | None,
        total_attempts: int,
        circuit_open_count: int,
        fallback_attempted: bool,
        fallback_used: bool,
    ) -> SearchDeadlineExceeded:
        return SearchDeadlineExceeded(
            provider=provider,
            attempt_count=total_attempts,
            circuit_open_count=circuit_open_count,
            fallback_attempted=fallback_attempted,
            fallback_used=fallback_used,
        )

    def _record_transient_failure(self, circuit: _Circuit) -> None:
        circuit.failures += 1
        if circuit.failures >= self._threshold:
            circuit.open_until = self._clock() + self._cooldown

    async def search(self, query: str) -> dict[str, Any]:
        deadline = (
            None
            if self._operation_timeout_seconds is None
            else self._clock() + self._operation_timeout_seconds
        )
        total_attempts = 0
        circuit_open_count = 0
        fallback_attempted = False
        fallback_category: str | None = None
        active_provider: str | None = None
        fallback_used = False
        for provider_index, provider in enumerate(self._providers):
            if not self._can_wait(deadline, 0.0):
                raise self._deadline_error(
                    provider=active_provider,
                    total_attempts=total_attempts,
                    circuit_open_count=circuit_open_count,
                    fallback_attempted=fallback_attempted,
                    fallback_used=fallback_used,
                )
            circuit = self._circuits[provider.provider]
            if circuit.open_until > self._clock():
                circuit_open_count += 1
                continue
            if provider_index > 0:
                fallback_attempted = True
                fallback_used = True
            active_provider = provider.provider
            for attempt in range(1, self._retry_policy.max_attempts + 1):
                if not self._can_wait(deadline, 0.0):
                    raise self._deadline_error(
                        provider=active_provider,
                        total_attempts=total_attempts,
                        circuit_open_count=circuit_open_count,
                        fallback_attempted=fallback_attempted,
                        fallback_used=fallback_used,
                    )
                total_attempts += 1
                try:
                    attempt_deadline = self._attempt_deadline(
                        deadline,
                        provider_index=provider_index,
                        attempt=attempt,
                    )
                    result = await self._search_with_deadline(
                        provider,
                        query,
                        attempt_deadline,
                        deadline,
                    )
                except _AttemptDeadlineExceeded:
                    error = SearchProviderError(
                        provider=provider.provider,
                        category="timeout",
                        retryable=True,
                    )
                except _OperationDeadlineExceeded:
                    self._record_transient_failure(circuit)
                    raise self._deadline_error(
                        provider=provider.provider,
                        total_attempts=total_attempts,
                        circuit_open_count=circuit_open_count,
                        fallback_attempted=fallback_attempted,
                        fallback_used=fallback_used,
                    ) from None
                except SearchProviderError as caught:
                    error = caught
                else:
                    circuit.failures = 0
                    circuit.open_until = 0.0
                    return SearchProviderResult(
                        result,
                        provider=provider.provider,
                        attempt_count=total_attempts,
                        circuit_open_count=circuit_open_count,
                        fallback_used=provider_index > 0,
                        category=fallback_category if provider_index > 0 else None,
                    )
                fallback_category = error.category
                if not error.retryable:
                    circuit.failures = 0
                    circuit.open_until = 0.0
                    break
                if deadline is not None and self._clock() >= deadline:
                    self._record_transient_failure(circuit)
                    raise self._deadline_error(
                        provider=provider.provider,
                        total_attempts=total_attempts,
                        circuit_open_count=circuit_open_count,
                        fallback_attempted=fallback_attempted,
                        fallback_used=fallback_used,
                    )
                if attempt < self._retry_policy.max_attempts:
                    delay = retry_delay(self._retry_policy, attempt, lambda: 0.0)
                    if not self._retry_preserves_fallback(
                        deadline,
                        provider_index=provider_index,
                        attempt=attempt,
                        delay=delay,
                    ):
                        if provider_index < len(self._providers) - 1:
                            self._record_transient_failure(circuit)
                            break
                        raise self._deadline_error(
                            provider=provider.provider,
                            total_attempts=total_attempts,
                            circuit_open_count=circuit_open_count,
                            fallback_attempted=fallback_attempted,
                            fallback_used=fallback_used,
                        )
                    try:
                        await self._sleep_with_deadline(delay, deadline)
                    except _OperationDeadlineExceeded:
                        self._record_transient_failure(circuit)
                        raise self._deadline_error(
                            provider=provider.provider,
                            total_attempts=total_attempts,
                            circuit_open_count=circuit_open_count,
                            fallback_attempted=fallback_attempted,
                            fallback_used=fallback_used,
                        ) from None
                    continue
                self._record_transient_failure(circuit)
                break
        if deadline is not None and self._clock() >= deadline:
            raise self._deadline_error(
                provider=active_provider,
                total_attempts=total_attempts,
                circuit_open_count=circuit_open_count,
                fallback_attempted=fallback_attempted,
                fallback_used=fallback_used,
            )
        raise SearchUnavailableError(
            total_attempts,
            circuit_open_count=circuit_open_count,
            fallback_attempted=fallback_attempted,
        )
