import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from gaia_agent.reliability import RetryPolicy
from gaia_agent.search import (
    SearchProviderChain,
    SearchProviderError,
    SerpAPISearchProvider,
    SerperSearchProvider,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_factory", "expected_method", "expected_url", "payload"),
    [
        (
            lambda client: SerperSearchProvider("serper-secret", client, timeout_seconds=1.5),
            "POST",
            "https://google.serper.dev/search",
            {
                "answerBox": {"answer": "42"},
                "organic": [
                    {"link": "https://example.test/one", "title": "One"},
                    {"link": "https://example.test/two"},
                ],
            },
        ),
        (
            lambda client: SerpAPISearchProvider("serpapi-secret", client, timeout_seconds=1.5),
            "GET",
            "https://serpapi.com/search.json",
            {
                "answer_box": {"answer": "42"},
                "organic_results": [
                    {"link": "https://example.test/one", "title": "One"},
                    {"link": "https://example.test/two"},
                ],
            },
        ),
    ],
)
async def test_search_adapters_normalize_provider_payloads(
    provider_factory: Callable[[httpx.AsyncClient], Any],
    expected_method: str,
    expected_url: str,
    payload: dict[str, Any],
) -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await provider_factory(client).search("meaning of life")

    assert result == {
        "answer": "42",
        "sources": [
            {"url": "https://example.test/one", "title": "One"},
            {"url": "https://example.test/two", "title": None},
        ],
    }
    assert observed[0].method == expected_method
    assert str(observed[0].url).startswith(expected_url)
    if expected_method == "POST":
        assert observed[0].headers["X-API-KEY"] == "serper-secret"
        assert json.loads(observed[0].content) == {"q": "meaning of life"}
    else:
        assert observed[0].url.params["api_key"] == "serpapi-secret"
        assert observed[0].url.params["q"] == "meaning of life"
    assert observed[0].extensions["timeout"]["read"] == 1.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_factory",
    [
        lambda client: SerperSearchProvider("serper-secret", client, timeout_seconds=1.5),
        lambda client: SerpAPISearchProvider("serpapi-secret", client, timeout_seconds=1.5),
    ],
)
async def test_search_adapters_reject_malformed_payload_without_key(
    provider_factory: Callable[[httpx.AsyncClient], Any],
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=["bad"]))
    ) as client:
        with pytest.raises(SearchProviderError) as caught:
            await provider_factory(client).search("question")

    assert caught.value.category == "parse_failure"
    assert caught.value.retryable is False
    assert "secret" not in str(caught.value)


class _ScriptedProvider:
    def __init__(self, provider: str, results: list[dict[str, Any] | Exception]) -> None:
        self.provider = provider
        self._results = iter(results)
        self.queries: list[str] = []

    async def search(self, query: str) -> dict[str, Any]:
        self.queries.append(query)
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


def _transient(provider: str) -> SearchProviderError:
    return SearchProviderError(provider=provider, category="timeout", retryable=True)


@pytest.mark.asyncio
async def test_search_chain_falls_back_in_order_after_bounded_transient_retries() -> None:
    openai = _ScriptedProvider("openai", [_transient("openai"), _transient("openai")])
    serper = _ScriptedProvider("serper", [{"answer": "fallback", "sources": []}])
    serpapi = _ScriptedProvider("serpapi", [{"answer": "unused", "sources": []}])
    chain = SearchProviderChain(
        [openai, serper, serpapi],
        RetryPolicy(max_attempts=2),
        clock=lambda: 0.0,
    )

    result = await chain.search("question")

    assert result == {"answer": "fallback", "sources": []}
    assert openai.queries == ["question", "question"]
    assert serper.queries == ["question"]
    assert serpapi.queries == []


@pytest.mark.asyncio
async def test_search_chain_opens_circuit_skips_provider_then_closes_after_probe() -> None:
    now = [0.0]
    openai = _ScriptedProvider(
        "openai",
        [_transient("openai"), _transient("openai"), {"answer": "recovered", "sources": []}],
    )
    serper = _ScriptedProvider(
        "serper",
        [
            {"answer": "one", "sources": []},
            {"answer": "two", "sources": []},
            {"answer": "three", "sources": []},
        ],
    )
    chain = SearchProviderChain(
        [openai, serper], RetryPolicy(max_attempts=1), clock=lambda: now[0]
    )

    assert await chain.search("first") == {"answer": "one", "sources": []}
    assert await chain.search("second") == {"answer": "two", "sources": []}
    assert await chain.search("third") == {"answer": "three", "sources": []}
    assert openai.queries == ["first", "second"]

    now[0] = 31.0
    assert await chain.search("probe") == {"answer": "recovered", "sources": []}
    assert openai.queries == ["first", "second", "probe"]


@pytest.mark.asyncio
async def test_search_chain_waits_with_bounded_backoff_before_retrying_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    openai = _ScriptedProvider("openai", [_transient("openai"), _transient("openai")])
    serper = _ScriptedProvider("serper", [{"answer": "fallback", "sources": []}])
    chain = SearchProviderChain(
        [openai, serper],
        RetryPolicy(max_attempts=2, base_delay_seconds=0.5, max_delay_seconds=1),
        clock=lambda: 0.0,
    )

    assert await chain.search("question") == {"answer": "fallback", "sources": []}
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_search_chain_stops_on_nonretryable_provider_failure() -> None:
    openai = _ScriptedProvider(
        "openai", [SearchProviderError(provider="openai", category="parse_failure", retryable=False)]
    )
    serper = _ScriptedProvider("serper", [{"answer": "must not run", "sources": []}])
    chain = SearchProviderChain([openai, serper], RetryPolicy(max_attempts=2), clock=lambda: 0.0)

    with pytest.raises(SearchProviderError, match="parse_failure"):
        await chain.search("question")

    assert openai.queries == ["question"]
    assert serper.queries == []


@pytest.mark.asyncio
async def test_search_chain_propagates_unclassified_provider_errors_without_opening_circuit() -> None:
    openai = _ScriptedProvider("openai", [RuntimeError("implementation failure")])
    serper = _ScriptedProvider("serper", [{"answer": "must not run", "sources": []}])
    chain = SearchProviderChain([openai, serper], RetryPolicy(max_attempts=2), clock=lambda: 0.0)

    with pytest.raises(RuntimeError, match="implementation failure"):
        await chain.search("question")

    assert serper.queries == []
    assert chain._circuits["openai"].failures == 0
