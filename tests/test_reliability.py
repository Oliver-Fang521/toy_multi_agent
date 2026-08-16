import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from gaia_agent.reliability import (
    RetryPolicy,
    is_transient_provider_error,
    retry_delay,
)


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.openai.com"),
    )


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
        APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")),
        RateLimitError("limited", response=_response(429), body=None),
        InternalServerError("server", response=_response(500), body=None),
    ],
)
def test_transient_provider_errors_are_classified(error: Exception) -> None:
    assert is_transient_provider_error(error)


def test_retry_delay_is_bounded() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=4.0)

    assert retry_delay(policy, 1, lambda: 0.0) == 1.0
    assert retry_delay(policy, 2, lambda: 0.0) == 2.0
    assert retry_delay(policy, 3, lambda: 1.0) <= 4.0
