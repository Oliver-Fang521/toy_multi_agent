from collections.abc import Callable
from dataclasses import dataclass

TRANSIENT_PROVIDER_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0


def is_transient_provider_error(error: BaseException) -> bool:
    return any(cls.__name__ in TRANSIENT_PROVIDER_ERROR_NAMES for cls in type(error).__mro__)


def retry_delay(policy: RetryPolicy, failed_attempt: int, jitter: Callable[[], float]) -> float:
    exponential = policy.base_delay_seconds * (2 ** (failed_attempt - 1))
    return min(
        policy.max_delay_seconds,
        exponential * (1.0 + max(0.0, min(1.0, jitter())) * 0.25),
    )
