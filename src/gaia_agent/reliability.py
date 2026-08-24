from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

TRANSIENT_PROVIDER_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
)


class InvocationDeadlineExceeded(TimeoutError):
    """A new external invocation was attempted after the vote closed."""


class InvocationDeadline:
    """Shared absolute vote boundary retained by detached orchestration tasks."""

    def __init__(self, deadline: float, clock: Callable[[], float]) -> None:
        self.deadline = deadline
        self._clock = clock
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def ensure_open(self) -> None:
        if self._closed or self._clock() >= self.deadline:
            raise InvocationDeadlineExceeded("vote invocation deadline exhausted")


_CURRENT_INVOCATION_DEADLINE: ContextVar[InvocationDeadline | None] = ContextVar(
    "gaia_invocation_deadline",
    default=None,
)


@contextmanager
def invocation_deadline_scope(
    deadline: InvocationDeadline | None,
) -> Iterator[None]:
    token: Token[InvocationDeadline | None] = _CURRENT_INVOCATION_DEADLINE.set(
        deadline
    )
    try:
        yield
    finally:
        _CURRENT_INVOCATION_DEADLINE.reset(token)


def ensure_current_invocation_allowed() -> None:
    deadline = _CURRENT_INVOCATION_DEADLINE.get()
    if deadline is not None:
        deadline.ensure_open()


class ProviderTransientExhaustedError(RuntimeError):
    """Safe typed cause raised after one provider operation exhausts its retry policy."""

    def __init__(self, *, retry_category: str, attempt_count: int) -> None:
        super().__init__("provider transient attempts exhausted")
        self.retry_category = (
            retry_category
            if retry_category in TRANSIENT_PROVIDER_ERROR_NAMES
            else "provider_transient"
        )
        self.attempt_count = max(1, attempt_count)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0


def is_transient_provider_error(error: BaseException) -> bool:
    return any(cls.__name__ in TRANSIENT_PROVIDER_ERROR_NAMES for cls in type(error).__mro__)


def transient_provider_error_category(error: BaseException) -> str:
    """Return only an allowlisted class category, never a custom exception name or message."""
    return next(
        (
            cls.__name__
            for cls in type(error).__mro__
            if cls.__name__ in TRANSIENT_PROVIDER_ERROR_NAMES
        ),
        "provider_transient",
    )


def retry_delay(policy: RetryPolicy, failed_attempt: int, jitter: Callable[[], float]) -> float:
    exponential = policy.base_delay_seconds * (2 ** (failed_attempt - 1))
    return min(
        policy.max_delay_seconds,
        exponential * (1.0 + max(0.0, min(1.0, jitter())) * 0.25),
    )
