import json
from collections.abc import Callable
from inspect import Parameter, Signature, signature
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator

from gaia_agent.models import FrozenModel, Usage


class ToolCallError(RuntimeError):
    """A recoverable tool failure safe for a provider to report opaquely to the model."""


class SafeToolFailure(FrozenModel):
    """The non-sensitive portion of a recoverable tool failure."""

    tool: str
    category: str = "tool_error"
    retryable: bool = False
    provider: str | None = None
    attempt_count: int = 1
    http_status: int | None = None

    @classmethod
    def from_exception(cls, tool: str, error: ToolCallError) -> "SafeToolFailure":
        category = getattr(error, "category", "tool_error")
        provider = getattr(error, "provider", None)
        attempt_count = getattr(error, "attempt_count", 1)
        http_status = getattr(error, "http_status", None)
        return cls(
            tool=tool,
            category=category if isinstance(category, str) else "tool_error",
            retryable=bool(getattr(error, "retryable", False)),
            provider=provider if isinstance(provider, str) else None,
            attempt_count=attempt_count
            if isinstance(attempt_count, int) and attempt_count > 0
            else 1,
            http_status=http_status if isinstance(http_status, int) else None,
        )


def serialize_tool_output(value: Any) -> str:
    """Serialize a tool output deterministically without expanding Unicode."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tool_output_size_bytes(value: Any) -> int:
    """Return the UTF-8 wire size of a serialized tool output."""
    return len(serialize_tool_output(value).encode("utf-8"))


class ToolSpec(FrozenModel):
    name: str
    description: str
    parameters: dict[str, Any]

    def as_openai_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class ModelRequest(FrozenModel):
    instructions: str
    input_items: tuple[dict[str, Any], ...]
    output_schema: dict[str, Any]
    allowed_tools: tuple[ToolSpec, ...] = ()
    max_output_tokens: int = Field(default=2_000, gt=0)
    max_tool_calls: int = Field(default=8, ge=0)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"

    @field_validator("output_schema")
    @classmethod
    def require_strict_object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        properties = value.get("properties")
        required = value.get("required")
        if (
            value.get("type") != "object"
            or value.get("additionalProperties") is not False
            or not isinstance(properties, dict)
            or set(required or ()) != set(properties)
        ):
            raise ValueError(
                "output_schema must be a strict object JSON Schema with all properties required"
            )
        return value


class Citation(FrozenModel):
    kind: Literal["url", "file"]
    title: str | None = None
    url: str | None = None
    file_id: str | None = None
    filename: str | None = None
    start_index: int | None = None
    end_index: int | None = None


class ToolEvent(FrozenModel):
    call_id: str
    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


class ModelResponse(FrozenModel):
    response_status: Literal["completed", "incomplete", "failed", "unknown"] = "unknown"
    incomplete_reason: Literal["none", "max_output_tokens", "content_filter", "other"] = "none"
    parsed: dict[str, Any] | None = None
    output_text: str = ""
    usage: Usage = Field(default_factory=Usage)
    citations: tuple[Citation, ...] = ()
    tool_events: tuple[ToolEvent, ...] = ()
    remote_file_ids: tuple[str, ...] = ()


class ToolExecutor(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


UsageObserver = Callable[[Usage], None]


class ModelProvider(Protocol):
    """Preferred provider shape.

    Runtime dispatch also accepts legacy structural providers whose ``complete``
    method omits ``usage_observer``. Such providers can report only the aggregate
    usage of a successfully returned response, not partial internal turns.
    """

    supports_vision: bool

    async def complete(
        self,
        request: ModelRequest,
        executor: ToolExecutor | None = None,
        *,
        usage_observer: UsageObserver | None = None,
    ) -> ModelResponse:
        raise NotImplementedError


async def complete_with_usage_observer(
    provider: ModelProvider,
    request: ModelRequest,
    executor: ToolExecutor | None = None,
    *,
    usage_observer: UsageObserver | None = None,
) -> ModelResponse:
    """Call an observer-aware or legacy structural provider exactly once.

    Signature binding happens before invocation, so a ``TypeError`` raised by the
    provider body is never mistaken for an incompatible signature and retried.
    Inspectable providers may take request/executor positionally or by keyword;
    the observer may be named or accepted through ``**kwargs``. Legacy providers
    publish their returned aggregate once after successful completion. An
    uninspectable callable uses the historical call shape exactly once
    (``complete(request)`` without tools, otherwise ``complete(request, executor)``),
    so it cannot receive per-turn usage until it adopts the observer interface.
    """

    complete = provider.complete
    try:
        complete_signature = signature(complete)
    except (TypeError, ValueError):
        # An uninspectable structural provider gets the historical call shape.
        # Do not retry another shape if that one raises from inside the provider.
        if executor is None:
            response = await complete(request)
        else:
            response = await complete(request, executor)
        if usage_observer is not None:
            usage_observer(response.usage)
        return response

    supports_observer = _supports_usage_observer(complete_signature)
    args, kwargs = _bind_complete_call(
        complete_signature,
        request,
        executor,
        usage_observer=usage_observer,
        supports_observer=supports_observer,
    )
    response = await complete(*args, **kwargs)
    if usage_observer is not None and not supports_observer:
        usage_observer(response.usage)
    return response


def _supports_usage_observer(complete_signature: Signature) -> bool:
    observer = complete_signature.parameters.get("usage_observer")
    return observer is not None or any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in complete_signature.parameters.values()
    )


def _bind_complete_call(
    complete_signature: Signature,
    request: ModelRequest,
    executor: ToolExecutor | None,
    *,
    usage_observer: UsageObserver | None,
    supports_observer: bool,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    observer = complete_signature.parameters.get("usage_observer")
    if observer is not None and observer.kind is Parameter.POSITIONAL_ONLY:
        positional = _positional_observer_arguments(
            complete_signature,
            request,
            executor,
            usage_observer,
        )
        executor_kwargs = _keyword_executor_arguments(complete_signature, executor)
        complete_signature.bind(*positional, **executor_kwargs)
        return positional, executor_kwargs

    observer_kwargs = {"usage_observer": usage_observer} if supports_observer else {}
    executor_parameter = complete_signature.parameters.get("executor")
    positional_executor = (
        executor_parameter is None or executor_parameter.kind is Parameter.POSITIONAL_ONLY
    )
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = (
        [
            ((request, executor), dict(observer_kwargs)),
            ((request,), {"executor": executor, **observer_kwargs}),
        ]
        if positional_executor
        else [
            ((request,), {"executor": executor, **observer_kwargs}),
            ((request, executor), dict(observer_kwargs)),
        ]
    )
    if executor is None:
        candidates.extend(
            [
                ((request,), dict(observer_kwargs)),
                ((), {"request": request, **observer_kwargs}),
            ]
        )
    candidates.append(((), {"request": request, "executor": executor, **observer_kwargs}))
    if request_name := _keyword_only_request_name(complete_signature):
        candidates.append(
            (
                (),
                {
                    request_name: request,
                    "executor": executor,
                    **observer_kwargs,
                },
            )
        )
        if executor is None:
            candidates.append(((), {request_name: request, **observer_kwargs}))
    last_error: TypeError | None = None
    for args, kwargs in candidates:
        try:
            complete_signature.bind(*args, **kwargs)
        except TypeError as error:
            last_error = error
            continue
        return args, kwargs
    raise TypeError(
        "provider complete signature is incompatible with ModelProvider"
    ) from last_error


def _keyword_executor_arguments(
    complete_signature: Signature,
    executor: ToolExecutor | None,
) -> dict[str, Any]:
    executor_parameter = complete_signature.parameters.get("executor")
    if executor_parameter is None or executor_parameter.kind is Parameter.POSITIONAL_ONLY:
        return {}
    return {"executor": executor}


def _keyword_only_request_name(complete_signature: Signature) -> str | None:
    for parameter in complete_signature.parameters.values():
        if parameter.kind is Parameter.KEYWORD_ONLY and parameter.name not in {
            "executor",
            "usage_observer",
        }:
            return parameter.name
    return None


def _positional_observer_arguments(
    complete_signature: Signature,
    request: ModelRequest,
    executor: ToolExecutor | None,
    usage_observer: UsageObserver | None,
) -> tuple[Any, ...]:
    values: list[Any] = []
    request_assigned = False
    for parameter in complete_signature.parameters.values():
        if parameter.kind is not Parameter.POSITIONAL_ONLY:
            break
        if not request_assigned:
            values.append(request)
            request_assigned = True
        elif parameter.name == "executor":
            values.append(executor)
        elif parameter.name == "usage_observer":
            values.append(usage_observer)
            return tuple(values)
        elif parameter.default is not Parameter.empty:
            values.append(parameter.default)
        else:
            raise TypeError(
                "provider complete positional parameters are incompatible with ModelProvider"
            )
    raise TypeError("provider usage_observer parameter could not be bound")
