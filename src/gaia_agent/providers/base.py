import json
from collections.abc import Callable
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
            attempt_count=attempt_count if isinstance(attempt_count, int) and attempt_count > 0 else 1,
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
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"

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
    supports_vision: bool

    async def complete(
        self,
        request: ModelRequest,
        executor: ToolExecutor | None = None,
        *,
        usage_observer: UsageObserver | None = None,
    ) -> ModelResponse:
        raise NotImplementedError
