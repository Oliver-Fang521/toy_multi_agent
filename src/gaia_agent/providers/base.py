from typing import Any, Literal, Protocol

from pydantic import Field, field_validator

from gaia_agent.models import FrozenModel, Usage


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


class ModelProvider(Protocol):
    async def complete(
        self, request: ModelRequest, executor: ToolExecutor | None = None
    ) -> ModelResponse:
        raise NotImplementedError
