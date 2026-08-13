"""DeepSeek Chat Completions provider adapter."""

from __future__ import annotations

import json
from typing import Any

from openai import APITimeoutError, InternalServerError, RateLimitError

from gaia_agent.models import Usage
from gaia_agent.observability import Observability
from gaia_agent.providers.base import (
    ModelRequest,
    ModelResponse,
    ToolCallError,
    ToolEvent,
    ToolExecutor,
    serialize_tool_output,
)
from gaia_agent.providers.openai import _extract_json_object


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {"input_text", "output_text", "text"}:
                    parts.append(str(part.get("text", "")))
                elif part_type in {"input_image", "image_url"}:
                    parts.append("[image]")
                elif part.get("text") is not None:
                    parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _input_items_to_messages(input_items: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in input_items:
        role = item.get("role", "user")
        content = _content_to_text(item.get("content", ""))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _chat_tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _assistant_message(message: Any) -> dict[str, Any]:
    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        for call in (message.tool_calls or [])
    ]
    return {
        "role": "assistant",
        "content": message.content or "",
        "reasoning_content": getattr(message, "reasoning_content", None) or "",
        "tool_calls": tool_calls,
    }


def _merge_usage(accumulated: Usage, usage: Any) -> Usage:
    if usage is None:
        return accumulated
    reasoning = 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None and getattr(details, "reasoning_tokens", None):
        reasoning = int(details.reasoning_tokens)
    return Usage(
        input_tokens=accumulated.input_tokens + getattr(usage, "prompt_tokens", 0),
        output_tokens=accumulated.output_tokens + getattr(usage, "completion_tokens", 0),
        reasoning_tokens=accumulated.reasoning_tokens + reasoning,
        total_tokens=accumulated.total_tokens + getattr(usage, "total_tokens", 0),
    )


def _validate_value(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    if "$ref" in schema:
        ref = schema["$ref"].removeprefix("#/$defs/")
        return _validate_value(value, root.get("$defs", {}).get(ref, {}), root)
    if "anyOf" in schema:
        return any(_validate_value(value, child, root) for child in schema["anyOf"])
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        for required in schema.get("required") or ():
            if required not in value:
                return False
        for key, child in (schema.get("properties") or {}).items():
            if key in value and not _validate_value(value[key], child, root):
                return False
        return True
    if kind == "array":
        if not isinstance(value, list):
            return False
        items = schema.get("items")
        if items is None:
            return True
        return all(_validate_value(item, items, root) for item in value)
    if kind == "string":
        if not isinstance(value, str):
            return False
        if "const" in schema:
            return value == schema["const"]
        if "enum" in schema:
            return value in schema["enum"]
        return True
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return True


def _schema_mismatch(parsed: Any, schema: dict[str, Any]) -> bool:
    return not _validate_value(parsed, schema, schema)


_RECOVERY_PROMPT = (
    "Your previous response did not match the required JSON schema. "
    "Return ONLY the JSON object exactly matching the schema, with no prose, "
    "no commentary, and no markdown code fences."
)


class DeepSeekProvider:
    """ModelProvider adapter using DeepSeek's Chat Completions tool-call loop."""

    # The Chat Completions adapter serializes image content as a text marker;
    # declaring this capability prevents visual plans from being sent as if the
    # model could inspect pixels.
    supports_vision = False

    def __init__(
        self,
        client: Any,
        model: str = "deepseek-v4-flash",
        *,
        observability: Observability | None = None,
    ) -> None:
        self._client = client.with_options(max_retries=0)
        self._model = model
        self._observability = observability

    async def complete(
        self, request: ModelRequest, executor: ToolExecutor | None = None
    ) -> ModelResponse:
        trace = self._observability.current_trace() if self._observability else None
        generation = (
            trace.generation(
                "provider-generation",
                {"model": self._model, "reasoning_effort": request.reasoning_effort},
            )
            if trace is not None
            else None
        )
        try:
            response = await self._complete(request, executor)
        except Exception as error:
            if generation is not None:
                generation.end(
                    status="error",
                    metadata={"error_type": type(error).__name__},
                )
            raise
        if generation is not None:
            generation.end(
                status="ok",
                metadata={
                    "parsed": response.parsed is not None,
                    "refused": bool(response.output_text and response.parsed is None),
                    "tool_call_count": len(response.tool_events),
                },
                summary={
                    "decision": "parsed" if response.parsed is not None else "unparsed",
                    "parsed": response.parsed is not None,
                    "refused": bool(response.output_text and response.parsed is None),
                    "tool_call_count": len(response.tool_events),
                },
                usage=response.usage,
            )
        return response

    async def _complete(
        self, request: ModelRequest, executor: ToolExecutor | None = None
    ) -> ModelResponse:
        schema_hint = json.dumps(request.output_schema, sort_keys=True)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{request.instructions}\n\nRespond with a single JSON object only, "
                    f"conforming exactly to this schema:\n{schema_hint}"
                ),
            }
        ]
        messages.extend(_input_items_to_messages(request.input_items))
        allowed_tool_names = {tool.name for tool in request.allowed_tools}
        common: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "reasoning_effort": request.reasoning_effort,
            "response_format": {"type": "json_object"},
            "max_tokens": max(request.max_output_tokens * 3, 6000),
        }
        if request.allowed_tools:
            common["tools"] = [_chat_tool(tool) for tool in request.allowed_tools]

        tool_call_count = 0
        tool_events: list[ToolEvent] = []
        usage = Usage()
        response = await self._create_with_retry(**common)
        usage = _merge_usage(usage, response.usage)
        recovery_used = False

        while True:
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            if not calls:
                if getattr(message, "refusal", None):
                    break
                output_text = message.content or ""
                parsed = _extract_json_object(output_text) if output_text else None
                if parsed is not None and not isinstance(parsed, dict):
                    raise ValueError("structured response must decode to an object")
                if not recovery_used and _schema_mismatch(parsed, request.output_schema):
                    recovery_used = True
                    messages.append({"role": "user", "content": _RECOVERY_PROMPT})
                    common["messages"] = list(messages)
                    common.pop("tools", None)
                    response = await self._create_with_retry(**common)
                    usage = _merge_usage(usage, response.usage)
                    continue
                break
            if undeclared := next(
                (call.function.name for call in calls if call.function.name not in allowed_tool_names),
                None,
            ):
                raise ValueError(f"undeclared tool: {undeclared}")
            if executor is None:
                raise RuntimeError("model requested a tool but no executor was provided")
            messages.append(_assistant_message(message))
            if tool_call_count + len(calls) > request.max_tool_calls:
                for call in calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": '{"error":"tool_budget_exhausted"}',
                        }
                    )
                common.pop("tools", None)
                common["messages"] = list(messages)
                response = await self._create_with_retry(**common)
                usage = _merge_usage(usage, response.usage)
                continue
            for call in calls:
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("function-call arguments must decode to an object")
                try:
                    result = await executor.call(call.function.name, arguments)
                except ToolCallError:
                    result = {"error": "tool_failed"}
                serialized = serialize_tool_output(result)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": serialized})
                tool_events.append(
                    ToolEvent(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                        output=result,
                    )
                )
            tool_call_count += len(calls)
            common["messages"] = list(messages)
            response = await self._create_with_retry(**common)
            usage = _merge_usage(usage, response.usage)

        output_text = ""
        parsed = None
        if getattr(message, "refusal", None):
            output_text = message.refusal
        elif message.content:
            output_text = message.content
            parsed = _extract_json_object(output_text)
            if parsed is not None and not isinstance(parsed, dict):
                raise ValueError("structured response must decode to an object")
        return ModelResponse(
            parsed=parsed,
            output_text=output_text,
            usage=usage,
            tool_events=tuple(tool_events),
        )

    async def _create_with_retry(self, **kwargs: Any) -> Any:
        for attempt in range(2):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, InternalServerError):
                if attempt == 1:
                    raise
        raise AssertionError("retry loop did not return or raise")
