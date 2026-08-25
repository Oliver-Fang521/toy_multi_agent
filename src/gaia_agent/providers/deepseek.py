"""DeepSeek Chat Completions provider adapter."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from gaia_agent.models import Usage
from gaia_agent.observability import Observability
from gaia_agent.providers.base import (
    ModelRequest,
    ModelResponse,
    ToolCallError,
    ToolEvent,
    ToolExecutor,
    UsageObserver,
    serialize_tool_output,
)
from gaia_agent.providers.openai import _extract_json_object
from gaia_agent.reliability import (
    ProviderTransientExhaustedError,
    RetryPolicy,
    ensure_current_invocation_allowed,
    is_transient_provider_error,
    retry_delay,
    transient_provider_error_category,
)


@dataclass
class _ProviderRetryMetadata:
    attempt_count: int = 0
    retry_count: int = 0
    retry_category: str | None = None
    retry_decision: str = "not_needed"

    def as_dict(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "attempt_number": max(1, self.attempt_count),
            "provider_attempt_count": self.attempt_count,
            "provider_retry_count": self.retry_count,
            "retry_decision": self.retry_decision,
        }
        if self.retry_category is not None:
            metadata["retry_category"] = self.retry_category
        return metadata


_CURRENT_RETRY_METADATA: ContextVar[_ProviderRetryMetadata | None] = ContextVar(
    "gaia_deepseek_retry_metadata", default=None
)


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


def _publish_and_merge_usage(
    accumulated: Usage,
    usage: Any,
    usage_observer: UsageObserver | None,
) -> Usage:
    delta = _merge_usage(Usage(), usage)
    if usage_observer is not None:
        usage_observer(delta)
    return _merge_usage(accumulated, usage)


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
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._client = client.with_options(max_retries=0)
        self._model = model
        self._observability = observability
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._jitter = jitter

    async def complete(
        self,
        request: ModelRequest,
        executor: ToolExecutor | None = None,
        *,
        usage_observer: UsageObserver | None = None,
    ) -> ModelResponse:
        retry_metadata = _ProviderRetryMetadata()
        retry_token: Token[_ProviderRetryMetadata | None] = _CURRENT_RETRY_METADATA.set(
            retry_metadata
        )
        trace = self._observability.current_trace() if self._observability else None
        generation = (
            trace.generation(
                "provider-generation",
                {"model": self._model, "reasoning_effort": request.reasoning_effort},
            )
            if trace is not None
            else None
        )
        span_status = "cancelled"
        span_metadata: dict[str, object] = {}
        span_summary: dict[str, object] = {}
        span_usage: Usage | None = None
        try:
            response = await self._complete(
                request,
                executor,
                usage_observer=usage_observer,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            span_status = "error"
            span_metadata = {"error_type": type(error).__name__, **retry_metadata.as_dict()}
            span_summary = retry_metadata.as_dict()
            raise
        else:
            span_status = "ok"
            span_metadata = {
                "parsed": response.parsed is not None,
                "refused": bool(response.output_text and response.parsed is None),
                "tool_call_count": len(response.tool_events),
                **retry_metadata.as_dict(),
            }
            span_summary = {
                "decision": "parsed" if response.parsed is not None else "unparsed",
                **span_metadata,
            }
            span_usage = response.usage
            return response
        finally:
            if generation is not None:
                generation.end(
                    status=span_status,
                    metadata=span_metadata,
                    summary=span_summary,
                    usage=span_usage,
                )
            _CURRENT_RETRY_METADATA.reset(retry_token)

    async def _complete(
        self,
        request: ModelRequest,
        executor: ToolExecutor | None = None,
        *,
        usage_observer: UsageObserver | None = None,
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
        usage = _publish_and_merge_usage(usage, response.usage, usage_observer)
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
                    usage = _publish_and_merge_usage(
                        usage, response.usage, usage_observer
                    )
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
                usage = _publish_and_merge_usage(
                    usage, response.usage, usage_observer
                )
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
            usage = _publish_and_merge_usage(usage, response.usage, usage_observer)

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
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                ensure_current_invocation_allowed()
                retry_metadata = _CURRENT_RETRY_METADATA.get()
                if retry_metadata is not None:
                    retry_metadata.attempt_count += 1
                response = await self._client.chat.completions.create(**kwargs)
                if retry_metadata is not None and retry_metadata.retry_count:
                    retry_metadata.retry_decision = "recovered"
                return response
            except Exception as error:
                retry_metadata = _CURRENT_RETRY_METADATA.get()
                if not is_transient_provider_error(error):
                    if retry_metadata is not None:
                        retry_metadata.retry_decision = "not_retryable"
                    raise
                category = transient_provider_error_category(error)
                if retry_metadata is not None:
                    retry_metadata.retry_category = category
                if attempt == self._retry_policy.max_attempts:
                    if retry_metadata is not None:
                        retry_metadata.retry_decision = "exhausted"
                    raise ProviderTransientExhaustedError(
                        retry_category=category,
                        attempt_count=attempt,
                    ) from error
                if retry_metadata is not None:
                    retry_metadata.retry_count += 1
                    retry_metadata.retry_decision = "retry"
                await self._sleep(retry_delay(self._retry_policy, attempt, self._jitter))
        raise AssertionError("bounded provider retry did not return or raise")
