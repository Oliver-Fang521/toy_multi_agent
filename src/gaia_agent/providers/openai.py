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
    Citation,
    ModelRequest,
    ModelResponse,
    SafeToolFailure,
    ToolCallError,
    ToolEvent,
    ToolExecutor,
    serialize_tool_output,
)
from gaia_agent.reliability import (
    RetryPolicy,
    is_transient_provider_error,
    retry_delay,
)


@dataclass
class _ProviderRetryMetadata:
    attempt_count: int = 0
    retry_count: int = 0
    retry_category: str | None = None

    def as_dict(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "provider_attempt_count": self.attempt_count,
            "provider_retry_count": self.retry_count,
        }
        if self.retry_category is not None:
            metadata["retry_category"] = self.retry_category
        return metadata


_CURRENT_RETRY_METADATA: ContextVar[_ProviderRetryMetadata | None] = ContextVar(
    "gaia_provider_retry_metadata", default=None
)


def _append_tool_turn(
    history: list[dict[str, Any]],
    response: Any,
    calls: list[Any],
    outputs: list[dict[str, str]],
) -> None:
    """Append a stateless tool-call turn: reasoning, then calls, then outputs."""
    for item in response.output:
        if item.type != "reasoning":
            continue
        reasoning_parts = [
            {"type": "reasoning_text", "text": part.text}
            for part in item.content
            if part.type == "reasoning_text"
        ]
        if reasoning_parts:
            history.append({"type": "reasoning", "content": reasoning_parts})
    history.extend(
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        }
        for call in calls
    )
    history.extend(outputs)


def _extract_json_object(text: str) -> Any:
    """Parse the first complete JSON object in model text, tolerating prose and fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class OpenAIProvider:
    supports_vision = True

    def __init__(
        self,
        client: Any,
        model: str = "gpt-5.6-sol",
        *,
        use_previous_response_id: bool = True,
        observability: Observability | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._client = client.with_options(max_retries=0)
        self._model = model
        self._use_previous_response_id = use_previous_response_id
        self._observability = observability
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._jitter = jitter

    async def complete(
        self, request: ModelRequest, executor: ToolExecutor | None = None
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
        try:
            try:
                response = await self._complete(request, executor)
            except Exception as error:
                if generation is not None:
                    generation.end(
                        status="error",
                        metadata={
                            "error_type": type(error).__name__,
                            **retry_metadata.as_dict(),
                        },
                        summary=retry_metadata.as_dict(),
                    )
                raise
            if generation is not None:
                generation.end(
                    status="ok",
                    metadata={
                        "parsed": response.parsed is not None,
                        "refused": bool(response.output_text and response.parsed is None),
                        "tool_call_count": len(response.tool_events),
                        **retry_metadata.as_dict(),
                    },
                    summary={
                        "decision": "parsed" if response.parsed is not None else "unparsed",
                        "parsed": response.parsed is not None,
                        "refused": bool(response.output_text and response.parsed is None),
                        "tool_call_count": len(response.tool_events),
                        **retry_metadata.as_dict(),
                    },
                    usage=response.usage,
                )
            return response
        finally:
            _CURRENT_RETRY_METADATA.reset(retry_token)

    async def _complete(
        self, request: ModelRequest, executor: ToolExecutor | None = None
    ) -> ModelResponse:
        common = {
            "model": self._model,
            "instructions": request.instructions,
            "tools": [tool.as_openai_dict() for tool in request.allowed_tools],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "gaia_response",
                    "schema": request.output_schema,
                    "strict": True,
                }
            },
            "reasoning": {"effort": request.reasoning_effort},
            "max_output_tokens": request.max_output_tokens,
        }
        if request.max_tool_calls > 0:
            common["max_tool_calls"] = request.max_tool_calls
        history: list[dict[str, Any]] = [dict(item) for item in request.input_items]
        response = await self._create_with_retry(input=request.input_items, **common)
        responses = [response]
        tool_events: list[ToolEvent] = []
        tool_call_count = 0
        allowed_tool_names = {tool.name for tool in request.allowed_tools}
        tool_free_common = {**common, "tools": []}
        tool_free_common.pop("max_tool_calls", None)

        while calls := [item for item in response.output if item.type == "function_call"]:
            if tool_call_count + len(calls) > request.max_tool_calls:
                outputs = [
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": '{"error":"tool_budget_exhausted"}',
                    }
                    for call in calls
                ]
                if not self._use_previous_response_id:
                    _append_tool_turn(history, response, calls, outputs)
                response = await self._continue(
                    previous_response=response,
                    input_items=outputs if self._use_previous_response_id else list(history),
                    common=tool_free_common,
                )
                responses.append(response)
                break
            if undeclared := next(
                (call.name for call in calls if call.name not in allowed_tool_names), None
            ):
                raise ValueError(f"undeclared tool: {undeclared}")
            if executor is None:
                raise RuntimeError("model requested a tool but no executor was provided")

            outputs: list[dict[str, str]] = []
            for call in calls:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("function-call arguments must decode to an object")
                try:
                    result = await executor.call(call.name, arguments)
                except ToolCallError as error:
                    failure = SafeToolFailure.from_exception(call.name, error)
                    details = failure.model_dump(exclude={"tool", "http_status"})
                    if failure.http_status is not None:
                        details["http_status"] = failure.http_status
                    result = {
                        "error": "tool_failed",
                        **details,
                    }
                output = {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": serialize_tool_output(result),
                }
                outputs.append(output)
                tool_events.append(
                    ToolEvent(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=arguments,
                        output=result,
                    )
                )
            tool_call_count += len(calls)
            if not self._use_previous_response_id:
                _append_tool_turn(history, response, calls, outputs)
            next_common = (
                tool_free_common
                if tool_call_count == request.max_tool_calls
                else common
            )
            response = await self._continue(
                previous_response=response,
                input_items=outputs if self._use_previous_response_id else list(history),
                common=next_common,
            )
            responses.append(response)

        output_text, refused = _extract_output_text(response)
        citations, remote_file_ids = _extract_sources(responses)
        parsed = None if refused or not output_text else _extract_json_object(output_text)
        if parsed is not None and not isinstance(parsed, dict):
            raise ValueError("structured response must decode to an object")
        return ModelResponse(
            parsed=parsed,
            output_text=output_text,
            usage=_aggregate_usage(responses),
            citations=tuple(citations),
            tool_events=tuple(tool_events),
            remote_file_ids=tuple(remote_file_ids),
        )

    async def _continue(
        self,
        *,
        previous_response: Any,
        input_items: list[dict[str, Any]],
        common: dict[str, Any],
    ) -> Any:
        if self._use_previous_response_id:
            return await self._create_with_retry(
                previous_response_id=previous_response.id,
                input=input_items,
                **common,
            )
        return await self._create_with_retry(input=input_items, **common)

    async def _create_with_retry(self, **kwargs: Any) -> Any:
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                retry_metadata = _CURRENT_RETRY_METADATA.get()
                if retry_metadata is not None:
                    retry_metadata.attempt_count += 1
                return await self._client.responses.create(**kwargs)
            except Exception as error:
                if (
                    not is_transient_provider_error(error)
                    or attempt == self._retry_policy.max_attempts
                ):
                    raise
                self._record_retry(error, attempt)
                await self._sleep(retry_delay(self._retry_policy, attempt, self._jitter))
        raise AssertionError("bounded provider retry did not return or raise")

    @staticmethod
    def _record_retry(error: BaseException, _failed_attempt: int) -> None:
        retry_metadata = _CURRENT_RETRY_METADATA.get()
        if retry_metadata is not None:
            retry_metadata.retry_count += 1
            retry_metadata.retry_category = type(error).__name__


def _aggregate_usage(responses: list[Any]) -> Usage:
    input_tokens = output_tokens = reasoning_tokens = total_tokens = 0
    for response in responses:
        usage = response.usage
        if usage is None:
            continue
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        total_tokens += usage.total_tokens
        reasoning_tokens += usage.output_tokens_details.reasoning_tokens
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def _extract_output_text(response: Any) -> tuple[str, bool]:
    parts: list[str] = []
    refused = False
    for item in response.output:
        if item.type != "message":
            continue
        for content in item.content:
            if content.type == "output_text":
                parts.append(content.text)
            elif content.type == "refusal":
                parts.append(content.refusal)
                refused = True
    return "".join(parts), refused


def _extract_sources(responses: list[Any]) -> tuple[list[Citation], list[str]]:
    citations: list[Citation] = []
    remote_file_ids: list[str] = []
    for response in responses:
        for item in response.output:
            if item.type != "message":
                continue
            for content in item.content:
                if content.type != "output_text":
                    continue
                for annotation in content.annotations:
                    if annotation.type == "url_citation":
                        citations.append(
                            Citation(
                                kind="url",
                                title=annotation.title,
                                url=annotation.url,
                                start_index=annotation.start_index,
                                end_index=annotation.end_index,
                            )
                        )
                    elif annotation.type in {"file_citation", "container_file_citation"}:
                        citations.append(
                            Citation(
                                kind="file",
                                file_id=annotation.file_id,
                                filename=annotation.filename,
                            )
                        )
                        if annotation.file_id not in remote_file_ids:
                            remote_file_ids.append(annotation.file_id)
                    elif annotation.type == "file_path":
                        citations.append(Citation(kind="file", file_id=annotation.file_id))
                        if annotation.file_id not in remote_file_ids:
                            remote_file_ids.append(annotation.file_id)
    return citations, remote_file_ids
