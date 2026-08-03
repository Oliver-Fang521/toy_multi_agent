import json
from typing import Any

from openai import APITimeoutError, InternalServerError, RateLimitError

from gaia_agent.models import Usage
from gaia_agent.providers.base import (
    Citation,
    ModelRequest,
    ModelResponse,
    ToolEvent,
    ToolExecutor,
)


class OpenAIProvider:
    def __init__(self, client: Any, model: str = "gpt-5.6-sol") -> None:
        self._client = client.with_options(max_retries=0)
        self._model = model

    async def complete(
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
        response = await self._create_with_retry(input=request.input_items, **common)
        responses = [response]
        tool_events: list[ToolEvent] = []
        tool_call_count = 0
        allowed_tool_names = {tool.name for tool in request.allowed_tools}

        while calls := [item for item in response.output if item.type == "function_call"]:
            if tool_call_count + len(calls) > request.max_tool_calls:
                raise RuntimeError("tool-call budget exhausted")
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
                result = await executor.call(call.name, arguments)
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, separators=(",", ":")),
                    }
                )
                tool_events.append(
                    ToolEvent(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=arguments,
                        output=result,
                    )
                )
            tool_call_count += len(calls)
            response = await self._create_with_retry(
                previous_response_id=response.id,
                input=outputs,
                **common,
            )
            responses.append(response)

        output_text, refused = _extract_output_text(response)
        citations, remote_file_ids = _extract_sources(responses)
        parsed = None if refused or not output_text else json.loads(output_text)
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

    async def _create_with_retry(self, **kwargs: Any) -> Any:
        for attempt in range(2):
            try:
                return await self._client.responses.create(**kwargs)
            except (RateLimitError, APITimeoutError, InternalServerError):
                if attempt == 1:
                    raise
        raise AssertionError("retry loop did not return or raise")


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
