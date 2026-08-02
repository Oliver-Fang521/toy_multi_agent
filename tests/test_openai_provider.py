from collections import deque
from typing import Any

import httpx
import pytest
from openai import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from openai.types.responses import Response
from pydantic import ValidationError

from gaia_agent.models import Usage
from gaia_agent.providers import ModelRequest, OpenAIProvider, ToolSpec


def _usage(input_tokens: int, output_tokens: int, reasoning_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _response(
    response_id: str,
    output: list[dict[str, Any]],
    *,
    usage: dict[str, Any] | None = None,
) -> Response:
    # Keep the fake aligned with the complete documented Response shape instead of
    # inventing a partial object containing only attributes consumed by the adapter.
    return Response.model_validate(
        {
            "id": response_id,
            "created_at": 1_785_624_000.0,
            "error": None,
            "incomplete_details": None,
            "instructions": "answer with JSON",
            "metadata": {},
            "model": "gpt-5.6-sol",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": False,
            "completed_at": 1_785_624_001.0,
            "conversation": None,
            "max_output_tokens": 500,
            "max_tool_calls": 4,
            "moderation": None,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "prompt_cache_options": None,
            "prompt_cache_retention": None,
            "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
            "safety_identifier": None,
            "service_tier": "default",
            "status": "completed",
            "text": {"format": {"type": "json_object"}, "verbosity": "medium"},
            "top_logprobs": 0,
            "truncation": "disabled",
            "usage": usage or _usage(0, 0, 0),
            "user": None,
        }
    )


class FakeResponsesClient:
    def __init__(self) -> None:
        self.responses = self
        self.queued: deque[Response | Exception] = deque()
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Response:
        self.requests.append(kwargs)
        value = self.queued.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def queue_function_call(
        self,
        name: str,
        arguments: str,
        call_id: str,
        *,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.queued.append(
            _response(
                f"resp-{len(self.queued) + 1}",
                [
                    {
                        "type": "function_call",
                        "id": f"fc-{call_id}",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "status": "completed",
                    }
                ],
                usage=usage,
            )
        )

    def queue_text(
        self,
        text: str,
        *,
        annotations: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.queued.append(
            _response(
                f"resp-{len(self.queued) + 1}",
                [
                    {
                        "id": f"msg-{len(self.queued) + 1}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": annotations or [],
                                "logprobs": [],
                            }
                        ],
                    }
                ],
                usage=usage,
            )
        )

    def queue_refusal(self, refusal: str) -> None:
        self.queued.append(
            _response(
                f"resp-{len(self.queued) + 1}",
                [
                    {
                        "id": f"msg-{len(self.queued) + 1}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": refusal}],
                    }
                ],
            )
        )


@pytest.fixture
def fake_responses_client() -> FakeResponsesClient:
    return FakeResponsesClient()


class RecordingExecutor:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return self.result


def model_request(**updates: Any) -> ModelRequest:
    values: dict[str, Any] = {
        "instructions": "answer with JSON",
        "input_items": ({"role": "user", "content": "What is six times seven?"},),
        "output_schema": {"type": "json_object"},
        "allowed_tools": (
            ToolSpec(
                name="execute_python",
                description="Run Python code",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            ),
        ),
        "max_output_tokens": 500,
        "max_tool_calls": 4,
        "reasoning_effort": "medium",
    }
    values.update(updates)
    return ModelRequest(**values)


@pytest.mark.asyncio
async def test_openai_provider_executes_function_call_and_continues(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("execute_python", '{"code":"print(6*7)"}', "call-1")
    fake_responses_client.queue_text('{"answer":"42"}')
    executor = RecordingExecutor({"stdout": "42"})
    provider = OpenAIProvider(client=fake_responses_client, model="gpt-5.6-sol")

    response = await provider.complete(model_request(), executor)

    assert executor.calls == [("execute_python", {"code": "print(6*7)"})]
    assert fake_responses_client.requests[1]["input"] == [
        {"type": "function_call_output", "call_id": "call-1", "output": '{"stdout":"42"}'}
    ]
    assert fake_responses_client.requests[1]["previous_response_id"] == "resp-1"
    assert fake_responses_client.requests[1]["instructions"] == "answer with JSON"
    assert response.parsed == {"answer": "42"}
    assert response.tool_events[0].model_dump() == {
        "call_id": "call-1",
        "name": "execute_python",
        "arguments": {"code": "print(6*7)"},
        "output": {"stdout": "42"},
    }


@pytest.mark.asyncio
async def test_openai_provider_maps_reasoning_schema_and_function_tools_strictly(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_text('{"answer":"42"}')

    await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(model_request())

    sent = fake_responses_client.requests[0]
    assert sent == {
        "model": "gpt-5.6-sol",
        "instructions": "answer with JSON",
        "input": ({"role": "user", "content": "What is six times seven?"},),
        "tools": [
            {
                "type": "function",
                "name": "execute_python",
                "description": "Run Python code",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        "text": {"format": {"type": "json_object"}},
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 500,
        "max_tool_calls": 4,
    }


@pytest.mark.asyncio
async def test_openai_provider_aggregates_usage_and_extracts_sources(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call(
        "execute_python", "{}", "call-1", usage=_usage(10, 4, 3)
    )
    fake_responses_client.queue_text(
        '{"answer":"source-backed"}',
        annotations=[
            {
                "type": "url_citation",
                "url": "https://example.test/source",
                "title": "Primary source",
                "start_index": 0,
                "end_index": 6,
            },
            {"type": "file_citation", "file_id": "file-123", "filename": "source.pdf", "index": 0},
            {"type": "file_path", "file_id": "file-generated", "index": 8},
        ],
        usage=_usage(7, 5, 2),
    )

    response = await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
        model_request(), RecordingExecutor({"ok": True})
    )

    assert response.usage == Usage(
        input_tokens=17, output_tokens=9, reasoning_tokens=5, total_tokens=26
    )
    assert [citation.model_dump() for citation in response.citations] == [
        {
            "kind": "url",
            "title": "Primary source",
            "url": "https://example.test/source",
            "file_id": None,
            "filename": None,
            "start_index": 0,
            "end_index": 6,
        },
        {
            "kind": "file",
            "title": None,
            "url": None,
            "file_id": "file-123",
            "filename": "source.pdf",
            "start_index": None,
            "end_index": None,
        },
        {
            "kind": "file",
            "title": None,
            "url": None,
            "file_id": "file-generated",
            "filename": None,
            "start_index": None,
            "end_index": None,
        },
    ]
    assert response.remote_file_ids == ("file-123", "file-generated")


def _status_error(error_type: type[Exception], status_code: int) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code, request=request)
    return error_type("synthetic failure", response=response, body={})  # type: ignore[call-arg]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transient_error",
    [
        _status_error(RateLimitError, 429),
        APITimeoutError(httpx.Request("POST", "https://api.openai.com/v1/responses")),
        _status_error(InternalServerError, 500),
    ],
)
async def test_openai_provider_retries_transient_failures_once(transient_error: Exception) -> None:
    client = FakeResponsesClient()
    client.queued.extend([transient_error, _response("resp-retry", [])])

    await OpenAIProvider(client, "gpt-5.6-sol").complete(model_request())

    assert len(client.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_error",
    [_status_error(AuthenticationError, 401), _status_error(BadRequestError, 400)],
)
async def test_openai_provider_does_not_retry_terminal_failures(terminal_error: Exception) -> None:
    client = FakeResponsesClient()
    client.queued.append(terminal_error)

    with pytest.raises(type(terminal_error)):
        await OpenAIProvider(client, "gpt-5.6-sol").complete(model_request())

    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_openai_provider_maps_refusal_without_attempting_json_parse(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_refusal("I cannot help with that request.")

    response = await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(model_request())

    assert response.parsed is None
    assert response.output_text == "I cannot help with that request."


@pytest.mark.asyncio
async def test_openai_provider_enforces_global_tool_call_budget(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("execute_python", "{}", "call-1")

    with pytest.raises(RuntimeError, match="tool-call budget exhausted"):
        await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
            model_request(max_tool_calls=0), RecordingExecutor({})
        )


@pytest.mark.asyncio
async def test_openai_provider_rejects_non_object_function_arguments(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("execute_python", "[]", "call-1")

    with pytest.raises(TypeError, match="arguments must decode to an object"):
        await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
            model_request(), RecordingExecutor({})
        )


@pytest.mark.asyncio
async def test_openai_provider_rejects_undeclared_function_before_execution(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("untrusted_tool", "{}", "call-1")
    executor = RecordingExecutor({})

    with pytest.raises(ValueError, match="undeclared tool: untrusted_tool"):
        await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
            model_request(), executor
        )

    assert executor.calls == []


def test_provider_request_is_frozen() -> None:
    request = model_request()

    with pytest.raises(ValidationError, match="frozen"):
        request.reasoning_effort = "high"
