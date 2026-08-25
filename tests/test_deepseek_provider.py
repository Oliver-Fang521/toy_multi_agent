import asyncio
import json
from collections import deque
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, RateLimitError

from gaia_agent.models import Usage
from gaia_agent.providers import DeepSeekProvider, ModelRequest, ToolSpec
from gaia_agent.reliability import RetryPolicy


def test_deepseek_provider_declares_no_native_vision_support() -> None:
    assert DeepSeekProvider.supports_vision is False


class FakeChatClient:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.queued: deque[Any] = deque()
        self.requests: list[dict[str, Any]] = []
        self.configured_max_retries: list[int] = []

    def with_options(self, *, max_retries: int) -> "FakeChatClient":
        self.configured_max_retries.append(max_retries)
        return self

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        value = self.queued.popleft()
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def fake_chat_client() -> FakeChatClient:
    return FakeChatClient()


def _status_error(error_type: type[Exception], status_code: int) -> Exception:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_type("synthetic failure", response=response, body={})  # type: ignore[call-arg]


def _chat_message(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    reasoning_content: str | None = None,
    refusal: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        refusal=refusal,
        reasoning_content=reasoning_content,
    )


def _chat_response(message: Any, usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _usage(prompt: int = 0, completion: int = 0, reasoning: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def model_request(**updates: Any) -> ModelRequest:
    values: dict[str, Any] = {
        "instructions": "answer with JSON",
        "input_items": ({"role": "user", "content": "What is six times seven?"},),
        "output_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
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


class RecordingExecutor:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return self.result


class _RecordingGeneration:
    def __init__(self) -> None:
        self.ended: list[dict[str, Any]] = []

    def end(self, **kwargs: Any) -> None:
        self.ended.append(kwargs)


class _RecordingTrace:
    def __init__(self) -> None:
        self.generation_span = _RecordingGeneration()

    def generation(self, name: str, metadata: dict[str, object]) -> _RecordingGeneration:
        assert name == "provider-generation"
        assert "prompt" not in metadata
        return self.generation_span


class _RecordingObservability:
    def __init__(self) -> None:
        self.trace = _RecordingTrace()

    def current_trace(self) -> _RecordingTrace:
        return self.trace


@pytest.mark.asyncio
async def test_deepseek_provider_executes_tool_loop_and_echoes_reasoning(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(
        _chat_response(
            _chat_message(
                content="",
                tool_calls=[
                    _tool_call("call-1", "execute_python", '{"code":"print(6*7)"}'),
                    _tool_call("call-2", "execute_python", '{"code":"print(8*8)"}'),
                ],
                reasoning_content="I should compute both products.",
            ),
            usage=_usage(10, 5, 4),
        )
    )
    fake_chat_client.queued.append(
        _chat_response(
            _chat_message(content='Here is the result:\n\n```json\n{"answer":"42"}\n```'),
            usage=_usage(20, 8, 6),
        )
    )
    executor = RecordingExecutor({"stdout": "42"})
    provider = DeepSeekProvider(client=fake_chat_client, model="deepseek-v4-flash")
    observed_usage: list[Usage] = []

    response = await provider.complete(
        model_request(), executor, usage_observer=observed_usage.append
    )

    assert executor.calls == [
        ("execute_python", {"code": "print(6*7)"}),
        ("execute_python", {"code": "print(8*8)"}),
    ]
    first = fake_chat_client.requests[0]
    assert first["messages"] == [
        {
            "role": "system",
            "content": (
                "answer with JSON\n\nRespond with a single JSON object only, conforming "
                "exactly to this schema:\n"
                + json.dumps(
                    {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    sort_keys=True,
                )
            ),
        },
        {"role": "user", "content": "What is six times seven?"},
    ]
    assert first["response_format"] == {"type": "json_object"}
    assert first["max_tokens"] == 6000
    assert first["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Run Python code",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    second_messages = fake_chat_client.requests[1]["messages"]
    assert second_messages[2] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should compute both products.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "execute_python", "arguments": '{"code":"print(6*7)"}'},
            },
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "execute_python", "arguments": '{"code":"print(8*8)"}'},
            },
        ],
    }
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"stdout":"42"}',
    }
    assert second_messages[4] == {
        "role": "tool",
        "tool_call_id": "call-2",
        "content": '{"stdout":"42"}',
    }
    assert response.parsed == {"answer": "42"}
    assert response.usage == Usage(input_tokens=30, output_tokens=13, reasoning_tokens=10, total_tokens=43)
    assert observed_usage == [
        Usage(input_tokens=10, output_tokens=5, reasoning_tokens=4, total_tokens=15),
        Usage(input_tokens=20, output_tokens=8, reasoning_tokens=6, total_tokens=28),
    ]
    assert len(response.tool_events) == 2


@pytest.mark.asyncio
async def test_deepseek_provider_publishes_completed_usage_before_cancelled_tool_call(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(
        _chat_response(
            _chat_message(
                content="",
                tool_calls=[
                    _tool_call("call-1", "execute_python", '{"code":"print(42)"}')
                ],
            ),
            usage=_usage(6, 3, 2),
        )
    )
    tool_started = asyncio.Event()

    class BlockingExecutor:
        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            assert name == "execute_python"
            assert arguments == {"code": "print(42)"}
            tool_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    observed_usage: list[Usage] = []
    recorder = _RecordingObservability()
    invocation = asyncio.create_task(
        DeepSeekProvider(
            fake_chat_client,
            "deepseek-v4-flash",
            observability=recorder,
        ).complete(
            model_request(),
            BlockingExecutor(),
            usage_observer=observed_usage.append,
        )
    )
    await asyncio.sleep(0)
    if invocation.done():
        await invocation
    await asyncio.wait_for(tool_started.wait(), timeout=0.05)

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert observed_usage == [
        Usage(input_tokens=6, output_tokens=3, reasoning_tokens=2, total_tokens=9)
    ]
    assert recorder.trace.generation_span.ended[-1]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_deepseek_provider_does_not_retry_terminal_errors(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(_status_error(BadRequestError, 400))

    with pytest.raises(BadRequestError):
        await DeepSeekProvider(fake_chat_client, "deepseek-v4-flash").complete(model_request())

    assert len(fake_chat_client.requests) == 1


@pytest.mark.asyncio
async def test_deepseek_provider_retries_transient_error_once(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(_status_error(RateLimitError, 429))
    fake_chat_client.queued.append(
        _chat_response(
            _chat_message(content='{"answer":"42"}'),
            usage=_usage(4, 2, 1),
        )
    )
    observed_usage: list[Usage] = []

    response = await DeepSeekProvider(fake_chat_client, "deepseek-v4-flash").complete(
        model_request(), usage_observer=observed_usage.append
    )

    assert response.parsed == {"answer": "42"}
    assert len(fake_chat_client.requests) == 2
    assert observed_usage == [
        Usage(input_tokens=4, output_tokens=2, reasoning_tokens=1, total_tokens=6)
    ]


@pytest.mark.asyncio
async def test_deepseek_provider_emits_bounded_retry_attempt_and_decision_metadata(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.extend(
        [
            _status_error(RateLimitError, 429),
            _chat_response(_chat_message(content='{"answer":"42"}')),
        ]
    )
    observability = _RecordingObservability()

    async def sleep(_delay: float) -> None:
        return None

    await DeepSeekProvider(
        fake_chat_client,
        "deepseek-v4-flash",
        observability=observability,  # type: ignore[arg-type]
        sleep=sleep,
        jitter=lambda: 0.0,
    ).complete(model_request())

    ended = observability.trace.generation_span.ended[-1]
    assert ended["metadata"] == {
        "parsed": True,
        "refused": False,
        "tool_call_count": 0,
        "attempt_number": 2,
        "provider_attempt_count": 2,
        "provider_retry_count": 1,
        "retry_category": "RateLimitError",
        "retry_decision": "recovered",
    }
    assert ended["summary"] == {
        "decision": "parsed",
        "parsed": True,
        "refused": False,
        "tool_call_count": 0,
        "attempt_number": 2,
        "provider_attempt_count": 2,
        "provider_retry_count": 1,
        "retry_category": "RateLimitError",
        "retry_decision": "recovered",
    }


@pytest.mark.asyncio
async def test_deepseek_provider_applies_configured_retry_attempts_and_backoff(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.extend(
        [
            APIConnectionError(
                request=httpx.Request("POST", "https://api.deepseek.com/chat/completions")
            ),
            _status_error(RateLimitError, 429),
            _chat_response(_chat_message(content='{"answer":"42"}')),
        ]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    response = await DeepSeekProvider(
        fake_chat_client,
        "deepseek-v4-flash",
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=1.25),
        sleep=sleep,
        jitter=lambda: 0.0,
    ).complete(model_request())

    assert response.parsed == {"answer": "42"}
    assert len(fake_chat_client.requests) == 3
    assert delays == [1.25, 2.5]


@pytest.mark.asyncio
async def test_deepseek_provider_rejects_undeclared_tool(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(
        _chat_response(
            _chat_message(
                content="",
                tool_calls=[_tool_call("call-1", "rm_rf", "{}")],
            )
        )
    )

    with pytest.raises(ValueError, match="undeclared tool"):
        await DeepSeekProvider(fake_chat_client, "deepseek-v4-flash").complete(
            model_request(), RecordingExecutor({})
        )

    assert len(fake_chat_client.requests) == 1


@pytest.mark.asyncio
async def test_deepseek_provider_maps_refusal_to_none(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(
        _chat_response(_chat_message(content=None, refusal="I cannot help with that."))
    )

    response = await DeepSeekProvider(fake_chat_client, "deepseek-v4-flash").complete(model_request())

    assert response.parsed is None
    assert response.output_text == "I cannot help with that."


@pytest.mark.asyncio
async def test_deepseek_provider_recovers_once_from_wrong_schema_response(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(
        _chat_response(_chat_message(content='{"wrong_key": "nope"}'))
    )
    fake_chat_client.queued.append(
        _chat_response(_chat_message(content='{"answer": "42"}'))
    )

    response = await DeepSeekProvider(fake_chat_client, "deepseek-v4-flash").complete(
        model_request()
    )

    assert response.parsed == {"answer": "42"}
    assert len(fake_chat_client.requests) == 2
    recovery_messages = fake_chat_client.requests[1]["messages"]
    assert recovery_messages[-1]["role"] == "user"
    assert "exactly matching the schema" in recovery_messages[-1]["content"]


@pytest.mark.asyncio
async def test_deepseek_provider_does_not_recover_more_than_once(
    fake_chat_client: FakeChatClient,
) -> None:
    fake_chat_client.queued.append(
        _chat_response(_chat_message(content="no usable json here"))
    )
    fake_chat_client.queued.append(
        _chat_response(_chat_message(content="still not json"))
    )

    response = await DeepSeekProvider(fake_chat_client, "deepseek-v4-flash").complete(
        model_request()
    )

    assert response.parsed is None
    assert len(fake_chat_client.requests) == 2


@pytest.mark.asyncio
async def test_deepseek_provider_recovers_from_nested_schema_violation(
    fake_chat_client: FakeChatClient,
) -> None:
    nested_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/Item"},
            },
        },
        "required": ["answer", "items"],
        "additionalProperties": False,
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            }
        },
    }
    fake_chat_client.queued.append(
        _chat_response(
            _chat_message(
                content='{"answer":"x","items":[{"value":"a"},{"bad":1}]}'
            )
        )
    )
    fake_chat_client.queued.append(
        _chat_response(
            _chat_message(
                content='{"answer":"x","items":[{"value":"a"},{"value":"b"}]}'
            )
        )
    )

    response = await DeepSeekProvider(fake_chat_client, "deepseek-v4-flash").complete(
        model_request(output_schema=nested_schema)
    )

    assert response.parsed == {"answer": "x", "items": [{"value": "a"}, {"value": "b"}]}
    assert len(fake_chat_client.requests) == 2
