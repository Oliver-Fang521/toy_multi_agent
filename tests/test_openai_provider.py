import asyncio
import json
from collections import deque
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp import Client
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from openai.types.responses import Response
from pydantic import ValidationError

from gaia_agent import mcp_server, providers
from gaia_agent.agents import EvidenceLinkerAgent
from gaia_agent.mcp_client import ToolExecutionError
from gaia_agent.mcp_server import AttachmentReader, MCPServerSettings, build_server
from gaia_agent.models import GaiaTask, Usage
from gaia_agent.observability import Observability, ObservabilityConfig
from gaia_agent.orchestrator import _RepeatedToolCall, _ToolBudgetExhausted
from gaia_agent.providers import ModelRequest, OpenAIProvider, ToolSpec
from gaia_agent.reliability import RetryPolicy


def test_openai_provider_declares_native_vision_support() -> None:
    assert OpenAIProvider.supports_vision is True


@pytest.mark.asyncio
async def test_evidence_linker_schema_uses_openai_strict_supported_subset(
    fake_responses_client,
) -> None:
    fake_responses_client.queue_text('{"evidence_entry_hashes":["known"]}')
    provider = OpenAIProvider(fake_responses_client, "gpt-5.6-sol")
    linker = EvidenceLinkerAgent(provider)
    task = GaiaTask(
        task_id="schema-task",
        question="Which card supports the candidate?",
        level=1,
        final_answer="hidden",
    )

    selected = await linker.link(
        task,
        "candidate",
        ({"entry_hash": "known", "excerpt": "support"},),
    )

    assert selected == ("known",)
    assert fake_responses_client.requests[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "gaia_response",
            "schema": {
                "type": "object",
                "properties": {
                    "evidence_entry_hashes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    }
                },
                "required": ["evidence_entry_hashes"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    }


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
        self.configured_max_retries: list[int] = []

    def with_options(self, *, max_retries: int) -> "FakeResponsesClient":
        self.configured_max_retries.append(max_retries)
        return self

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


class _TraceObservation:
    def __init__(self, name: str, kwargs: dict[str, Any]) -> None:
        self.name = name
        self.kwargs = kwargs
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _TraceContext:
    def __init__(self, observation: _TraceObservation) -> None:
        self.observation = observation

    def __enter__(self) -> _TraceObservation:
        return self.observation

    def __exit__(self, *_args: object) -> None:
        return None


class _TraceClient:
    def __init__(self) -> None:
        self.observations: list[_TraceObservation] = []

    def start_as_current_observation(self, *, name: str, as_type: str, **kwargs: Any):
        observation = _TraceObservation(name, {"as_type": as_type, **kwargs})
        self.observations.append(observation)
        return _TraceContext(observation)


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
async def test_openai_provider_records_safe_generation_metadata(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_text('{"answer":"42"}', usage=_usage(3, 4, 1))
    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "task-1",
        level=1,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        run_index=0,
        evaluation_mode="test",
    )
    response = await OpenAIProvider(
        fake_responses_client,
        model="gpt-5.6-sol",
        observability=observability,
    ).complete(model_request())
    trace.close()

    assert response.parsed == {"answer": "42"}
    generation = trace_client.observations[-1]
    assert generation.name == "provider-generation"
    assert generation.kwargs["metadata"] == {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
    }
    assert generation.updates[-1]["metadata"] == {
        "parsed": True,
        "refused": False,
        "tool_call_count": 0,
        "attempt_number": 1,
        "provider_attempt_count": 1,
        "provider_retry_count": 0,
        "retry_decision": "not_needed",
        "status": "ok",
    }
    assert generation.updates[-1]["output"]["reasoning_summary"] == {
        "decision": "parsed",
        "parsed": True,
        "refused": False,
        "tool_call_count": 0,
        "attempt_number": 1,
        "provider_attempt_count": 1,
        "provider_retry_count": 0,
        "retry_decision": "not_needed",
    }
    assert generation.updates[-1]["usage_details"] == {
        "input_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 1,
        "total_tokens": 7,
    }


@pytest.mark.asyncio
async def test_openai_provider_records_safe_generation_retry_metadata() -> None:
    client = FakeResponsesClient()
    client.queued.extend(
        [
            APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
            _response("resp-retry-metadata", []),
        ]
    )
    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "task-1",
        level=1,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        run_index=0,
        evaluation_mode="test",
    )

    async def sleep(_delay: float) -> None:
        return None

    await OpenAIProvider(
        client,
        model="gpt-5.6-sol",
        observability=observability,
        sleep=sleep,
        jitter=lambda: 0.0,
    ).complete(model_request())
    trace.close()

    generation = trace_client.observations[-1]
    assert generation.updates[-1]["metadata"] == {
        "parsed": False,
        "refused": False,
        "tool_call_count": 0,
        "attempt_number": 2,
        "provider_attempt_count": 2,
        "provider_retry_count": 1,
        "retry_category": "APIConnectionError",
        "retry_decision": "recovered",
        "status": "ok",
    }
    assert generation.updates[-1]["output"]["reasoning_summary"] == {
        "decision": "unparsed",
        "parsed": False,
        "refused": False,
        "tool_call_count": 0,
        "attempt_number": 2,
        "provider_attempt_count": 2,
        "provider_retry_count": 1,
        "retry_category": "APIConnectionError",
        "retry_decision": "recovered",
    }
@pytest.mark.asyncio
async def test_openai_provider_stateless_loop_resends_full_history_without_previous_id(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("execute_python", '{"code":"print(6*7)"}', "call-1")
    fake_responses_client.queue_text('{"answer":"42"}')
    executor = RecordingExecutor({"stdout": "42"})
    provider = OpenAIProvider(
        client=fake_responses_client,
        model="gpt-5.6-sol",
        use_previous_response_id=False,
    )

    response = await provider.complete(model_request(), executor)

    assert executor.calls == [("execute_python", {"code": "print(6*7)"})]
    continuation = fake_responses_client.requests[1]
    assert "previous_response_id" not in continuation
    assert continuation["input"] == [
        {"role": "user", "content": "What is six times seven?"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "execute_python",
            "arguments": '{"code":"print(6*7)"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": '{"stdout":"42"}'},
    ]
    assert response.parsed == {"answer": "42"}


@pytest.mark.asyncio
async def test_stateless_loop_echoes_reasoning_and_groups_parallel_calls(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queued.append(
        _response(
            "resp-1",
            [
                {
                    "type": "reasoning",
                    "id": "rs-1",
                    "status": "completed",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan the work"}],
                },
                {
                    "type": "function_call",
                    "id": "fc-1",
                    "call_id": "call-1",
                    "name": "execute_python",
                    "arguments": "{}",
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-2",
                    "call_id": "call-2",
                    "name": "execute_python",
                    "arguments": "{}",
                    "status": "completed",
                },
            ],
        )
    )
    fake_responses_client.queue_text('{"answer":"42"}')
    provider = OpenAIProvider(
        client=fake_responses_client,
        model="gpt-5.6-sol",
        use_previous_response_id=False,
    )

    await provider.complete(model_request(), RecordingExecutor({"stdout": "42"}))

    continuation = fake_responses_client.requests[1]
    assert continuation["input"] == [
        {"role": "user", "content": "What is six times seven?"},
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "plan the work"}],
        },
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "execute_python",
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "call_id": "call-2",
            "name": "execute_python",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": '{"stdout":"42"}'},
        {"type": "function_call_output", "call_id": "call-2", "output": '{"stdout":"42"}'},
    ]


@pytest.mark.asyncio
async def test_unicode_web_output_stays_bounded_through_server_and_provider(
    tmp_path,
    fake_responses_client: FakeResponsesClient,
) -> None:
    secret = "sk-private-unicode-secret"
    usage = SimpleNamespace(
        input_tokens=101,
        output_tokens=17,
        total_tokens=118,
        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )

    class UnicodeWebBackend:
        async def web_search(self, query: str) -> dict[str, Any]:
            return mcp_server._with_response_usage(
                {
                    "answer": secret + "漢字🚀" * 2_400,
                    "sources": [
                        {
                            "url": f"https://source-{index}.example/article",
                            "title": secret + "來源🚀" * 40,
                        }
                        for index in range(12)
                    ],
                },
                SimpleNamespace(usage=usage),
            )

    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=UnicodeWebBackend(),
        api_key=secret,
        hf_token="",
    )
    async with Client(build_server(settings)) as client:
        tool_result = await client.call_tool("web_search", {"query": "unicode"})
    assert isinstance(tool_result.structured_content, dict)
    public_result = dict(tool_result.structured_content)

    class PoppingExecutor:
        def __init__(self) -> None:
            self.usage: dict[str, Any] | None = None
            self.output: dict[str, Any] | None = None

        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.output = dict(public_result)
            self.usage = self.output.pop("__gaia_usage__")
            return self.output

    web_tool = ToolSpec(
        name="web_search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    fake_responses_client.queue_function_call(
        "web_search", '{"query":"unicode"}', "call-unicode"
    )
    fake_responses_client.queue_text('{"answer":"done"}')
    executor = PoppingExecutor()

    response = await OpenAIProvider(
        fake_responses_client, "gpt-5.6-sol"
    ).complete(
        model_request(allowed_tools=(web_tool,), max_tool_calls=1),
        executor,
    )

    server_bytes = json.dumps(
        public_result, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    wire_output = fake_responses_client.requests[1]["input"][0]["output"]
    assert len(server_bytes) <= 16 * 1024
    assert len(wire_output.encode("utf-8")) <= 16 * 1024
    assert json.loads(wire_output) == executor.output
    assert executor.usage == {
        "input_tokens": 101,
        "output_tokens": 17,
        "reasoning_tokens": 3,
        "total_tokens": 118,
    }
    assert response.parsed == {"answer": "done"}
    assert public_result["answer"].startswith("[REDACTED]漢字🚀")
    assert len(public_result["sources"]) <= 8
    assert secret not in repr(public_result)
    assert secret not in wire_output


def test_mcp_tool_execution_error_uses_provider_neutral_recoverable_contract() -> None:
    tool_call_error = getattr(providers, "ToolCallError", None)

    assert tool_call_error is not None
    assert issubclass(ToolExecutionError, tool_call_error)


@pytest.mark.asyncio
async def test_openai_provider_recovers_from_tool_failure_without_leaking_message(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("execute_python", '{"code":"bad"}', "call-1")
    fake_responses_client.queue_function_call(
        "web_search", '{"query":"safe fallback"}', "call-2"
    )
    fake_responses_client.queue_text('{"answer":"recovered"}')
    web_tool = ToolSpec(
        name="web_search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    class RecoveringExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((name, arguments))
            if name == "execute_python":
                error = ToolExecutionError("sk-super-secret-tool-message")
                error.category = "timeout"
                error.retryable = True
                error.provider = "serper"
                error.attempt_count = 2
                error.http_status = 503
                raise error
            return {"answer": "fallback succeeded", "sources": []}

    executor = RecoveringExecutor()
    request = model_request(
        allowed_tools=(*model_request().allowed_tools, web_tool),
        max_tool_calls=2,
    )

    response = await OpenAIProvider(
        fake_responses_client, "gpt-5.6-sol"
    ).complete(request, executor)

    assert response.parsed == {"answer": "recovered"}
    assert executor.calls == [
        ("execute_python", {"code": "bad"}),
        ("web_search", {"query": "safe fallback"}),
    ]
    assert json.loads(fake_responses_client.requests[1]["input"][0]["output"]) == {
        "error": "tool_failed",
        "category": "timeout",
        "retryable": True,
        "provider": "serper",
        "attempt_count": 2,
        "http_status": 503,
    }
    assert fake_responses_client.requests[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": (
                '{"error":"tool_failed","category":"timeout","retryable":true,'
                '"provider":"serper","attempt_count":2,"http_status":503}'
            ),
        }
    ]
    assert "sk-super-secret-tool-message" not in repr(fake_responses_client.requests[1])
    assert [event.output for event in response.tool_events] == [
        {
            "error": "tool_failed",
            "category": "timeout",
            "retryable": True,
            "provider": "serper",
            "attempt_count": 2,
            "http_status": 503,
        },
        {"answer": "fallback succeeded", "sources": []},
    ]


@pytest.mark.asyncio
async def test_openai_provider_finalizes_after_recoverable_failure_reaches_tool_budget(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("execute_python", "{}", "call-1")
    fake_responses_client.queue_function_call("execute_python", "{}", "call-2")
    fake_responses_client.queue_text('{"answer":"budget-finalized"}')

    class FailingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            raise ToolExecutionError("recoverable failure")

    executor = FailingExecutor()

    response = await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
        model_request(max_tool_calls=1), executor
    )

    assert response.parsed == {"answer": "budget-finalized"}
    assert executor.calls == 1
    assert fake_responses_client.requests[1]["input"][0]["output"] == (
        '{"error":"tool_failed","category":"tool_error","retryable":false,'
        '"provider":null,"attempt_count":1}'
    )
    assert fake_responses_client.requests[1]["tools"] == []
    assert "max_tool_calls" not in fake_responses_client.requests[1]
    assert fake_responses_client.requests[2]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": '{"error":"tool_budget_exhausted"}',
        }
    ]
    assert len(fake_responses_client.requests) == 3


@pytest.mark.asyncio
async def test_openai_provider_proactively_finalizes_when_executed_batch_reaches_cap() -> None:
    client = FakeResponsesClient()
    client.queued.append(
        _response(
            "resp-exact-cap",
            [
                {
                    "type": "function_call",
                    "id": "fc-call-1",
                    "call_id": "call-1",
                    "name": "execute_python",
                    "arguments": '{"code":"one"}',
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc-call-2",
                    "call_id": "call-2",
                    "name": "execute_python",
                    "arguments": '{"code":"two"}',
                    "status": "completed",
                },
            ],
            usage=_usage(10, 4, 1),
        )
    )
    client.queue_text('{"answer":"finalized"}', usage=_usage(3, 2, 0))
    executor = RecordingExecutor({"stdout": "observable"})

    response = await OpenAIProvider(client, "gpt-5.6-sol").complete(
        model_request(max_tool_calls=2), executor
    )

    assert response.parsed == {"answer": "finalized"}
    assert response.usage.total_tokens == 19
    assert executor.calls == [
        ("execute_python", {"code": "one"}),
        ("execute_python", {"code": "two"}),
    ]
    assert [event.call_id for event in response.tool_events] == ["call-1", "call-2"]
    finalization = client.requests[1]
    assert finalization["previous_response_id"] == "resp-exact-cap"
    assert finalization["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"stdout":"observable"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": '{"stdout":"observable"}',
        },
    ]
    assert finalization["tools"] == []
    assert "max_tool_calls" not in finalization
    assert len(client.requests) == 2


@pytest.mark.asyncio
async def test_openai_provider_does_not_swallow_nonrecoverable_executor_error(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call("execute_python", "{}", "call-1")

    class BrokenExecutor:
        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("tool budget sentinel")

    with pytest.raises(RuntimeError, match="tool budget sentinel"):
        await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
            model_request(), BrokenExecutor()
        )

    assert len(fake_responses_client.requests) == 1


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
        "text": {
            "format": {
                "type": "json_schema",
                "name": "gaia_response",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 500,
        "max_tool_calls": 4,
    }


@pytest.mark.asyncio
async def test_openai_provider_omits_zero_tool_call_limit_for_tool_free_request(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_text('{"answer":"42"}')

    await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
        model_request(allowed_tools=(), max_tool_calls=0)
    )

    sent = fake_responses_client.requests[0]
    assert sent["tools"] == []
    assert "max_tool_calls" not in sent


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

    observed_usage: list[Usage] = []
    response = await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
        model_request(),
        RecordingExecutor({"ok": True}),
        usage_observer=observed_usage.append,
    )

    assert response.usage == Usage(
        input_tokens=17, output_tokens=9, reasoning_tokens=5, total_tokens=26
    )
    assert observed_usage == [
        Usage(input_tokens=10, output_tokens=4, reasoning_tokens=3, total_tokens=14),
        Usage(input_tokens=7, output_tokens=5, reasoning_tokens=2, total_tokens=12),
    ]
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


@pytest.mark.asyncio
async def test_openai_provider_publishes_completed_usage_before_cancelled_tool_call(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_function_call(
        "execute_python", "{}", "call-1", usage=_usage(5, 2, 1)
    )
    tool_started = asyncio.Event()

    class BlockingExecutor:
        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            assert name == "execute_python"
            assert arguments == {}
            tool_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    observed_usage: list[Usage] = []
    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "cancel-openai",
        level=1,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        evaluation_mode="test",
    )
    provider = OpenAIProvider(fake_responses_client, "gpt-5.6-sol", observability=observability)
    invocation = asyncio.create_task(
        provider.complete(
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
        Usage(input_tokens=5, output_tokens=2, reasoning_tokens=1, total_tokens=7)
    ]
    generation = trace_client.observations[-1]
    assert generation.updates[-1]["metadata"]["status"] == "cancelled"
    trace.close()


@pytest.mark.asyncio
async def test_openai_provider_usage_observers_are_invocation_local_under_concurrency() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class InterleavingResponsesClient:
        def __init__(self) -> None:
            self.responses = self

        def with_options(self, *, max_retries: int) -> "InterleavingResponsesClient":
            assert max_retries == 0
            return self

        async def create(self, **kwargs: Any) -> Response:
            if kwargs["instructions"] == "first":
                first_started.set()
                await release_first.wait()
                return _response("first", [], usage=_usage(3, 1, 0))
            await first_started.wait()
            release_first.set()
            return _response("second", [], usage=_usage(5, 2, 1))

    provider = OpenAIProvider(InterleavingResponsesClient(), "gpt-5.6-sol")
    first_usage: list[Usage] = []
    second_usage: list[Usage] = []

    first, second = await asyncio.gather(
        provider.complete(
            model_request(instructions="first", allowed_tools=(), max_tool_calls=0),
            usage_observer=first_usage.append,
        ),
        provider.complete(
            model_request(instructions="second", allowed_tools=(), max_tool_calls=0),
            usage_observer=second_usage.append,
        ),
    )

    assert first.usage == Usage(input_tokens=3, output_tokens=1, total_tokens=4)
    assert second.usage == Usage(
        input_tokens=5, output_tokens=2, reasoning_tokens=1, total_tokens=7
    )
    assert first_usage == [first.usage]
    assert second_usage == [second.usage]


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
    client.queued.extend(
        [transient_error, _response("resp-retry", [], usage=_usage(4, 2, 1))]
    )
    observed_usage: list[Usage] = []

    await OpenAIProvider(client, "gpt-5.6-sol").complete(
        model_request(), usage_observer=observed_usage.append
    )

    assert len(client.requests) == 2
    assert observed_usage == [
        Usage(input_tokens=4, output_tokens=2, reasoning_tokens=1, total_tokens=6)
    ]


@pytest.mark.asyncio
async def test_openai_provider_retries_initial_connection_error_in_place() -> None:
    client = FakeResponsesClient()
    client.queued.extend(
        [
            APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
            _response("resp-retry", []),
        ]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    await OpenAIProvider(
        client,
        "gpt-5.6-sol",
        retry_policy=RetryPolicy(max_attempts=4),
        sleep=sleep,
        jitter=lambda: 0.0,
    ).complete(model_request())

    assert len(client.requests) == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_openai_provider_raises_typed_safe_cause_after_transient_attempts_exhaust() -> None:
    from gaia_agent.reliability import ProviderTransientExhaustedError

    client = FakeResponsesClient()
    client.queued.extend(
        APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/responses")
        )
        for _ in range(4)
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(ProviderTransientExhaustedError) as caught:
        await OpenAIProvider(
            client,
            "gpt-5.6-sol",
            retry_policy=RetryPolicy(max_attempts=4),
            sleep=sleep,
            jitter=lambda: 0.0,
        ).complete(model_request())

    assert caught.value.attempt_count == 4
    assert caught.value.retry_category == "APIConnectionError"
    assert str(caught.value) == "provider transient attempts exhausted"
    assert len(client.requests) == 4
    assert delays == [0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_openai_provider_retries_failed_continuation_without_reexecuting_tool() -> None:
    client = FakeResponsesClient()
    client.queue_function_call("execute_python", '{"code":"print(6*7)"}', "call-1")
    client.queued.append(
        APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    )
    client.queue_text('{"answer":"42"}')
    executor = RecordingExecutor({"stdout": "42"})
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    response = await OpenAIProvider(
        client,
        "gpt-5.6-sol",
        retry_policy=RetryPolicy(max_attempts=4),
        sleep=sleep,
        jitter=lambda: 0.0,
    ).complete(model_request(), executor)

    assert len(client.requests) == 3
    assert executor.calls == [("execute_python", {"code": "print(6*7)"})]
    assert response.parsed == {"answer": "42"}
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_openai_provider_disables_sdk_internal_retries() -> None:
    client = FakeResponsesClient()
    client.queue_text('{"answer":"42"}')

    await OpenAIProvider(client, "gpt-5.6-sol").complete(model_request())

    assert client.configured_max_retries == [0]


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
async def test_openai_provider_extracts_json_wrapped_in_markdown_prose(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_text(
        'Based on my search I can answer.\n\n```json\n{"answer": "42"}\n```\n\nHope this helps.'
    )

    response = await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(model_request())

    assert response.parsed == {"answer": "42"}


@pytest.mark.asyncio
async def test_openai_provider_returns_none_for_non_json_prose(
    fake_responses_client: FakeResponsesClient,
) -> None:
    fake_responses_client.queue_text("I looked but could not find a usable answer.")

    response = await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(model_request())

    assert response.parsed is None


@pytest.mark.asyncio
async def test_openai_provider_finalizes_structured_output_without_executing_over_cap_call(
    fake_responses_client: FakeResponsesClient,
) -> None:
    secret = "private-over-budget-arguments"
    fake_responses_client.queue_function_call(
        "execute_python",
        json.dumps({"code": secret}),
        "call-1",
        usage=_usage(7, 3, 1),
    )
    fake_responses_client.queue_text(
        '{"answer":"bounded"}',
        usage=_usage(5, 2, 0),
    )
    executor = RecordingExecutor({"stdout": "must not execute"})

    response = await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
        model_request(max_tool_calls=0), executor
    )

    assert response.parsed == {"answer": "bounded"}
    assert response.usage.total_tokens == 17
    assert response.tool_events == ()
    assert executor.calls == []
    finalization = fake_responses_client.requests[1]
    assert finalization["previous_response_id"] == "resp-1"
    assert finalization["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"error":"tool_budget_exhausted"}',
        }
    ]
    assert finalization["tools"] == []
    assert "max_tool_calls" not in finalization
    assert finalization["text"] == fake_responses_client.requests[0]["text"]
    assert secret not in repr(finalization)


@pytest.mark.asyncio
async def test_openai_provider_propagates_budget_finalization_api_error() -> None:
    client = FakeResponsesClient()
    client.queue_function_call("execute_python", "{}", "call-1")
    client.queued.append(_status_error(BadRequestError, 400))

    with pytest.raises(BadRequestError):
        await OpenAIProvider(client, "gpt-5.6-sol").complete(
            model_request(max_tool_calls=0), RecordingExecutor({})
        )

    assert len(client.requests) == 2
    assert client.requests[1]["tools"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_error",
    [
        _ToolBudgetExhausted("global tool-call budget exhausted"),
        _RepeatedToolCall("repeated tool call"),
    ],
)
async def test_openai_provider_does_not_swallow_global_tool_guards(
    fake_responses_client: FakeResponsesClient,
    terminal_error: RuntimeError,
) -> None:
    fake_responses_client.queue_function_call("execute_python", "{}", "call-1")

    class GuardedExecutor:
        async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise terminal_error

    with pytest.raises(type(terminal_error), match=str(terminal_error)):
        await OpenAIProvider(fake_responses_client, "gpt-5.6-sol").complete(
            model_request(max_tool_calls=1), GuardedExecutor()
        )

    assert len(fake_responses_client.requests) == 1


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


@pytest.mark.asyncio
async def test_openai_provider_preserves_sources_from_intermediate_responses() -> None:
    client = FakeResponsesClient()
    client.queued.append(
        _response(
            "resp-mixed",
            [
                {
                    "id": "msg-source",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "source",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.test/intermediate",
                                    "title": "Intermediate source",
                                    "start_index": 0,
                                    "end_index": 6,
                                },
                                {"type": "file_path", "file_id": "file-intermediate", "index": 6},
                            ],
                            "logprobs": [],
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "id": "fc-call-1",
                    "call_id": "call-1",
                    "name": "execute_python",
                    "arguments": "{}",
                    "status": "completed",
                },
            ],
        )
    )
    client.queue_text('{"answer":"42"}')

    response = await OpenAIProvider(client, "gpt-5.6-sol").complete(
        model_request(), RecordingExecutor({"stdout": "42"})
    )

    assert [citation.url for citation in response.citations if citation.kind == "url"] == [
        "https://example.test/intermediate"
    ]
    assert response.remote_file_ids == ("file-intermediate",)


def test_model_request_rejects_non_strict_output_format() -> None:
    with pytest.raises(ValidationError, match="strict object JSON Schema"):
        model_request(output_schema={"type": "json_object"})


def test_provider_request_is_frozen() -> None:
    request = model_request()

    with pytest.raises(ValidationError, match="frozen"):
        request.reasoning_effort = "high"
