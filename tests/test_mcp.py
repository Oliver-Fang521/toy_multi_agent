import asyncio
import json
import os
import socket
import threading
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp import Client
from openai import APITimeoutError
from openpyxl import Workbook
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

from gaia_agent import mcp_server
from gaia_agent.config import RunConfig
from gaia_agent.mcp_client import AttachmentPolicyError, MCPToolClient, ToolExecutionError
from gaia_agent.mcp_server import (
    AttachmentReader,
    HostedOpenAIBackend,
    HTTPWebReader,
    MCPServerSettings,
    WebpageError,
    build_server,
)
from gaia_agent.models import Usage
from gaia_agent.observability import Observability, ObservabilityConfig
from gaia_agent.reliability import RetryPolicy
from gaia_agent.search import (
    OpenAISearchProvider,
    SearchProviderChain,
    SearchUnavailableError,
)


@pytest.fixture
def run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        openai_api_key="openai-test-secret",
        hf_token="hf-test-secret",
        sample_size=3,
        timeout_seconds=0.2,
        output_dir=tmp_path / "results",
    )


def test_mcp_server_env_propagates_openai_base_url(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    client = MCPToolClient(task_root=tmp_path, config=run_config)

    assert client._minimal_server_env()["OPENAI_BASE_URL"] == "https://api.deepseek.com"


def test_mcp_server_env_propagates_specialist_search_budget(tmp_path: Path) -> None:
    config = RunConfig(
        openai_api_key="openai-test-secret",
        hf_token="hf-test-secret",
        sample_size=3,
        timeout_seconds=100,
        specialist_timeout_seconds=70,
        output_dir=tmp_path / "results",
    )

    env = MCPToolClient(task_root=tmp_path, config=config)._minimal_server_env()

    assert env["GAIA_SPECIALIST_TIMEOUT_SECONDS"] == "70.0"


@pytest.mark.asyncio
async def test_settings_chain_classifies_transient_openai_failure_and_uses_serper_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingOpenAIBackend:
        provider = "openai"

        async def search(self, query: str) -> dict[str, Any]:
            raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.test"))

    class SerperFallback:
        provider = "serper"

        async def search(self, query: str) -> dict[str, Any]:
            return {"answer": "fallback", "sources": []}

    monkeypatch.setenv("GAIA_TASK_ROOT", str(tmp_path))
    monkeypatch.setenv("GAIA_ATTACHMENTS", "[]")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.delenv("GAIA_MCP_FAKE_BACKEND", raising=False)
    monkeypatch.setattr(mcp_server, "AsyncOpenAI", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(mcp_server, "HostedOpenAIBackend", lambda **kwargs: FailingOpenAIBackend())
    monkeypatch.setattr(
        mcp_server, "SerperSearchProvider", lambda *args, **kwargs: SerperFallback()
    )

    settings = mcp_server._settings_from_env()
    try:
        assert settings.search_provider is not None
        assert await settings.search_provider.search("question") == {
            "answer": "fallback",
            "sources": [],
        }
    finally:
        await settings.aclose()


@pytest.mark.asyncio
async def test_settings_chain_propagates_untyped_openai_implementation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadBackend:
        async def search(self, query: str) -> dict[str, Any]:
            raise RuntimeError("implementation defect")

    class HealthySerper:
        provider = "serper"

        def __init__(self) -> None:
            self.calls = 0

        async def search(self, query: str) -> dict[str, Any]:
            self.calls += 1
            return {"answer": "must not run", "sources": []}

    serper = HealthySerper()
    monkeypatch.setenv("GAIA_TASK_ROOT", str(tmp_path))
    monkeypatch.setenv("GAIA_ATTACHMENTS", "[]")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.delenv("GAIA_MCP_FAKE_BACKEND", raising=False)
    monkeypatch.setattr(mcp_server, "AsyncOpenAI", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(mcp_server, "HostedOpenAIBackend", lambda **kwargs: BadBackend())
    monkeypatch.setattr(mcp_server, "SerperSearchProvider", lambda *args, **kwargs: serper)

    settings = mcp_server._settings_from_env()
    try:
        assert settings.search_provider is not None
        with pytest.raises(RuntimeError, match="implementation defect"):
            await settings.search_provider.search("question")
        assert serper.calls == 0
    finally:
        await settings.aclose()


@pytest.mark.asyncio
async def test_settings_close_owned_search_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_TASK_ROOT", str(tmp_path))
    monkeypatch.setenv("GAIA_ATTACHMENTS", "[]")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.delenv("GAIA_MCP_FAKE_BACKEND", raising=False)
    monkeypatch.setattr(mcp_server, "AsyncOpenAI", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(mcp_server, "HostedOpenAIBackend", lambda **kwargs: SimpleNamespace())

    settings = mcp_server._settings_from_env()
    assert settings.search_client is not None

    await settings.aclose()

    assert settings.search_client.is_closed is True


def test_mcp_server_env_propagates_only_optional_search_provider_keys(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "serper-test-secret")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-test-secret")
    client = MCPToolClient(task_root=tmp_path, config=run_config)

    env = client._minimal_server_env()

    assert env["SERPER_API_KEY"] == "serper-test-secret"
    assert env["SERPAPI_API_KEY"] == "serpapi-test-secret"
    assert "PATH" not in env


@pytest.mark.asyncio
async def test_mcp_client_categorizes_webpage_http_errors_without_retaining_raw_message(
    tmp_path: Path, run_config: RunConfig
) -> None:
    class FailingWebClient:
        async def call_tool(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                is_error=True,
                content=[
                    SimpleNamespace(
                        type="text",
                        text=(
                            "Error executing tool read_webpage: HTTP 503 for "
                            "https://private.example?api_key=secret with body=classified"
                        ),
                    )
                ],
            )

    client = MCPToolClient(task_root=tmp_path, config=run_config, allowed={"read_webpage"})
    client._client = FailingWebClient()

    with pytest.raises(ToolExecutionError) as caught:
        await client.call("read_webpage", {"url": "https://private.example"})

    assert caught.value.category == "http_status"
    assert caught.value.retryable is True
    assert caught.value.http_status == 503
    assert "private.example" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert "classified" not in str(caught.value)


@pytest.mark.asyncio
async def test_read_webpage_retries_only_transient_failures_before_returning_content(
    tmp_path: Path,
) -> None:
    class FlakyReader:
        def __init__(self) -> None:
            self.calls = 0

        async def read(self, url: str) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                error = ToolExecutionError("timeout with https://private.example and body=secret")
                error.category = "timeout"
                error.retryable = True
                raise error
            return {
                "url": url,
                "status_code": 200,
                "content_type": "text/plain",
                "text": "recovered page",
                "truncated": False,
            }

    reader = FlakyReader()
    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=reader,
        backend=SimpleNamespace(),
        api_key="",
        hf_token="",
    )
    async with Client(build_server(settings)) as client:
        result = await client.call_tool("read_webpage", {"url": "https://example.test/article"})

    assert result.is_error is False
    assert result.structured_content == {
        "url": "https://example.test/article",
        "status_code": 200,
        "content_type": "text/plain",
        "text": "recovered page",
        "truncated": False,
    }
    assert reader.calls == 2


@pytest.mark.asyncio
async def test_read_webpage_returns_bounded_search_snippets_after_transient_failure(
    tmp_path: Path,
) -> None:
    class UnavailableReader:
        async def read(self, url: str) -> dict[str, Any]:
            error = ToolExecutionError("timeout with key=secret and body=private")
            error.category = "timeout"
            error.retryable = True
            raise error

    class SearchFallback:
        provider = "serper"

        async def search(self, query: str) -> dict[str, Any]:
            assert query == "https://example.test/article"
            return {
                "answer": "A public search summary.",
                "sources": [{"url": "https://source.test", "title": "Source"}],
            }

    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=UnavailableReader(),
        backend=SimpleNamespace(),
        api_key="secret",
        hf_token="",
        search_provider=SearchFallback(),
    )
    async with Client(build_server(settings)) as client:
        result = await client.call_tool("read_webpage", {"url": "https://example.test/article"})

    assert result.is_error is False
    assert result.structured_content == {
        "url": "https://example.test/article",
        "answer": "A public search summary.",
        "sources": [{"url": "https://source.test", "title": "Source"}],
        "fallback": "search",
    }
    assert result.meta["gaia.trace"] == {
        "provider": "serper",
        "attempt_count": 1,
        "fallback_used": True,
    }


@pytest.mark.asyncio
async def test_read_webpage_bounds_and_redacts_oversized_search_fallback(
    tmp_path: Path,
) -> None:
    secret = "search-provider-secret"

    class UnavailableReader:
        async def read(self, url: str) -> dict[str, Any]:
            error = ToolExecutionError("timeout")
            error.category = "timeout"
            error.retryable = True
            raise error

    class SearchFallback:
        provider = "serper"

        async def search(self, query: str) -> dict[str, Any]:
            return {
                "answer": secret + "漢字🚀" * 4_000,
                "sources": [
                    {
                        "url": f"https://source-{index}.test/article",
                        "title": secret + "title" * 5_000,
                    }
                    for index in range(20)
                ],
            }

    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=UnavailableReader(),
        backend=SimpleNamespace(),
        api_key=secret,
        hf_token="",
        search_provider=SearchFallback(),
    )
    long_url = "https://example.test/" + "path" * 8_000
    async with Client(build_server(settings)) as client:
        tool_result = await client.call_tool("read_webpage", {"url": long_url})

    assert tool_result.is_error is False
    assert isinstance(tool_result.structured_content, dict)
    result = dict(tool_result.structured_content)
    assert set(result) == {"url", "answer", "sources", "fallback"}
    assert result["fallback"] == "search"
    assert "answer" in result
    assert len(result["sources"]) <= mcp_server.MAX_WEB_SEARCH_SOURCES
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= (
        mcp_server.MAX_WEB_SEARCH_OUTPUT_BYTES
    )
    assert secret not in repr(result)


@pytest.mark.asyncio
async def test_read_webpage_reports_parser_value_errors_as_parse_failure(
    tmp_path: Path,
) -> None:
    class ParserFailureReader:
        async def read(self, url: str) -> dict[str, Any]:
            raise ValueError("HTML parser rejected malformed content")

    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=ParserFailureReader(),
        backend=SimpleNamespace(),
        api_key="",
        hf_token="",
    )
    async with Client(build_server(settings)) as client:
        result = await client.call_tool("read_webpage", {"url": "https://example.test/article"})

    assert result.is_error is True
    assert '"category":"parse_failure"' in result.content[0].text


@pytest.mark.asyncio
async def test_http_reader_classifies_content_decode_errors_as_parse_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with httpx.AsyncClient() as client:
        reader = HTTPWebReader(client=client)

        async def fail_to_decode(url: str) -> dict[str, Any]:
            raise httpx.DecodingError("raw body must not escape")

        monkeypatch.setattr(reader, "_read_once", fail_to_decode)
        with pytest.raises(WebpageError) as caught:
            await reader.read("https://example.test/article")

    assert caught.value.category == "parse_failure"
    assert caught.value.retryable is False
    assert "raw body" not in str(caught.value)


@pytest.mark.asyncio
async def test_read_webpage_reports_empty_content_as_safe_nonretryable_category(
    tmp_path: Path,
) -> None:
    class EmptyReader:
        async def read(self, url: str) -> dict[str, Any]:
            return {
                "url": url,
                "status_code": 200,
                "content_type": "text/plain",
                "text": "",
                "truncated": False,
            }

    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=EmptyReader(),
        backend=SimpleNamespace(),
        api_key="",
        hf_token="",
    )
    async with Client(build_server(settings)) as client:
        result = await client.call_tool("read_webpage", {"url": "https://example.test/article"})

    assert result.is_error is True
    assert '"category":"empty_content"' in result.content[0].text


@pytest.mark.asyncio
async def test_read_webpage_reports_policy_rejection_without_retry_or_search_fallback(
    tmp_path: Path,
) -> None:
    class PolicyReader:
        def __init__(self) -> None:
            self.calls = 0

        async def read(self, url: str) -> dict[str, Any]:
            self.calls += 1
            raise ValueError("URL must refer to a public host")

    class ForbiddenFallback:
        provider = "serper"

        async def search(self, query: str) -> dict[str, Any]:
            raise AssertionError("policy errors must not invoke search fallback")

    reader = PolicyReader()
    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=reader,
        backend=SimpleNamespace(),
        api_key="",
        hf_token="",
        search_provider=ForbiddenFallback(),
    )
    async with Client(build_server(settings)) as client:
        result = await client.call_tool("read_webpage", {"url": "https://private.test"})

    assert result.is_error is True
    assert '"category":"policy_rejection"' in result.content[0].text
    assert reader.calls == 1


@pytest.fixture
def fake_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_MCP_FAKE_BACKEND", "1")


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve_public(host: str, port: int, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.72.14", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve_public)


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


@pytest.mark.asyncio
async def test_stdio_client_discovers_and_filters_tools(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"execute_python"}
    ) as client:
        assert [tool.name for tool in client.tool_specs] == ["execute_python"]
        assert client.tool_specs[0].parameters["additionalProperties"] is False
        assert set(client.tool_specs[0].parameters["required"]) == {"instruction", "paths"}
        assert client.is_connected

    assert not client.is_connected


@pytest.mark.asyncio
async def test_stdio_startup_does_not_import_from_untrusted_task_root(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    marker = tmp_path / "untrusted-code-ran"
    (tmp_path / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    forged_package = tmp_path / "gaia_agent"
    forged_package.mkdir()
    (forged_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('forged')\n",
        encoding="utf-8",
    )

    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"submit_answer"}
    ) as client:
        assert [tool.name for tool in client.tool_specs] == ["submit_answer"]

    assert not marker.exists()


@pytest.mark.asyncio
async def test_stdio_client_calls_real_server_and_rejects_disallowed_tool(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"submit_answer"}
    ) as client:
        result = await client.call(
            "submit_answer", {"answer": "  forty-two  ", "evidence_ids": ["source-1"]}
        )
        assert result == {"answer": "forty-two", "evidence_ids": ["source-1"]}
        with pytest.raises(ToolExecutionError, match="not allowed"):
            await client.call("web_search", {"query": "ignored"})


@pytest.mark.asyncio
async def test_mcp_client_records_tool_shape_without_arguments_or_results(
    tmp_path: Path,
    fake_backend_env: None,
    run_config: RunConfig,
) -> None:
    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "task-1",
        level=1,
        model="test-model",
        reasoning_effort="low",
        run_index=0,
        evaluation_mode="test",
    )
    async with MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        allowed={"submit_answer"},
        observability=observability,
    ) as client:
        result = await client.call(
            "submit_answer", {"answer": "secret answer", "evidence_ids": ["secret-id"]}
        )
    trace.close()

    assert result["answer"] == "secret answer"
    tool_span = trace_client.observations[-1]
    assert tool_span.name == "mcp-tool"
    assert tool_span.kwargs["metadata"] == {"tool": "submit_answer", "call_number": 1}
    assert tool_span.kwargs["input"] == {
        "argument_count": 2,
        "argument_keys": ["answer", "evidence_ids"],
    }
    assert tool_span.updates[-1]["metadata"] == {
        "result_key_count": 2,
        "status": "ok",
    }
    assert tool_span.updates[-1]["output"] == {
        "result_key_count": 2,
        "result_keys": ["answer", "evidence_ids"],
    }
    assert "secret answer" not in repr(tool_span.kwargs)
    assert "secret-id" not in repr(tool_span.kwargs)

    with pytest.raises(ToolExecutionError, match="not connected"):
        await client.call("submit_answer", {"answer": "42", "evidence_ids": []})


@pytest.mark.asyncio
async def test_mcp_span_omits_argument_and_result_keys_outside_the_tool_contract(
    tmp_path: Path,
    run_config: RunConfig,
) -> None:
    class UnexpectedShapeClient:
        async def call_tool(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                is_error=False,
                content=[],
                structured_content={
                    "url": "https://private.test/article",
                    "text": "private body",
                    "prompt": "private prompt",
                    "raw_body": "private raw body",
                },
            )

    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "task-1",
        level=1,
        model="test-model",
        reasoning_effort="max",
        run_index=0,
        evaluation_mode="test",
    )
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        allowed={"read_webpage"},
        observability=observability,
    )
    client._client = UnexpectedShapeClient()

    await client.call(
        "read_webpage",
        {
            "url": "https://private.test/article",
            "prompt": "private prompt",
            "raw_body": "private raw body",
        },
    )
    trace.close()

    span = trace_client.observations[-1]
    assert span.kwargs["input"] == {
        "argument_count": 1,
        "argument_keys": ["url"],
    }
    assert span.updates[-1]["metadata"]["result_key_count"] == 2
    assert span.updates[-1]["output"] == {
        "result_key_count": 2,
        "result_keys": ["text", "url"],
    }
    assert "prompt" not in repr(span.kwargs)
    assert "raw_body" not in repr(span.updates)


@pytest.mark.asyncio
async def test_mcp_failure_span_records_only_bounded_structured_failure_metadata(
    tmp_path: Path, run_config: RunConfig
) -> None:
    secret = "sk-private-trace-secret"

    class FailingClient:
        async def call_tool(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                is_error=True,
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "tool_failed",
                                "category": "http_status",
                                "retryable": True,
                                "provider": "serper",
                                "attempt_count": 3,
                                "http_status": 503,
                                "circuit_open_count": 2,
                                "fallback_attempted": True,
                                "raw": f"{secret} https://private.test body=classified",
                            }
                        ),
                    )
                ],
            )

    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "task-1",
        level=1,
        model="test-model",
        reasoning_effort="max",
        run_index=0,
        evaluation_mode="test",
    )
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        allowed={"web_search"},
        observability=observability,
    )
    client._client = FailingClient()

    with pytest.raises(ToolExecutionError):
        await client.call("web_search", {"query": f"secret query {secret}"})
    trace.close()

    span = trace_client.observations[-1]
    assert span.updates[-1]["metadata"] == {
        "error_type": "ToolExecutionError",
        "provider": "serper",
        "category": "http_status",
        "retryable": True,
        "attempt_count": 3,
        "http_status": 503,
        "circuit_open_count": 2,
        "fallback_attempted": True,
        "status": "error",
    }
    assert span.updates[-1]["output"] == {"result_key_count": 0, "result_keys": []}
    assert secret not in repr(span.kwargs)
    assert secret not in repr(span.updates)
    assert "private.test" not in repr(span.updates)
    assert "classified" not in repr(span.updates)


@pytest.mark.asyncio
async def test_mcp_success_span_records_fallback_decision_without_result_content(
    tmp_path: Path, run_config: RunConfig
) -> None:
    secret = "private fallback answer"

    class FallbackClient:
        async def call_tool(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                is_error=False,
                content=[],
                structured_content={
                    "url": "https://private.test/article",
                    "answer": secret,
                    "sources": [{"url": "https://private.test/source", "title": secret}],
                    "fallback": "search",
                },
            )

    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "task-1",
        level=1,
        model="test-model",
        reasoning_effort="max",
        run_index=0,
        evaluation_mode="test",
    )
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        allowed={"read_webpage"},
        observability=observability,
    )
    client._client = FallbackClient()

    result = await client.call("read_webpage", {"url": "https://private.test/article"})
    trace.close()

    assert result["answer"] == secret
    span = trace_client.observations[-1]
    assert span.updates[-1]["metadata"] == {
        "result_key_count": 4,
        "fallback_used": True,
        "status": "ok",
    }
    assert span.updates[-1]["output"] == {
        "result_key_count": 4,
        "result_keys": ["answer", "fallback", "sources", "url"],
    }
    assert secret not in repr(span.kwargs)
    assert secret not in repr(span.updates)
    assert "private.test" not in repr(span.updates)


@pytest.mark.asyncio
async def test_mcp_success_span_records_private_search_routing_metadata(
    tmp_path: Path, run_config: RunConfig
) -> None:
    class SearchClient:
        async def call_tool(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                is_error=False,
                content=[],
                meta={
                    "gaia.trace": {
                        "provider": "serper",
                        "attempt_count": 3,
                        "circuit_open_count": 1,
                        "fallback_used": True,
                    }
                },
                structured_content={"answer": "private answer", "sources": []},
            )

    trace_client = _TraceClient()
    observability = Observability(
        ObservabilityConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=trace_client,
    )
    trace = observability.start_task(
        "task-1", level=1, model="test-model", reasoning_effort="max", run_index=0,
        evaluation_mode="test",
    )
    client = MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"web_search"}, observability=observability
    )
    client._client = SearchClient()

    result = await client.call("web_search", {"query": "private question"})
    trace.close()

    assert result == {"answer": "private answer", "sources": []}
    assert trace_client.observations[-1].updates[-1]["metadata"] == {
        "result_key_count": 2,
        "provider": "serper",
        "attempt_count": 3,
        "circuit_open_count": 1,
        "fallback_used": True,
        "status": "ok",
    }
    assert "private answer" not in repr(trace_client.observations[-1].updates)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_provider", ["https://private.test/provider", ["serper"]]
)
async def test_mcp_malformed_failure_metadata_fails_closed_without_raw_text(
    tmp_path: Path, run_config: RunConfig, invalid_provider: Any
) -> None:
    class MalformedFailureClient:
        async def call_tool(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                is_error=True,
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "tool_failed",
                                "category": "http_status",
                                "retryable": True,
                                "provider": invalid_provider,
                                "attempt_count": 1,
                                "http_status": 503,
                                "raw": "classified body",
                            }
                        ),
                    )
                ],
            )

    client = MCPToolClient(task_root=tmp_path, config=run_config, allowed={"web_search"})
    client._client = MalformedFailureClient()

    with pytest.raises(ToolExecutionError) as caught:
        await client.call("web_search", {"query": "private query"})

    assert str(caught.value) == "tool failed: web_search"
    assert "private.test" not in str(caught.value)
    assert "classified" not in str(caught.value)


@pytest.mark.asyncio
async def test_execute_python_accepts_instruction_without_attachments(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"execute_python"}
    ) as client:
        result = await client.call("execute_python", {"instruction": "calculate 6 * 7"})

    assert result == {
        "output_text": "deterministic code result for calculate 6 * 7",
        "outputs": [],
    }


@pytest.mark.asyncio
async def test_stdio_transcribes_only_registered_audio(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"synthetic audio")
    text = tmp_path / "notes.txt"
    text.write_text("not audio", encoding="utf-8")

    async with MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        allowed={"transcribe_audio"},
        attachments={audio, text},
    ) as client:
        transcript = await client.call("transcribe_audio", {"path": "sample.mp3"})
        with pytest.raises(ToolExecutionError, match="registered"):
            await client.call("transcribe_audio", {"path": "missing.mp3"})
        with pytest.raises(ToolExecutionError, match="audio"):
            await client.call("transcribe_audio", {"path": "notes.txt"})

    assert transcript == {"text": "deterministic audio transcript"}


@pytest.mark.asyncio
async def test_stdio_shutdown_preserves_tool_error_type(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    with pytest.raises(ToolExecutionError, match="not allowed"):
        async with MCPToolClient(
            task_root=tmp_path, config=run_config, allowed={"submit_answer"}
        ) as client:
            await client.call("web_search", {"query": "ignored"})


@pytest.mark.asyncio
async def test_read_attachment_rejects_path_escape(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"read_attachment"}
    ) as client:
        with pytest.raises(ToolExecutionError, match="outside task root"):
            await client.call("read_attachment", {"path": "../secret.txt"})


@pytest.mark.asyncio
async def test_read_attachment_requires_registration(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    attachment = tmp_path / "notes.txt"
    attachment.write_text("registered content", encoding="utf-8")

    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"read_attachment"}
    ) as client:
        with pytest.raises(ToolExecutionError, match="not registered"):
            await client.call("read_attachment", {"path": "notes.txt"})


@pytest.mark.asyncio
async def test_read_attachment_extracts_registered_text(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    attachment = tmp_path / "notes.txt"
    attachment.write_text("registered content", encoding="utf-8")

    async with MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        allowed={"read_attachment"},
        attachments={attachment},
    ) as client:
        result = await client.call("read_attachment", {"path": "notes.txt"})

    assert result["kind"] == "text"
    assert result["text"] == "registered content"
    assert result["path"] == "notes.txt"


@pytest.mark.asyncio
async def test_stdio_backend_timeout_is_bounded(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"web_search"}
    ) as client:
        with pytest.raises(ToolExecutionError, match="timed out"):
            await client.call("web_search", {"query": "__GAIA_TIMEOUT__"})


@pytest.mark.asyncio
async def test_stdio_tool_output_is_truncated_to_64_kib(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"execute_python"}
    ) as client:
        result = await client.call(
            "execute_python", {"instruction": "__GAIA_LARGE_OUTPUT__", "paths": []}
        )

    assert result["truncated"] is True
    assert len(json.dumps(result, separators=(",", ":")).encode()) <= 64 * 1024


@pytest.mark.asyncio
async def test_stdio_tool_output_limit_accounts_for_json_escaping(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"execute_python"}
    ) as client:
        result = await client.call(
            "execute_python", {"instruction": "__GAIA_ESCAPED_OUTPUT__", "paths": []}
        )

    assert result["truncated"] is True
    assert len(json.dumps(result, separators=(",", ":")).encode()) <= 64 * 1024


@pytest.mark.asyncio
async def test_stdio_errors_redact_both_configured_secrets(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"web_search"}
    ) as client:
        with pytest.raises(ToolExecutionError) as caught:
            await client.call("web_search", {"query": "__GAIA_SECRET_ERROR__"})

    message = str(caught.value)
    assert "openai-test-secret" not in message
    assert "hf-test-secret" not in message
    assert message.count("[REDACTED]") >= 2


@pytest.mark.asyncio
async def test_stdio_results_are_recursively_redacted(
    tmp_path: Path, fake_backend_env: None, run_config: RunConfig
) -> None:
    async with MCPToolClient(
        task_root=tmp_path, config=run_config, allowed={"web_search"}
    ) as client:
        result = await client.call("web_search", {"query": "__GAIA_NESTED_SECRET__"})

    assert result == {
        "answer": "safe",
        "sources": [],
        "debug": {
            "api_key": "[REDACTED]",
            "message": "[REDACTED] and [REDACTED]",
        },
    }


@pytest.mark.asyncio
async def test_safe_tool_call_preserves_trusted_usage_outside_payload_redaction(
    tmp_path: Path,
) -> None:
    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=SimpleNamespace(),
        api_key="openai-test-secret",
        hf_token="hf-test-secret",
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=13,
            output_tokens=5,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
    )

    result = await mcp_server._safe_tool_call(
        settings,
        lambda: mcp_server._with_response_usage(
            {
                "answer": "safe",
                "debug": {
                    "access_token": "openai-test-secret",
                    "secret_note": "hf-test-secret",
                },
            },
            response,
        ),
    )

    assert result == {
        "answer": "safe",
        "debug": {
            "access_token": "[REDACTED]",
            "secret_note": "[REDACTED]",
        },
        "__gaia_usage__": {
            "input_tokens": 13,
            "output_tokens": 5,
            "reasoning_tokens": 2,
            "total_tokens": 18,
        },
    }


@pytest.mark.asyncio
async def test_safe_tool_call_serializes_only_safe_search_decisions(tmp_path: Path) -> None:
    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=SimpleNamespace(),
        api_key="openai-test-secret",
        hf_token="hf-test-secret",
    )
    failure = SearchUnavailableError(
        3,
        circuit_open_count=1,
        fallback_attempted=True,
    )
    failure.raw_response = "openai-test-secret https://private.test classified body"

    async def fail() -> dict[str, Any]:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        await mcp_server._safe_tool_call(settings, fail())

    payload = json.loads(str(caught.value))
    assert payload == {
        "error": "tool_failed",
        "provider": None,
        "category": "search_unavailable",
        "retryable": True,
        "attempt_count": 3,
    }
    assert "openai-test-secret" not in str(caught.value)
    assert "private.test" not in str(caught.value)
    assert "classified" not in str(caught.value)


@pytest.mark.asyncio
async def test_safe_tool_call_bounds_final_json_while_preserving_trusted_usage(
    tmp_path: Path,
) -> None:
    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=SimpleNamespace(),
        api_key="openai-test-secret",
        hf_token="hf-test-secret",
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=13,
            output_tokens=5,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
    )

    result = await mcp_server._safe_tool_call(
        settings,
        lambda: mcp_server._with_response_usage(
            {"answer": "x" * mcp_server.MAX_TOOL_OUTPUT_BYTES},
            response,
        ),
    )

    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= mcp_server.MAX_TOOL_OUTPUT_BYTES
    assert result["truncated"] is True
    assert result["__gaia_usage__"] == {
        "input_tokens": 13,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "total_tokens": 18,
    }


@pytest.mark.asyncio
async def test_safe_tool_call_discards_untrusted_reserved_usage(
    tmp_path: Path,
) -> None:
    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=SimpleNamespace(),
        api_key="openai-test-secret",
        hf_token="hf-test-secret",
    )

    result = await mcp_server._safe_tool_call(
        settings,
        lambda: {
            "answer": "safe",
            "api_token": "openai-test-secret",
            "__gaia_usage__": {
                "input_tokens": 3,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "total_tokens": 8,
            },
        },
    )

    assert result == {"answer": "safe", "api_token": "[REDACTED]"}


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_value", [-1, True, "3", "openai-test-secret"])
async def test_safe_tool_call_rejects_malformed_trusted_usage_without_leaking(
    tmp_path: Path,
    invalid_value: Any,
) -> None:
    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=SimpleNamespace(),
        api_key="openai-test-secret",
        hf_token="hf-test-secret",
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=invalid_value,
            output_tokens=5,
            total_tokens=8,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
    )

    with pytest.raises(RuntimeError) as caught:
        await mcp_server._safe_tool_call(
            settings,
            lambda: mcp_server._with_response_usage(
                {"answer": "safe"},
                response,
            ),
        )

    assert "openai-test-secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_mcp_client_strips_and_accumulates_reserved_hosted_usage(
    tmp_path: Path, run_config: RunConfig
) -> None:
    responses = {
        "web_search": {
            "answer": "result",
            "sources": [],
            "__gaia_usage__": {
                "input_tokens": 3,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "total_tokens": 8,
            },
        },
        "execute_python": {
            "output_text": "42",
            "outputs": [],
            "__gaia_usage__": {
                "input_tokens": 7,
                "output_tokens": 11,
                "reasoning_tokens": 4,
                "total_tokens": 18,
            },
        },
    }

    class InterleavingClient:
        async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return SimpleNamespace(
                is_error=False,
                structured_content=dict(responses[name]),
                content=[],
            )

    client = MCPToolClient(task_root=tmp_path, config=run_config)
    client._client = InterleavingClient()

    search, code = await asyncio.gather(
        client.call("web_search", {"query": "question"}),
        client.call("execute_python", {"instruction": "6 * 7"}),
    )

    assert search == {"answer": "result", "sources": []}
    assert code == {"output_text": "42", "outputs": []}
    assert "__gaia_usage__" not in repr((search, code))
    assert client.usage == Usage(
        input_tokens=10,
        output_tokens=16,
        reasoning_tokens=6,
        total_tokens=26,
    )


@pytest.mark.asyncio
async def test_mcp_client_rejects_malformed_reserved_hosted_usage(
    tmp_path: Path, run_config: RunConfig
) -> None:
    class MalformedUsageClient:
        async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            return SimpleNamespace(
                is_error=False,
                structured_content={
                    "answer": "result",
                    "sources": [],
                    "__gaia_usage__": {
                        "input_tokens": 3,
                        "output_tokens": 5,
                        "reasoning_tokens": 2,
                        "total_tokens": 8,
                        "unexpected": 1,
                    },
                },
                content=[],
            )

    client = MCPToolClient(task_root=tmp_path, config=run_config)
    client._client = MalformedUsageClient()

    with pytest.raises(ToolExecutionError, match="invalid hosted usage metadata"):
        await client.call("web_search", {"query": "question"})

    assert client.usage == Usage()


@pytest.mark.asyncio
async def test_attachment_bytes_require_registration_and_reject_symlink_swap(
    tmp_path: Path, run_config: RunConfig
) -> None:
    attachment = tmp_path / "registered.png"
    attachment.write_bytes(b"registered")
    unregistered = tmp_path / "unregistered.png"
    unregistered.write_bytes(b"private")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    outside.write_bytes(b"outside")
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        attachments=(attachment,),
    )

    with pytest.raises(AttachmentPolicyError, match="not registered"):
        await client.read_attachment_bytes(unregistered)

    attachment.unlink()
    attachment.symlink_to(outside)
    with pytest.raises(AttachmentPolicyError, match="outside task root|not registered|changed"):
        await client.read_attachment_bytes(attachment)


@pytest.mark.asyncio
async def test_attachment_bytes_reject_sparse_file_over_32_mib_without_reading(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = tmp_path / "large.png"
    with attachment.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        attachments=(attachment,),
    )
    reads: list[int] = []
    original_read = os.read

    def tracked_read(fd: int, size: int) -> bytes:
        reads.append(size)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", tracked_read)

    with pytest.raises(AttachmentPolicyError, match="exceeds size limit"):
        await client.read_attachment_bytes(attachment)

    assert reads == []


@pytest.mark.asyncio
async def test_attachment_bytes_are_read_off_the_event_loop(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = tmp_path / "image.png"
    attachment.write_bytes(b"image bytes")
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        attachments=(attachment,),
    )
    original_read = os.read
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_read(fd: int, size: int) -> bytes:
        loop.call_soon_threadsafe(started.set)
        time.sleep(0.05)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", slow_read)
    read_task = asyncio.create_task(client.read_attachment_bytes(attachment))
    await started.wait()
    await asyncio.sleep(0)

    assert not read_task.done()
    assert await read_task == b"image bytes"


@pytest.mark.asyncio
async def test_attachment_read_oserror_is_normalized_without_raw_details(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = tmp_path / "image.png"
    attachment.write_bytes(b"image bytes")
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        attachments=(attachment,),
    )

    def fail_read(fd: int, size: int) -> bytes:
        raise OSError(5, "raw device failure /private/sensitive-image.png")

    monkeypatch.setattr(os, "read", fail_read)

    with pytest.raises(AttachmentPolicyError) as caught:
        await client.read_attachment_bytes(attachment)

    assert str(caught.value) == "attachment I/O failed"
    assert "raw device" not in str(caught.value)
    assert "sensitive-image" not in str(caught.value)


@pytest.mark.asyncio
async def test_attachment_close_oserror_is_normalized_without_raw_details(
    tmp_path: Path, run_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = tmp_path / "image.png"
    attachment.write_bytes(b"image bytes")
    client = MCPToolClient(
        task_root=tmp_path,
        config=run_config,
        attachments=(attachment,),
    )
    original_close = os.close

    def fail_close(fd: int) -> None:
        original_close(fd)
        raise OSError(5, "raw close failure /private/sensitive-image.png")

    monkeypatch.setattr(os, "close", fail_close)

    with pytest.raises(AttachmentPolicyError) as caught:
        await client.read_attachment_bytes(attachment)

    assert str(caught.value) == "attachment I/O failed"
    assert "raw close" not in str(caught.value)


@pytest.mark.asyncio
async def test_http_reader_extracts_html_text_and_final_url(public_dns: None) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://93.184.216.34/article")
        assert request.headers["host"] == "example.test"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><head><title>T</title><script>nope()</script></head>"
            "<body><h1>Hello</h1><p>world</p></body></html>",
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await HTTPWebReader(client=client, timeout_seconds=1).read(
            "https://example.test/article"
        )

    assert result == {
        "url": "https://example.test/article",
        "status_code": 200,
        "content_type": "text/html; charset=utf-8",
        "text": "T Hello world",
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_http_reader_raises_for_http_status(public_dns: None) -> None:
    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        return httpx.Response(503, text="backend unavailable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = HTTPWebReader(client=client, timeout_seconds=1)
        with pytest.raises(WebpageError) as caught:
            await reader.read("https://example.test/down")

    assert attempts == ["93.184.216.34"]
    assert caught.value.category == "http_status"
    assert caught.value.retryable is True
    assert caught.value.http_status == 503


@pytest.mark.asyncio
async def test_http_reader_rejects_private_dns_before_emitting_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[httpx.Request] = []

    def resolve_private(*args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]

    async def handler(request: httpx.Request) -> httpx.Response:
        emitted.append(request)
        return httpx.Response(200, text="unsafe", request=request)

    monkeypatch.setattr(socket, "getaddrinfo", resolve_private)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="public"):
            await HTTPWebReader(client=client, timeout_seconds=1).read(
                "https://private.test/secret"
            )

    assert emitted == []


@pytest.mark.asyncio
async def test_http_reader_validates_redirect_dns_before_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted_hosts: list[str] = []

    def resolve_by_host(host: str, port: int, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        address = "93.184.216.34" if host == "public.test" else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    async def handler(request: httpx.Request) -> httpx.Response:
        emitted_hosts.append(request.headers["host"])
        if request.headers["host"] == "public.test":
            return httpx.Response(
                302,
                headers={"location": "http://metadata.test/latest"},
                request=request,
            )
        return httpx.Response(200, text="credential", request=request)

    monkeypatch.setattr(socket, "getaddrinfo", resolve_by_host)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="public"):
            await HTTPWebReader(client=client, timeout_seconds=1).read(
                "https://public.test/start"
            )

    assert emitted_hosts == ["public.test"]


@pytest.mark.asyncio
async def test_http_reader_pins_validated_address_against_dns_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions = iter(["93.184.216.34", "127.0.0.1"])
    resolution_count = 0

    def rebinding_dns(host: str, port: int, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        nonlocal resolution_count
        resolution_count += 1
        address = next(resolutions)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "public.test"
        assert request.extensions["sni_hostname"] == "public.test"
        return httpx.Response(200, text="safe", request=request)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_dns)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await HTTPWebReader(client=client, timeout_seconds=1).read(
            "https://public.test/article"
        )

    assert result["url"] == "https://public.test/article"
    assert resolution_count == 1


@pytest.mark.asyncio
async def test_http_reader_falls_back_across_validated_public_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    def dual_stack_dns(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        return [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("2606:4700:4700::1111", port, 0, 0),
            ),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
        ]

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        assert request.headers["host"] == "public.test"
        assert request.extensions["sni_hostname"] == "public.test"
        assert set(request.extensions["timeout"].values()) == {None}
        if request.url.host == "2606:4700:4700::1111":
            raise httpx.ConnectError("IPv6 unavailable", request=request)
        return httpx.Response(200, text="fallback succeeded", request=request)

    monkeypatch.setattr(socket, "getaddrinfo", dual_stack_dns)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await HTTPWebReader(client=client, timeout_seconds=1).read(
            "https://public.test/article"
        )

    assert result["text"] == "fallback succeeded"
    assert result["url"] == "https://public.test/article"
    assert attempts == ["2606:4700:4700::1111", "93.184.216.34"]


class FakeFiles:
    def __init__(self) -> None:
        self.created: list[tuple[str, bytes]] = []

    async def create(self, *, file: Any, purpose: str) -> Any:
        self.created.append((purpose, file.read()))
        return SimpleNamespace(id=f"file-{len(self.created)}")


class FakeResponses:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.response


class FakeTranscriptions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def create(self, *, model: str, file: Any) -> Any:
        self.requests.append(
            {
                "model": model,
                "filename": Path(file.name).name,
                "body": file.read(),
            }
        )
        return self.response


class FakeHostedClient:
    def __init__(self, response: Any) -> None:
        self.responses = FakeResponses(response)
        self.files = FakeFiles()
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions(response))


@pytest.mark.asyncio
async def test_hosted_search_requires_web_tool_and_extracts_sources(tmp_path: Path) -> None:
    response = SimpleNamespace(
        output_text="The answer",
        usage=SimpleNamespace(
            input_tokens=13,
            output_tokens=5,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
        output=[
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(
                    sources=[
                        SimpleNamespace(url="https://one.test", title="One"),
                        {"url": "https://two.test", "title": "Two"},
                    ]
                ),
            )
        ],
    )
    client = FakeHostedClient(response)
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=1,
    )

    result = await backend.web_search("question")

    assert client.responses.requests == [
        {
            "model": "gpt-5.6-sol",
            "input": "question",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
            "reasoning": {"effort": "medium"},
        }
    ]
    assert result == {
        "answer": "The answer",
        "sources": [
            {"url": "https://one.test", "title": "One"},
            {"url": "https://two.test", "title": "Two"},
        ],
        "__gaia_usage__": {
            "input_tokens": 13,
            "output_tokens": 5,
            "reasoning_tokens": 2,
            "total_tokens": 18,
        },
    }


@pytest.mark.asyncio
async def test_hosted_openai_empty_result_falls_through_to_healthy_provider(
    tmp_path: Path,
) -> None:
    response = SimpleNamespace(
        output_text="",
        output=[],
        usage=SimpleNamespace(
            input_tokens=2,
            output_tokens=0,
            total_tokens=2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    backend = HostedOpenAIBackend(
        client=FakeHostedClient(response),
        model="gpt-5.6-luna",
        reasoning_effort="max",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=1,
    )

    class HealthySerper:
        provider = "serper"

        async def search(self, query: str) -> dict[str, Any]:
            return {"answer": "healthy fallback", "sources": []}

    chain = SearchProviderChain(
        [OpenAISearchProvider(backend), HealthySerper()],
        RetryPolicy(max_attempts=2),
        clock=lambda: 0.0,
    )

    result = await chain.search("question")

    assert result == {"answer": "healthy fallback", "sources": []}
    assert result.provider == "serper"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("global_timeout", "expected_web_timeout"),
    [(300.0, 60.0), (12.5, 12.5)],
)
async def test_hosted_web_search_caps_only_its_response_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    global_timeout: float,
    expected_web_timeout: float,
) -> None:
    response = SimpleNamespace(
        output_text="answer",
        output=[],
        usage=SimpleNamespace(
            input_tokens=2,
            output_tokens=1,
            total_tokens=3,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    observed_timeouts: list[float] = []

    async def capturing_wait_for(awaitable: Any, *, timeout: float) -> Any:
        observed_timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(mcp_server.asyncio, "wait_for", capturing_wait_for)
    backend = HostedOpenAIBackend(
        client=FakeHostedClient(response),
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=global_timeout,
    )

    await backend.web_search("question")

    assert observed_timeouts == [expected_web_timeout]
    assert backend._timeout_seconds == global_timeout


@pytest.mark.asyncio
async def test_hosted_web_search_reserves_configured_time_for_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        output_text="answer",
        output=[],
        usage=SimpleNamespace(
            input_tokens=2,
            output_tokens=1,
            total_tokens=3,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    observed_timeouts: list[float] = []

    async def capturing_wait_for(awaitable: Any, *, timeout: float) -> Any:
        observed_timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(mcp_server.asyncio, "wait_for", capturing_wait_for)
    backend = HostedOpenAIBackend(
        client=FakeHostedClient(response),
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=10.0,
        web_search_timeout_seconds=5.0,
    )

    await backend.web_search("question")

    assert observed_timeouts == [5.0]
    assert observed_timeouts[0] < backend._timeout_seconds


@pytest.mark.asyncio
async def test_hosted_search_bounds_result_and_sources_while_preserving_usage(
    tmp_path: Path,
) -> None:
    sources = [
        SimpleNamespace(url=f"https://source-{index}.example/article", title=f"Source {index}")
        for index in range(20)
    ]
    response = SimpleNamespace(
        output_text="answer-" + "x" * (64 * 1024),
        usage=SimpleNamespace(
            input_tokens=101,
            output_tokens=17,
            total_tokens=118,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
        output=[SimpleNamespace(type="web_search_call", action=SimpleNamespace(sources=sources))],
    )
    backend = HostedOpenAIBackend(
        client=FakeHostedClient(response),
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=1,
    )

    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=backend,
        api_key="",
        hf_token="",
    )
    async with Client(build_server(settings)) as client:
        tool_result = await client.call_tool("web_search", {"query": "question"})
    assert isinstance(tool_result.structured_content, dict)
    result = dict(tool_result.structured_content)

    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 16 * 1024
    assert result["answer"].startswith("answer-")
    assert result["sources"][0] == {
        "url": "https://source-0.example/article",
        "title": "Source 0",
    }
    assert len(result["sources"]) <= 8
    assert result["__gaia_usage__"] == {
        "input_tokens": 101,
        "output_tokens": 17,
        "reasoning_tokens": 3,
        "total_tokens": 118,
    }


@pytest.mark.asyncio
async def test_safe_web_bound_is_rechecked_after_redaction_expands_payload(
    tmp_path: Path,
) -> None:
    secret = "key"
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=13,
            output_tokens=5,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
    )

    class ExpandingWebBackend:
        async def web_search(self, query: str) -> dict[str, Any]:
            return mcp_server._with_response_usage(
                {
                    "answer": secret * 4_000,
                    "sources": [
                        {"url": "https://source.example", "title": secret * 100}
                    ],
                },
                response,
            )

    settings = MCPServerSettings(
        task_root=tmp_path,
        attachment_reader=AttachmentReader(tmp_path, set()),
        web_reader=SimpleNamespace(),
        backend=ExpandingWebBackend(),
        api_key=secret,
        hf_token="",
    )

    async with Client(build_server(settings)) as client:
        tool_result = await client.call_tool("web_search", {"query": "question"})

    assert isinstance(tool_result.structured_content, dict)
    result = dict(tool_result.structured_content)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 16 * 1024
    assert set(result) == {"answer", "sources", "__gaia_usage__"}
    assert result["answer"].startswith("[REDACTED]")
    assert result["sources"]
    assert result["__gaia_usage__"]["total_tokens"] == 18
    assert secret not in repr(result)


@pytest.mark.asyncio
async def test_hosted_audio_transcription_uses_registered_mp3_and_maps_usage(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"synthetic mp3")
    response = SimpleNamespace(
        text="spoken answer",
        usage=SimpleNamespace(input_tokens=351, output_tokens=139, total_tokens=490),
    )
    client = FakeHostedClient(response)
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {audio}),
        timeout_seconds=1,
    )

    result = await backend.transcribe_audio("sample.mp3")

    assert client.audio.transcriptions.requests == [
        {
            "model": "gpt-4o-mini-transcribe",
            "filename": "sample.mp3",
            "body": b"synthetic mp3",
        }
    ]
    assert result == {
        "text": "spoken answer",
        "__gaia_usage__": {
            "input_tokens": 351,
            "output_tokens": 139,
            "reasoning_tokens": 0,
            "total_tokens": 490,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".mp4", ".webm", ".MP4"])
async def test_hosted_audio_transcription_accepts_explicit_container_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    audio = tmp_path / f"sample{suffix}"
    audio.write_bytes(b"synthetic media")
    response = SimpleNamespace(text="spoken answer", usage=None)
    client = FakeHostedClient(response)
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {audio}),
        timeout_seconds=1,
    )

    result = await backend.transcribe_audio(audio.name)

    assert result == {"text": "spoken answer"}
    assert client.audio.transcriptions.requests == [
        {
            "model": "gpt-4o-mini-transcribe",
            "filename": audio.name,
            "body": b"synthetic media",
        }
    ]


@pytest.mark.asyncio
async def test_hosted_audio_transcription_rejects_aac(tmp_path: Path) -> None:
    unsupported = tmp_path / "sample.aac"
    unsupported.write_bytes(b"synthetic aac")
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {unsupported}),
        timeout_seconds=1,
    )

    with pytest.raises(ValueError, match="audio format"):
        await backend.transcribe_audio("sample.aac")

    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_transcription_rejects_25_mb_overflow(tmp_path: Path) -> None:
    oversized = tmp_path / "large.mp3"
    with oversized.open("wb") as stream:
        stream.truncate(25_000_001)
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {oversized}),
        timeout_seconds=1,
    )

    with pytest.raises(ValueError, match="audio size limit"):
        await backend.transcribe_audio("large.mp3")

    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_transcription_rejects_symlink_swap_without_leaking_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"registered audio")
    replacement = tmp_path / "replacement.mp3"
    replacement.write_bytes(b"unregistered audio")
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {audio}),
        timeout_seconds=1,
    )
    original_open = os.open

    def swap_then_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        audio.unlink()
        audio.symlink_to(replacement)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(mcp_server.os, "open", swap_then_open)

    with pytest.raises(ValueError) as caught:
        await backend.transcribe_audio("sample.mp3")

    assert str(caught.value) == "attachment could not be safely read"
    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_transcription_rejects_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"registered audio")
    replacement = tmp_path / "replacement.mp3"
    replacement.write_bytes(b"unregistered audio")
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {audio}),
        timeout_seconds=1,
    )
    original_open = os.open

    def replace_then_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        replacement.replace(audio)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(mcp_server.os, "open", replace_then_open)

    with pytest.raises(ValueError, match="safely read"):
        await backend.transcribe_audio("sample.mp3")

    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_transcription_rejects_parent_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = tmp_path / "task"
    task_root.mkdir()
    registered_dir = task_root / "registered"
    registered_dir.mkdir()
    audio = registered_dir / "sample.mp3"
    audio.write_bytes(b"registered audio")
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "sample.mp3").write_bytes(b"external audio")
    reader = AttachmentReader(task_root, {audio})
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=reader,
        timeout_seconds=1,
    )
    original_resolve = reader.resolve_registered

    def resolve_then_swap_parent(path: str) -> Path:
        resolved = original_resolve(path)
        registered_dir.rename(task_root / "registered-original")
        registered_dir.symlink_to(external_dir, target_is_directory=True)
        return resolved

    monkeypatch.setattr(reader, "resolve_registered", resolve_then_swap_parent)

    with pytest.raises(ValueError, match="safely read"):
        await backend.transcribe_audio("registered/sample.mp3")

    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_transcription_rejects_growth_after_snapshot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"registered audio")
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {audio}),
        timeout_seconds=1,
    )
    original_fstat = os.fstat
    calls = 0

    def grow_before_second_fstat(fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            with audio.open("ab") as stream:
                stream.write(b"growth")
        return original_fstat(fd)

    monkeypatch.setattr(mcp_server.os, "fstat", grow_before_second_fstat)

    with pytest.raises(ValueError, match="safely read"):
        await backend.transcribe_audio("sample.mp3")

    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_transcription_rejects_unregistered_non_audio_and_oversized(
    tmp_path: Path,
) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("not audio", encoding="utf-8")
    oversized = tmp_path / "large.mp3"
    with oversized.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {text, oversized}),
        timeout_seconds=1,
    )

    with pytest.raises(ValueError, match="registered"):
        await backend.transcribe_audio("missing.mp3")
    with pytest.raises(ValueError, match="audio"):
        await backend.transcribe_audio("notes.txt")
    with pytest.raises(ValueError, match="size limit"):
        await backend.transcribe_audio("large.mp3")

    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_code_interpreter_rejects_audio_before_files_api_upload(tmp_path: Path) -> None:
    audio = tmp_path / "sample.MP3"
    audio.write_bytes(b"registered audio")
    response = SimpleNamespace(output_text="unused", output=[], usage=None)
    client = FakeHostedClient(response)
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {audio}),
        timeout_seconds=1,
    )

    with pytest.raises(ValueError, match="transcribe_audio"):
        await backend.execute_python("inspect audio", ["sample.MP3"])

    assert client.files.created == []
    assert client.responses.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_transcription_respects_timeout(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"synthetic mp3")
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))

    async def delayed_create(*, model: str, file: Any) -> Any:
        await asyncio.sleep(1)
        return SimpleNamespace(text="late", usage=None)

    client.audio.transcriptions.create = delayed_create
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {audio}),
        timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await backend.transcribe_audio("sample.mp3")


@pytest.mark.asyncio
async def test_hosted_audio_snapshot_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_snapshot(reader: AttachmentReader, path: str) -> Any:
        started.set()
        release.wait(timeout=1)
        return mcp_server._NamedBytesIO(b"registered audio", "sample.mp3")

    monkeypatch.setattr(mcp_server, "_snapshot_registered_audio", blocking_snapshot)
    client = FakeHostedClient(SimpleNamespace(text="spoken answer", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=1,
    )
    timer = threading.Timer(0.2, release.set)
    timer.start()
    task = asyncio.create_task(backend.transcribe_audio("sample.mp3"))

    try:
        await asyncio.sleep(0.02)
        assert started.is_set()
        assert not release.is_set()
    finally:
        release.set()
        timer.cancel()
        await task


@pytest.mark.asyncio
async def test_hosted_audio_timeout_covers_snapshot_before_api_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_snapshot(reader: AttachmentReader, path: str) -> Any:
        time.sleep(0.1)
        return mcp_server._NamedBytesIO(b"registered audio", "sample.mp3")

    monkeypatch.setattr(mcp_server, "_snapshot_registered_audio", slow_snapshot)
    client = FakeHostedClient(SimpleNamespace(text="late", usage=None))
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=0.02,
    )

    with pytest.raises(TimeoutError):
        await backend.transcribe_audio("sample.mp3")

    assert client.audio.transcriptions.requests == []


@pytest.mark.asyncio
async def test_hosted_audio_uses_one_deadline_for_snapshot_and_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_snapshot(reader: AttachmentReader, path: str) -> Any:
        time.sleep(0.08)
        return mcp_server._NamedBytesIO(b"registered audio", "sample.mp3")

    api_started = asyncio.Event()

    async def slow_create(*, model: str, file: Any) -> Any:
        api_started.set()
        await asyncio.sleep(0.08)
        return SimpleNamespace(text="late", usage=None)

    monkeypatch.setattr(mcp_server, "_snapshot_registered_audio", slow_snapshot)
    client = FakeHostedClient(SimpleNamespace(text="unused", usage=None))
    client.audio.transcriptions.create = slow_create
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, set()),
        timeout_seconds=0.12,
    )

    with pytest.raises(TimeoutError):
        await backend.transcribe_audio("sample.mp3")

    assert api_started.is_set()


@pytest.mark.asyncio
async def test_code_interpreter_uses_only_registered_files_and_caches_uploads(
    tmp_path: Path,
) -> None:
    attachment = tmp_path / "data.csv"
    attachment.write_text("x\n42\n", encoding="utf-8")
    unregistered = tmp_path / "private.csv"
    unregistered.write_text("secret", encoding="utf-8")
    response = SimpleNamespace(
        output_text="42",
        usage=SimpleNamespace(
            input_tokens=17,
            output_tokens=7,
            total_tokens=24,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
        output=[
            SimpleNamespace(
                type="code_interpreter_call",
                outputs=[{"type": "logs", "logs": "42"}],
            )
        ],
    )
    client = FakeHostedClient(response)
    backend = HostedOpenAIBackend(
        client=client,
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        attachment_reader=AttachmentReader(tmp_path, {attachment}),
        timeout_seconds=1,
    )

    first = await backend.execute_python("calculate", ["data.csv"])
    second = await backend.execute_python("calculate again", ["data.csv"])

    assert len(client.files.created) == 1
    assert client.files.created == [("assistants", b"x\n42\n")]
    assert client.responses.requests[0] == {
        "model": "gpt-5.6-sol",
        "input": "calculate",
        "tools": [
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "memory_limit": "1g",
                    "file_ids": ["file-1"],
                },
            }
        ],
        "tool_choice": "required",
        "include": ["code_interpreter_call.outputs"],
        "reasoning": {"effort": "medium"},
    }
    assert first == {
        "output_text": "42",
        "outputs": [{"type": "logs", "logs": "42"}],
        "__gaia_usage__": {
            "input_tokens": 17,
            "output_tokens": 7,
            "reasoning_tokens": 3,
            "total_tokens": 24,
        },
    }
    assert second["output_text"] == "42"
    with pytest.raises(ValueError, match="not registered"):
        await backend.execute_python("steal", ["private.csv"])


def test_attachment_reader_extracts_json_and_returns_media_metadata(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text('{"answer": 42}', encoding="utf-8")
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n")
    reader = AttachmentReader(tmp_path, {data, image})

    json_result = reader.read("data.json")
    image_result = reader.read("photo.png")

    assert json_result["kind"] == "json"
    assert json_result["data"] == {"answer": 42}
    assert image_result == {
        "path": "photo.png",
        "kind": "image",
        "media_type": "image/png",
        "size_bytes": 6,
    }


def test_attachment_reader_parses_csv_rows(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text('name,value\n"alpha, beta",42\n', encoding="utf-8")
    reader = AttachmentReader(tmp_path, {data})

    result = reader.read("data.csv")

    assert result == {
        "path": "data.csv",
        "kind": "csv",
        "rows": [["name", "value"], ["alpha, beta", "42"]],
    }


def test_attachment_reader_serializes_xlsx_cell_types(tmp_path: Path) -> None:
    data = tmp_path / "data.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Answers"
    sheet.append(["when", "answer"])
    sheet.append([date(2026, 8, 2), 42])
    workbook.save(data)
    reader = AttachmentReader(tmp_path, {data})

    result = reader.read("data.xlsx")

    assert result == {
        "path": "data.xlsx",
        "kind": "xlsx",
        "sheets": {
            "Answers": [["when", "answer"], ["2026-08-02 00:00:00", 42]],
        },
    }


def test_attachment_reader_enforces_csv_row_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data.csv"
    data.write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "MAX_TABULAR_ROWS", 1, raising=False)
    reader = AttachmentReader(tmp_path, {data})

    with pytest.raises(ValueError, match="resource limit.*CSV rows"):
        reader.read("data.csv")


def test_attachment_reader_enforces_json_decoded_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data.json"
    data.write_text('{"value":"1234"}', encoding="utf-8")
    monkeypatch.setattr(mcp_server, "MAX_DECODED_TEXT_BYTES", 8, raising=False)
    reader = AttachmentReader(tmp_path, {data})

    with pytest.raises(ValueError, match="resource limit.*decoded bytes"):
        reader.read("data.json")


def test_attachment_reader_enforces_pdf_page_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with data.open("wb") as stream:
        writer.write(stream)
    monkeypatch.setattr(mcp_server, "MAX_PDF_PAGES", 1, raising=False)
    reader = AttachmentReader(tmp_path, {data})

    with pytest.raises(ValueError, match="resource limit.*PDF pages"):
        reader.read("data.pdf")


def test_attachment_reader_bounds_pdf_stream_decompression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "compressed.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=72, height=72)
    stream = DecodedStreamObject()
    stream.set_data(b"q\n" * 100)
    page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    with data.open("wb") as output:
        writer.write(output)
    monkeypatch.setattr(mcp_server, "MAX_PDF_DECODED_STREAM_BYTES", 16, raising=False)
    reader = AttachmentReader(tmp_path, {data})

    with pytest.raises(ValueError, match="resource limit.*PDF decoded stream bytes"):
        reader.read("compressed.pdf")


def test_attachment_reader_rejects_xlsx_expansion_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data.xlsx"
    workbook = Workbook()
    workbook.active.append(["small fixture"])
    workbook.save(data)
    monkeypatch.setattr(mcp_server, "MAX_XLSX_UNCOMPRESSED_BYTES", 1, raising=False)
    reader = AttachmentReader(tmp_path, {data})

    with pytest.raises(ValueError, match="resource limit.*XLSX expanded bytes"):
        reader.read("data.xlsx")


@pytest.mark.asyncio
async def test_attachment_resource_limit_is_a_structured_mcp_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data.csv"
    data.write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "MAX_TABULAR_ROWS", 1, raising=False)
    settings = MCPServerSettings.for_fake_backend(
        task_root=tmp_path,
        attachments={data},
        api_key="test",
        hf_token="test-hf",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        timeout_seconds=1,
    )

    async with Client(build_server(settings)) as client:
        result = await client.call_tool("read_attachment", {"path": "data.csv"})

    assert result.is_error is True
    assert "resource limit" in result.content[0].text


@pytest.mark.asyncio
async def test_build_server_exposes_exact_public_tool_plane(tmp_path: Path) -> None:
    settings = MCPServerSettings.for_fake_backend(
        task_root=tmp_path,
        attachments=set(),
        api_key="test",
        hf_token="test-hf",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        timeout_seconds=1,
    )
    server = build_server(settings)

    assert server.name == "gaia-tools"
    assert [tool.name for tool in await server.list_tools()] == [
        "web_search",
        "read_webpage",
        "execute_python",
        "read_attachment",
        "transcribe_audio",
        "submit_answer",
    ]


@pytest.mark.parametrize(
    (
        "global_timeout",
        "specialist_timeout",
        "expected_webpage_timeout",
        "expected_hosted_web_timeout",
    ),
    [(300.0, 210.0, 20.0, 60.0), (12.5, 8.75, 12.5, 8.75)],
)
def test_settings_caps_direct_webpage_timeout_without_shortening_hosted_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    global_timeout: float,
    specialist_timeout: float,
    expected_webpage_timeout: float,
    expected_hosted_web_timeout: float,
) -> None:
    class CapturingWebReader:
        def __init__(self, *, timeout_seconds: float) -> None:
            self.timeout_seconds = timeout_seconds

    monkeypatch.setenv("GAIA_TASK_ROOT", str(tmp_path))
    monkeypatch.setenv("GAIA_ATTACHMENTS", "[]")
    monkeypatch.setenv("GAIA_TIMEOUT_SECONDS", str(global_timeout))
    monkeypatch.setenv("GAIA_SPECIALIST_TIMEOUT_SECONDS", str(specialist_timeout))
    monkeypatch.delenv("GAIA_MCP_FAKE_BACKEND", raising=False)
    monkeypatch.setattr(mcp_server, "HTTPWebReader", CapturingWebReader)
    monkeypatch.setattr(
        mcp_server,
        "AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(config=kwargs),
    )

    settings = mcp_server._settings_from_env()

    assert settings.web_reader.timeout_seconds == expected_webpage_timeout
    assert isinstance(settings.backend, HostedOpenAIBackend)
    assert settings.backend._timeout_seconds == global_timeout
    assert settings.backend._web_search_timeout_seconds == expected_hosted_web_timeout
    assert isinstance(settings.search_provider, SearchProviderChain)
    assert settings.search_provider._operation_timeout_seconds == specialist_timeout


def test_fake_settings_caps_direct_webpage_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingWebReader:
        def __init__(self, *, timeout_seconds: float) -> None:
            self.timeout_seconds = timeout_seconds

    monkeypatch.setattr(mcp_server, "HTTPWebReader", CapturingWebReader)

    settings = MCPServerSettings.for_fake_backend(
        task_root=tmp_path,
        attachments=set(),
        api_key="test",
        hf_token="test-hf",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
        timeout_seconds=300,
    )

    assert settings.web_reader.timeout_seconds == 20
