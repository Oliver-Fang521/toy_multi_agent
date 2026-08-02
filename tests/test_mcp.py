import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openpyxl import Workbook

from gaia_agent.config import RunConfig
from gaia_agent.mcp_client import MCPToolClient, ToolExecutionError
from gaia_agent.mcp_server import (
    AttachmentReader,
    HostedOpenAIBackend,
    HTTPWebReader,
    MCPServerSettings,
    build_server,
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


@pytest.fixture
def fake_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIA_MCP_FAKE_BACKEND", "1")


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

    with pytest.raises(ToolExecutionError, match="not connected"):
        await client.call("submit_answer", {"answer": "42", "evidence_ids": []})


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
async def test_http_reader_extracts_html_text_and_final_url() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://example.test/article")
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
async def test_http_reader_raises_for_http_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="backend unavailable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = HTTPWebReader(client=client, timeout_seconds=1)
        with pytest.raises(httpx.HTTPStatusError):
            await reader.read("https://example.test/down")


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


class FakeHostedClient:
    def __init__(self, response: Any) -> None:
        self.responses = FakeResponses(response)
        self.files = FakeFiles()


@pytest.mark.asyncio
async def test_hosted_search_requires_web_tool_and_extracts_sources(tmp_path: Path) -> None:
    response = SimpleNamespace(
        output_text="The answer",
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
    }


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
    assert first == {"output_text": "42", "outputs": [{"type": "logs", "logs": "42"}]}
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
        "submit_answer",
    ]
