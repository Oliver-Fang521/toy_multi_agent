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
    build_server,
)
from gaia_agent.models import Usage


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


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve_public(host: str, port: int, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.72.14", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve_public)


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
        with pytest.raises(httpx.HTTPStatusError):
            await reader.read("https://example.test/down")

    assert attempts == ["93.184.216.34"]


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
    ("global_timeout", "expected_webpage_timeout", "expected_hosted_web_timeout"),
    [(300.0, 20.0, 60.0), (12.5, 12.5, 12.5)],
)
def test_settings_caps_direct_webpage_timeout_without_shortening_hosted_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    global_timeout: float,
    expected_webpage_timeout: float,
    expected_hosted_web_timeout: float,
) -> None:
    class CapturingWebReader:
        def __init__(self, *, timeout_seconds: float) -> None:
            self.timeout_seconds = timeout_seconds

    monkeypatch.setenv("GAIA_TASK_ROOT", str(tmp_path))
    monkeypatch.setenv("GAIA_ATTACHMENTS", "[]")
    monkeypatch.setenv("GAIA_TIMEOUT_SECONDS", str(global_timeout))
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
