from __future__ import annotations

import asyncio
import csv
import ipaddress
import json
import mimetypes
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from mcp.server import MCPServer
from openai import AsyncOpenAI
from openpyxl import load_workbook
from pypdf import PdfReader

MAX_TOOL_OUTPUT_BYTES = 64 * 1024
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
_SENSITIVE_KEY_PARTS = ("api_key", "authorization", "credential", "password", "secret", "token")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _replace_secrets(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
                else _redact(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, secrets) for item in value)
    if isinstance(value, str):
        return _replace_secrets(value, secrets)
    return value


def _bounded_result(result: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) <= MAX_TOOL_OUTPUT_BYTES:
        return result
    low = 0
    high = min(len(encoded), MAX_TOOL_OUTPUT_BYTES)
    best: dict[str, Any] = {"output": "", "truncated": True}
    while low <= high:
        middle = (low + high) // 2
        candidate = {
            "output": encoded[:middle].decode("utf-8", errors="ignore"),
            "truncated": True,
        }
        candidate_size = len(
            json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if candidate_size <= MAX_TOOL_OUTPUT_BYTES:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


class WebReader(Protocol):
    async def read(self, url: str) -> dict[str, Any]: ...


class HostedBackend(Protocol):
    async def web_search(self, query: str) -> dict[str, Any]: ...

    async def execute_python(self, instruction: str, paths: list[str]) -> dict[str, Any]: ...


class HTTPWebReader:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 20,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> None:
        self._client = client or httpx.AsyncClient(follow_redirects=True)
        self._owns_client = client is None
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes

    async def read(self, url: str) -> dict[str, Any]:
        _validate_public_http_url(url)
        async with self._client.stream(
            "GET", url, follow_redirects=True, timeout=self._timeout_seconds
        ) as response:
            response.raise_for_status()
            _validate_public_http_url(str(response.url))
            content = bytearray()
            truncated = False
            async for chunk in response.aiter_bytes():
                remaining = self._max_bytes - len(content)
                if len(chunk) > remaining:
                    content.extend(chunk[:remaining])
                    truncated = True
                    break
                content.extend(chunk)
            media_type = response.headers.get("content-type", "")
            text = bytes(content).decode(response.encoding or "utf-8", errors="replace")
            if "html" in media_type.lower():
                soup = BeautifulSoup(text, "html.parser")
                for element in soup(["script", "style", "noscript"]):
                    element.decompose()
                text = " ".join(soup.get_text(" ", strip=True).split())
            return {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": media_type,
                "text": text,
                "truncated": truncated,
            }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use HTTP(S)")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are not allowed")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("URL must refer to a public host")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("URL must refer to a public host")


class AttachmentReader:
    def __init__(self, task_root: Path, registered: set[Path] | frozenset[Path]) -> None:
        self.task_root = task_root.resolve()
        resolved: set[Path] = set()
        for path in registered:
            candidate = path if path.is_absolute() else self.task_root / path
            candidate = candidate.resolve()
            if not _is_relative_to(candidate, self.task_root):
                raise ValueError(f"attachment is outside task root: {path}")
            resolved.add(candidate)
        self.registered = frozenset(resolved)

    def resolve_registered(self, path: str) -> Path:
        candidate = (self.task_root / path).resolve()
        if not _is_relative_to(candidate, self.task_root):
            raise ValueError(f"attachment path is outside task root: {path}")
        if candidate not in self.registered:
            raise ValueError(f"attachment is not registered for this task: {path}")
        if not candidate.is_file():
            raise ValueError(f"attachment does not exist or is not a file: {path}")
        if candidate.stat().st_size > MAX_ATTACHMENT_BYTES:
            raise ValueError(f"attachment exceeds size limit: {path}")
        return candidate

    def read(self, path: str) -> dict[str, Any]:
        resolved = self.resolve_registered(path)
        relative = str(resolved.relative_to(self.task_root))
        suffix = resolved.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = resolved.read_text(encoding="utf-8")
            return {"path": relative, "kind": "text", "text": text}
        if suffix in {".csv", ".tsv"}:
            with resolved.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream, delimiter="\t" if suffix == ".tsv" else ","))
            return {"path": relative, "kind": "csv", "rows": rows}
        if suffix == ".json":
            return {
                "path": relative,
                "kind": "json",
                "data": json.loads(resolved.read_text(encoding="utf-8")),
            }
        if suffix == ".pdf":
            pages = [page.extract_text() or "" for page in PdfReader(resolved).pages]
            return {
                "path": relative,
                "kind": "pdf",
                "page_count": len(pages),
                "text": "\n\n".join(pages),
            }
        if suffix == ".xlsx":
            workbook = load_workbook(resolved, read_only=True, data_only=True)
            sheets: dict[str, list[list[Any]]] = {}
            try:
                for sheet in workbook.worksheets:
                    sheets[sheet.title] = [
                        [_jsonable(value) for value in row]
                        for row in sheet.iter_rows(values_only=True)
                    ]
            finally:
                workbook.close()
            return {"path": relative, "kind": "xlsx", "sheets": sheets}
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        if media_type.startswith("image/"):
            kind = "image"
        elif media_type.startswith("audio/"):
            kind = "audio"
        else:
            kind = "binary"
        return {
            "path": relative,
            "kind": kind,
            "media_type": media_type,
            "size_bytes": resolved.stat().st_size,
        }


class HostedOpenAIBackend:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        reasoning_effort: str,
        attachment_reader: AttachmentReader,
        timeout_seconds: float,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._attachments = attachment_reader
        self._timeout_seconds = timeout_seconds
        self._remote_file_ids: dict[Path, str] = {}

    async def web_search(self, query: str) -> dict[str, Any]:
        response = await asyncio.wait_for(
            self._client.responses.create(
                model=self._model,
                input=query,
                tools=[{"type": "web_search"}],
                tool_choice="required",
                include=["web_search_call.action.sources"],
                reasoning={"effort": self._reasoning_effort},
            ),
            timeout=self._timeout_seconds,
        )
        sources: list[dict[str, Any]] = []
        for item in response.output:
            if _field(item, "type") != "web_search_call":
                continue
            action = _field(item, "action")
            for source in _field(action, "sources", []) or []:
                url = _field(source, "url")
                if url:
                    sources.append({"url": url, "title": _field(source, "title")})
        return {"answer": response.output_text, "sources": sources}

    async def execute_python(self, instruction: str, paths: list[str]) -> dict[str, Any]:
        file_ids = [await self._upload_registered(path) for path in paths]
        response = await asyncio.wait_for(
            self._client.responses.create(
                model=self._model,
                input=instruction,
                tools=[
                    {
                        "type": "code_interpreter",
                        "container": {
                            "type": "auto",
                            "memory_limit": "1g",
                            "file_ids": file_ids,
                        },
                    }
                ],
                tool_choice="required",
                include=["code_interpreter_call.outputs"],
                reasoning={"effort": self._reasoning_effort},
            ),
            timeout=self._timeout_seconds,
        )
        outputs: list[Any] = []
        for item in response.output:
            if _field(item, "type") == "code_interpreter_call":
                outputs.extend(_jsonable(output) for output in (_field(item, "outputs", []) or []))
        return {"output_text": response.output_text, "outputs": outputs}

    async def _upload_registered(self, path: str) -> str:
        resolved = self._attachments.resolve_registered(path)
        if remote_id := self._remote_file_ids.get(resolved):
            return remote_id
        with resolved.open("rb") as stream:
            uploaded = await asyncio.wait_for(
                self._client.files.create(file=stream, purpose="assistants"),
                timeout=self._timeout_seconds,
            )
        self._remote_file_ids[resolved] = uploaded.id
        return uploaded.id


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


class FakeHostedBackend:
    def __init__(self, api_key: str, hf_token: str) -> None:
        self._api_key = api_key
        self._hf_token = hf_token

    async def web_search(self, query: str) -> dict[str, Any]:
        if query == "__GAIA_TIMEOUT__":
            await asyncio.sleep(60)
        if query == "__GAIA_SECRET_ERROR__":
            raise RuntimeError(f"backend rejected {self._api_key} and {self._hf_token}")
        if query == "__GAIA_NESTED_SECRET__":
            return {
                "answer": "safe",
                "sources": [],
                "debug": {
                    "api_key": self._api_key,
                    "message": f"{self._api_key} and {self._hf_token}",
                },
            }
        return {
            "answer": f"deterministic search result for {query}",
            "sources": [{"url": "https://example.test/source", "title": "Example"}],
        }

    async def execute_python(self, instruction: str, paths: list[str]) -> dict[str, Any]:
        if instruction == "__GAIA_LARGE_OUTPUT__":
            return {"output_text": "x" * (MAX_TOOL_OUTPUT_BYTES + 10_000), "outputs": []}
        if instruction == "__GAIA_ESCAPED_OUTPUT__":
            return {"output_text": '"\\' * MAX_TOOL_OUTPUT_BYTES, "outputs": []}
        return {"output_text": f"deterministic code result for {instruction}", "outputs": []}


@dataclass(frozen=True)
class MCPServerSettings:
    task_root: Path
    attachment_reader: AttachmentReader
    web_reader: WebReader
    backend: HostedBackend
    api_key: str
    hf_token: str

    @property
    def secrets(self) -> tuple[str, ...]:
        return (self.api_key, self.hf_token)

    @classmethod
    def for_fake_backend(
        cls,
        *,
        task_root: Path,
        attachments: set[Path],
        api_key: str,
        hf_token: str,
        model: str,
        reasoning_effort: str,
        timeout_seconds: float,
    ) -> MCPServerSettings:
        attachment_reader = AttachmentReader(task_root, attachments)
        return cls(
            task_root=task_root.resolve(),
            attachment_reader=attachment_reader,
            web_reader=HTTPWebReader(timeout_seconds=timeout_seconds),
            backend=FakeHostedBackend(api_key, hf_token),
            api_key=api_key,
            hf_token=hf_token,
        )


async def _safe_tool_call(
    settings: MCPServerSettings,
    operation: Awaitable[dict[str, Any]] | Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = await operation if hasattr(operation, "__await__") else operation()
        return _bounded_result(_redact(result, settings.secrets))
    # This is the server boundary: redact arbitrary backend/parser failures before
    # MCP serializes them into consumer-visible tool errors.
    except Exception as error:  # noqa: BLE001
        message = _replace_secrets(str(error), settings.secrets)
        raise RuntimeError(message) from None


def build_server(settings: MCPServerSettings) -> MCPServer:
    server = MCPServer("gaia-tools")

    @server.tool()
    async def web_search(query: str) -> dict[str, Any]:
        """Search the public web and return a concise answer with source URLs."""
        return await _safe_tool_call(settings, settings.backend.web_search(query))

    @server.tool()
    async def read_webpage(url: str) -> dict[str, Any]:
        """Read public HTTP(S) page text with redirects, timeout, and size limits."""
        return await _safe_tool_call(settings, settings.web_reader.read(url))

    @server.tool()
    async def execute_python(instruction: str, paths: list[str] | None = None) -> dict[str, Any]:
        """Run a Python analysis instruction in hosted OpenAI Code Interpreter."""
        return await _safe_tool_call(
            settings, settings.backend.execute_python(instruction, paths or [])
        )

    @server.tool()
    async def read_attachment(path: str) -> dict[str, Any]:
        """Extract text and metadata from an attachment registered for this task."""
        return await _safe_tool_call(settings, lambda: settings.attachment_reader.read(path))

    @server.tool()
    async def submit_answer(answer: str, evidence_ids: list[str]) -> dict[str, Any]:
        """Return the concise terminal answer and evidence references."""
        return await _safe_tool_call(
            settings, lambda: {"answer": answer.strip(), "evidence_ids": evidence_ids}
        )

    return server


def _settings_from_env() -> MCPServerSettings:
    task_root = Path(os.environ["GAIA_TASK_ROOT"]).resolve()
    raw_attachments = json.loads(os.environ.get("GAIA_ATTACHMENTS", "[]"))
    if not isinstance(raw_attachments, list) or not all(
        isinstance(path, str) for path in raw_attachments
    ):
        raise ValueError("GAIA_ATTACHMENTS must be a JSON list of paths")
    attachment_reader = AttachmentReader(task_root, {Path(path) for path in raw_attachments})
    api_key = os.environ.get("OPENAI_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    timeout_seconds = float(os.environ.get("GAIA_TIMEOUT_SECONDS", "300"))
    if os.environ.get("GAIA_MCP_FAKE_BACKEND") == "1":
        backend: HostedBackend = FakeHostedBackend(api_key, hf_token)
    else:
        client = AsyncOpenAI(api_key=api_key, max_retries=0)
        backend = HostedOpenAIBackend(
            client=client,
            model=os.environ.get("GAIA_MODEL", "gpt-5.6-sol"),
            reasoning_effort=os.environ.get("GAIA_REASONING_EFFORT", "medium"),
            attachment_reader=attachment_reader,
            timeout_seconds=timeout_seconds,
        )
    return MCPServerSettings(
        task_root=task_root,
        attachment_reader=attachment_reader,
        web_reader=HTTPWebReader(timeout_seconds=timeout_seconds),
        backend=backend,
        api_key=api_key,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    build_server(_settings_from_env()).run()
