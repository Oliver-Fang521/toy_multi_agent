import asyncio
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from gaia_agent.models import GaiaTask, Usage
from gaia_agent.providers import ModelRequest, ModelResponse, ToolEvent, ToolSpec


class FakeProvider:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.executors: list[object | None] = []
        self._responses: deque[ModelResponse] = deque()

    def return_parsed(
        self,
        parsed: dict[str, Any],
        *,
        usage: Usage | None = None,
        tool_events: tuple[ToolEvent, ...] = (),
    ) -> None:
        self._responses.append(
            ModelResponse(
                parsed=parsed,
                usage=usage or Usage(),
                tool_events=tool_events,
            )
        )

    async def complete(
        self,
        request: ModelRequest,
        executor: object | None = None,
    ) -> ModelResponse:
        self.requests.append(request)
        self.executors.append(executor)
        return self._responses.popleft()


class FakeToolClient:
    def __init__(self) -> None:
        self.tool_specs = (
            ToolSpec(
                name="web_search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="read_webpage",
                description="Read a webpage",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="execute_python",
                description="Execute Python in the hosted sandbox",
                parameters={
                    "type": "object",
                    "properties": {"instruction": {"type": "string"}},
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="read_attachment",
                description="Read a registered task attachment",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="submit_answer",
                description="Submit a terminal answer",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            ),
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        raise AssertionError("fake provider must not call tools unless a test explicitly enables it")

    async def read_attachment_bytes(self, path: Path) -> bytes:
        return await asyncio.to_thread(path.read_bytes)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def fake_tools() -> FakeToolClient:
    return FakeToolClient()


@pytest.fixture
def task() -> GaiaTask:
    return GaiaTask(
        task_id="synthetic-agent-task",
        question="Who discovered the synthetic element?",
        level=1,
        final_answer="Hidden Reference Answer",
    )


@pytest.fixture
def image_task(tmp_path: Path) -> GaiaTask:
    image_path = tmp_path / "synthetic.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return GaiaTask(
        task_id="synthetic-image-task",
        question="What is visible in the image?",
        level=2,
        final_answer="Hidden Image Answer",
        attachment_path=image_path,
    )
