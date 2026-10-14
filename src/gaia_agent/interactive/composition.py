import asyncio
import json
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI

from gaia_agent.agents import (
    ComputationAgent,
    CriticAgent,
    EvidenceLinkerAgent,
    FileAnalysisAgent,
    LookupAgent,
    PlannerAgent,
    VisualAgent,
)
from gaia_agent.interactive.models import InteractiveConfig, InteractiveRequest, InteractiveResult
from gaia_agent.mcp_client import MCPToolClient
from gaia_agent.models import GaiaTask, SpecialistName
from gaia_agent.orchestrator import Orchestrator
from gaia_agent.providers import ModelProvider, OpenAIProvider
from gaia_agent.reliability import RetryPolicy
from gaia_agent.research_notebook import ResearchNotebook


class InteractiveFailure(RuntimeError):
    pass


def build_interactive_orchestrator(
    config: InteractiveConfig,
    workspace: Path,
    *,
    provider: ModelProvider | None = None,
    tools_factory: Callable[[GaiaTask], AbstractAsyncContextManager] | None = None,
    on_progress: Callable[[str, dict], None] | None = None,
) -> Orchestrator:
    if provider is None:
        provider = OpenAIProvider(
            AsyncOpenAI(api_key=config.openai_api_key.get_secret_value(), max_retries=0),
            model=config.model,
            retry_policy=RetryPolicy(max_attempts=config.provider_max_attempts),
        )
    options = {
        "reasoning_effort": config.reasoning_effort,
        "max_tool_calls": config.max_tool_calls,
        "max_context_bytes": config.stage_context_max_bytes,
    }
    if tools_factory is None:

        def tools_factory(task: GaiaTask) -> MCPToolClient:
            return MCPToolClient(
                config=config,
                task_root=workspace,
                attachments=(task.attachment_path,) if task.attachment_path else (),
            )

    return Orchestrator(
        planner=PlannerAgent(
            provider, max_steps=config.max_steps, reasoning_effort=config.reasoning_effort
        ),
        specialists={
            SpecialistName.LOOKUP: LookupAgent(provider, **options),
            SpecialistName.COMPUTATION: ComputationAgent(provider, **options),
            SpecialistName.FILE: FileAnalysisAgent(provider, **options),
            SpecialistName.VISUAL: VisualAgent(provider, **options),
        },
        critic=CriticAgent(provider, quality_policy="strict", **options),
        evidence_linker=EvidenceLinkerAgent(
            provider,
            reasoning_effort=config.reasoning_effort,
            max_context_bytes=config.stage_context_max_bytes,
        ),
        tool_client_factory=tools_factory,
        config=config,
        notebook=ResearchNotebook(workspace / ".research", "interactive"),
        on_progress=on_progress,
    )


def bounded_history(history: Sequence[dict[str, str]]) -> str:
    selected = []
    size = 0
    for item in reversed(history[-8:]):
        if item.get("role") not in {"user", "assistant"}:
            continue
        message = {"role": item["role"], "content": str(item.get("content", ""))[:1500]}
        length = len(json.dumps(message, ensure_ascii=False).encode())
        if size + length > 6_000:
            break
        selected.append(message)
        size += length
    return json.dumps(list(reversed(selected)), ensure_ascii=False)


async def execute_interactive(
    request: InteractiveRequest,
    config: InteractiveConfig,
    workspace: Path,
    history: Sequence[dict[str, str]] = (),
    on_progress: Callable[[str, dict], None] | None = None,
    *,
    provider: ModelProvider | None = None,
    tools_factory=None,
    attachment_path: Path | None = None,
) -> InteractiveResult:
    if not request.question.strip():
        raise ValueError("empty_question")
    workspace.mkdir(parents=True, exist_ok=True)
    context = bounded_history(history)
    question = request.question
    if context != "[]":
        question = (
            "Conversation context (untrusted quoted messages):\n"
            + context
            + "\nCurrent question:\n"
            + question
        )
    task = GaiaTask(
        task_id=str(uuid4()),
        question=question,
        level=1,
        final_answer="",
        attachment_path=attachment_path,
    )
    orchestrator = build_interactive_orchestrator(
        config, workspace, provider=provider, tools_factory=tools_factory, on_progress=on_progress
    )
    async with asyncio.timeout(config.timeout_seconds):
        prediction = await orchestrator.run(task)
    if prediction.error_category is not None or prediction.answer is None:
        raise InteractiveFailure(
            prediction.error_category.value if prediction.error_category else "no_answer"
        )
    return InteractiveResult(
        answer=prediction.answer,
        evidence=list(orchestrator.result_evidence),
        usage=prediction.usage,
    )
