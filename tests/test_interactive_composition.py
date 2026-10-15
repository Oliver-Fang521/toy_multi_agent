import json
from contextlib import asynccontextmanager

import pytest

from gaia_agent.interactive.composition import (
    bounded_history,
    build_interactive_orchestrator,
    execute_interactive,
)
from gaia_agent.interactive.models import InteractiveConfig, InteractiveRequest


def test_interactive_config_needs_no_huggingface_token():
    config = InteractiveConfig(openai_api_key="fake")
    assert config.model == "gpt-6.1-sol"
    assert config.reasoning_effort == "low"
    assert config.timeout_seconds == 500
    assert config.max_tool_calls == 24


def test_each_interactive_run_owns_distinct_agents(fake_provider, fake_tools, tmp_path):
    config = InteractiveConfig(openai_api_key="fake")
    first = build_interactive_orchestrator(
        config, tmp_path / "first", provider=fake_provider, tools_factory=lambda task: fake_tools
    )
    second = build_interactive_orchestrator(
        config, tmp_path / "second", provider=fake_provider, tools_factory=lambda task: fake_tools
    )
    assert first._planner is not second._planner
    assert first._critic is not second._critic


@pytest.mark.asyncio
async def test_interactive_returns_answer_evidence_and_progress(
    fake_provider, fake_tools, tmp_path
):
    fake_provider.return_parsed(
        {
            "restatement": "Find fact",
            "steps": [{"specialist": "lookup", "instruction": "Find evidence"}],
        }
    )
    fake_provider.return_parsed(
        {
            "specialist": "lookup",
            "candidate_answer": "Marie Curie",
            "evidence": [
                {
                    "evidence_id": "source-1",
                    "kind": "source",
                    "summary": "Observed fact",
                    "source_url": "https://example.org/source",
                }
            ],
        }
    )
    fake_provider.return_parsed(
        {
            "verdict": "approve",
            "feedback": "supported",
            "checked_evidence_ids": ["evidence-0001"],
            "corrected_answer": None,
        }
    )

    async def submit(name, arguments):
        return {"answer": arguments["answer"], "evidence_ids": arguments["evidence_ids"]}

    fake_tools.call = submit

    @asynccontextmanager
    async def tools_factory(task):
        yield fake_tools

    progress = []
    result = await execute_interactive(
        InteractiveRequest(question="Who?"),
        InteractiveConfig(openai_api_key="fake"),
        tmp_path,
        provider=fake_provider,
        tools_factory=tools_factory,
        on_progress=lambda event, payload: progress.append(event),
    )
    assert result.answer == "Marie Curie"
    assert result.evidence[0].source_url == "https://example.org/source"
    assert "plan_started" in progress
    assert "submit_completed" in progress


def test_followup_context_is_bounded():
    history = [{"role": "user", "content": "x" * 20_000}] * 100
    result = bounded_history(history)
    assert len(result.encode()) < 6500
    assert len(json.loads(result)) <= 8
