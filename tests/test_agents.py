import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from gaia_agent.agents import (
    ComputationAgent,
    CriticAgent,
    FileAnalysisAgent,
    LookupAgent,
    PlannerAgent,
    VisualAgent,
)
from gaia_agent.models import Evidence, PlanStep, SpecialistName, Usage
from gaia_agent.providers import ModelRequest, ModelResponse, ToolEvent


@pytest.mark.asyncio
async def test_planner_emits_bounded_specialist_steps_without_tools(fake_provider, task) -> None:
    fake_provider.return_parsed(
        {
            "restatement": "Find and verify the requested fact",
            "steps": [{"specialist": "lookup", "instruction": "Find a primary source"}],
        }
    )

    plan = await PlannerAgent(fake_provider, max_steps=4).plan(task)

    assert plan.steps[0].specialist == SpecialistName.LOOKUP
    assert fake_provider.requests[0].allowed_tools == ()
    assert fake_provider.requests[0].max_tool_calls == 0
    assert fake_provider.executors == [None]


@pytest.mark.asyncio
async def test_planner_repairs_one_malformed_response_locally(fake_provider, task) -> None:
    fake_provider.return_text("not valid planner output")
    fake_provider.return_parsed(
        {
            "restatement": "Find and verify the requested fact",
            "steps": [{"specialist": "lookup", "instruction": "Find a primary source"}],
        }
    )

    plan = await PlannerAgent(fake_provider, max_steps=4).plan(task)

    assert plan.steps[0].specialist == SpecialistName.LOOKUP
    assert len(fake_provider.requests) == 2
    assert fake_provider.requests[1].allowed_tools == ()
    assert fake_provider.requests[1].max_tool_calls == 0
    assert "only the JSON object" in fake_provider.requests[1].instructions


@pytest.mark.asyncio
async def test_planner_falls_back_to_a_deterministic_route_after_two_failures(
    fake_provider, task
) -> None:
    fake_provider.return_text("not valid planner output")
    fake_provider.return_text("still not valid planner output")

    plan = await PlannerAgent(fake_provider, max_steps=4).plan(task)

    assert plan.steps[0].specialist == SpecialistName.LOOKUP
    assert plan.steps[0].instruction
    assert len(fake_provider.requests) == 2


@pytest.mark.asyncio
async def test_max_reasoning_uses_a_larger_output_budget(fake_provider, task) -> None:
    fake_provider.return_parsed(
        {
            "restatement": "Find and verify the requested fact",
            "steps": [{"specialist": "lookup", "instruction": "Find a primary source"}],
        }
    )

    await PlannerAgent(fake_provider, reasoning_effort="max").plan(task)

    assert fake_provider.requests[0].max_output_tokens == 8_000


@pytest.mark.asyncio
async def test_non_max_reasoning_keeps_the_default_output_budget(fake_provider, task) -> None:
    fake_provider.return_parsed(
        {
            "restatement": "Find and verify the requested fact",
            "steps": [{"specialist": "lookup", "instruction": "Find a primary source"}],
        }
    )

    await PlannerAgent(fake_provider, reasoning_effort="high").plan(task)

    assert fake_provider.requests[0].max_output_tokens == 2_000


def test_specialist_tool_allowlists_are_narrow() -> None:
    assert LookupAgent.allowed_tool_names == frozenset({"web_search", "read_webpage"})
    assert ComputationAgent.allowed_tool_names == frozenset({"execute_python"})
    assert FileAnalysisAgent.allowed_tool_names == frozenset(
        {"read_attachment", "execute_python", "transcribe_audio"}
    )
    assert VisualAgent.allowed_tool_names == frozenset({"read_attachment"})


def test_lookup_prompt_bounds_web_research_before_synthesis() -> None:
    prompt = LookupAgent.instructions.casefold()

    assert "at most three web searches" in prompt
    assert "at most three webpage reads" in prompt
    assert "then synthesize" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("steps", [[], [{"specialist": "lookup", "instruction": "x"}] * 5])
async def test_planner_rejects_empty_or_oversized_plans(fake_provider, task, steps) -> None:
    fake_provider.return_parsed({"restatement": "Bounded plan", "steps": steps})

    with pytest.raises(ValueError, match="between 1 and 4"):
        await PlannerAgent(fake_provider, max_steps=4).plan(task)


@pytest.mark.asyncio
async def test_planner_request_has_a_recursive_strict_schema_and_baked_in_bounds(
    fake_provider, task
) -> None:
    fake_provider.return_parsed(
        {
            "restatement": "Use lookup",
            "steps": [{"specialist": "lookup", "instruction": "Find a source"}],
        }
    )

    await PlannerAgent(fake_provider, max_steps=4).plan(task)
    schema = fake_provider.requests[0].output_schema

    _assert_all_object_schemas_are_strict(schema)
    assert schema["properties"]["steps"]["minItems"] == 1
    assert schema["properties"]["steps"]["maxItems"] == 4


@pytest.mark.asyncio
async def test_planner_context_is_deterministic_and_excludes_reference_answer(
    fake_provider, image_task
) -> None:
    parsed = {
        "restatement": "Inspect the image",
        "steps": [{"specialist": "visual", "instruction": "Inspect visible evidence"}],
    }
    fake_provider.return_parsed(parsed)
    fake_provider.return_parsed(parsed)
    planner = PlannerAgent(fake_provider, max_steps=2)

    await planner.plan(image_task)
    await planner.plan(image_task)
    first = fake_provider.requests[0].input_items
    second = fake_provider.requests[1].input_items
    context = json.loads(first[0]["content"][0]["text"])

    assert first == second
    assert context == {
        "attachment": "synthetic.png",
        "level": 2,
        "question": "What is visible in the image?",
        "task_id": "synthetic-image-task",
    }
    assert "Hidden Image Answer" not in repr(first)


@pytest.mark.asyncio
async def test_planner_routes_visual_steps_to_file_for_non_vision_provider(
    fake_provider, image_task
) -> None:
    fake_provider.supports_vision = False
    fake_provider.return_parsed(
        {
            "restatement": "Inspect the image",
            "steps": [{"specialist": "visual", "instruction": "Inspect pixels"}],
        }
    )

    plan = await PlannerAgent(fake_provider, max_steps=2).plan(image_task)

    assert plan.steps[0].specialist == SpecialistName.FILE
    assert "no native image-input capability" in plan.steps[0].instruction


@pytest.mark.asyncio
async def test_planner_exposes_only_the_current_provider_invocation_usage(
    fake_provider, task
) -> None:
    parsed = {
        "restatement": "Use lookup",
        "steps": [{"specialist": "lookup", "instruction": "Find a source"}],
    }
    fake_provider.return_parsed(parsed, usage=Usage(input_tokens=3, total_tokens=3))
    fake_provider.return_parsed(parsed, usage=Usage(output_tokens=5, total_tokens=5))
    planner = PlannerAgent(fake_provider)

    await planner.plan(task)
    first = planner.last_usage
    await planner.plan(task)
    second = planner.last_usage

    assert first == Usage(input_tokens=3, total_tokens=3)
    assert second == Usage(output_tokens=5, total_tokens=5)


@pytest.mark.asyncio
async def test_planner_usage_is_isolated_between_concurrent_calls(task) -> None:
    release_first = asyncio.Event()

    class InterleavingProvider:
        async def complete(
            self, request: ModelRequest, executor: object | None = None
        ) -> ModelResponse:
            context = json.loads(request.input_items[0]["content"][0]["text"])
            if context["task_id"] == "first":
                await release_first.wait()
                usage = Usage(input_tokens=3, total_tokens=3)
            else:
                release_first.set()
                usage = Usage(output_tokens=5, total_tokens=5)
            return ModelResponse(
                parsed={
                    "restatement": "Use lookup",
                    "steps": [{"specialist": "lookup", "instruction": "Find a source"}],
                },
                usage=usage,
            )

    planner = PlannerAgent(InterleavingProvider())

    async def invoke(task_id: str) -> Usage:
        await planner.plan(task.model_copy(update={"task_id": task_id}))
        if task_id == "second":
            await asyncio.sleep(0.01)
        return planner.last_usage

    first, second = await asyncio.gather(invoke("first"), invoke("second"))

    assert first == Usage(input_tokens=3, total_tokens=3)
    assert second == Usage(output_tokens=5, total_tokens=5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_type", "specialist", "tool_names", "kind"),
    [
        (LookupAgent, SpecialistName.LOOKUP, ["web_search", "read_webpage"], "source"),
        (ComputationAgent, SpecialistName.COMPUTATION, ["execute_python"], "computation"),
    ],
)
async def test_specialist_filters_mcp_specs_and_parses_typed_result(
    fake_provider,
    fake_tools,
    task,
    agent_type,
    specialist,
    tool_names,
    kind,
) -> None:
    source_url = "https://primary.example/fact" if kind == "source" else None
    fake_provider.return_parsed(
        {
            "specialist": specialist.value,
            "evidence": [
                {
                    "evidence_id": f"{kind}-1",
                    "kind": kind,
                    "summary": "Observable result",
                    "source_url": source_url,
                }
            ],
            "candidate_answer": "  Marie Curie  ",
        },
        usage=Usage(input_tokens=10, output_tokens=4, total_tokens=14),
    )
    step = PlanStep(specialist=specialist, instruction="Perform the assigned step")

    result = await agent_type(fake_provider).run(task, step, (), fake_tools)

    assert result.specialist == specialist
    assert result.candidate_answer == "Marie Curie"
    assert result.evidence[0].source_url == source_url
    assert result.usage.total_tokens == 14
    assert [tool.name for tool in fake_provider.requests[0].allowed_tools] == tool_names
    assert fake_provider.executors == [fake_tools]
    assert fake_tools.calls == []
    _assert_all_object_schemas_are_strict(fake_provider.requests[0].output_schema)


@pytest.mark.asyncio
async def test_specialist_repairs_one_malformed_response_without_tools(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_text("not valid specialist output")
    fake_provider.return_parsed(
        {
            "specialist": "lookup",
            "evidence": [
                {
                    "evidence_id": "source-1",
                    "kind": "source",
                    "summary": "Observable result",
                    "source_url": "https://primary.example/fact",
                }
            ],
            "candidate_answer": "Marie Curie",
        }
    )

    result = await LookupAgent(fake_provider).run(
        task,
        PlanStep(specialist="lookup", instruction="Find a source"),
        (),
        fake_tools,
    )

    assert result.candidate_answer == "Marie Curie"
    assert len(fake_provider.requests) == 2
    assert fake_provider.executors == [fake_tools, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("output_text, expected", [("FINAL ANSWER: 42", "42"), ("42", "42")])
async def test_specialist_extracts_a_concise_answer_from_non_json_text(
    fake_provider, fake_tools, task, output_text, expected
) -> None:
    fake_provider.return_text(output_text)
    step = PlanStep(specialist="computation", instruction="Calculate")

    result = await ComputationAgent(fake_provider).run(task, step, (), fake_tools)

    assert result.candidate_answer == expected
    assert result.evidence == ()


@pytest.mark.asyncio
async def test_specialist_defaults_missing_result_fields_while_schema_stays_strict(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed({"specialist": "lookup"})
    step = PlanStep(specialist="lookup", instruction="Find a source")

    result = await LookupAgent(fake_provider).run(task, step, (), fake_tools)

    schema = fake_provider.requests[0].output_schema
    assert set(schema["required"]) == set(schema["properties"])
    assert result.evidence == ()
    assert result.candidate_answer is None


@pytest.mark.asyncio
async def test_specialist_usage_survives_malformed_semantic_output(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed(
        {"specialist": "lookup", "evidence": "not-an-array", "candidate_answer": None},
        usage=Usage(input_tokens=9, output_tokens=4, total_tokens=13),
    )
    agent = LookupAgent(fake_provider)

    with pytest.raises(ValueError):
        await agent.run(
            task,
            PlanStep(specialist="lookup", instruction="Find a source"),
            (),
            fake_tools,
        )

    assert agent.last_usage == Usage(input_tokens=9, output_tokens=4, total_tokens=13)


@pytest.mark.asyncio
async def test_specialist_usage_is_task_local_without_stale_or_concurrent_leakage(
    task, fake_tools
) -> None:
    release_first = asyncio.Event()

    class InterleavingProvider:
        async def complete(
            self, request: ModelRequest, executor: object | None = None
        ) -> ModelResponse:
            context = json.loads(request.input_items[0]["content"][0]["text"])
            if context["task_id"] == "first":
                await release_first.wait()
                usage = Usage(input_tokens=3, total_tokens=3)
            else:
                release_first.set()
                usage = Usage(output_tokens=5, total_tokens=5)
            return ModelResponse(
                parsed={"specialist": "lookup", "evidence": [], "candidate_answer": "answer"},
                usage=usage,
            )

    agent = LookupAgent(InterleavingProvider())
    step = PlanStep(specialist="lookup", instruction="Find a source")

    async def invoke(task_id: str) -> Usage:
        await agent.run(task.model_copy(update={"task_id": task_id}), step, (), fake_tools)
        if task_id == "second":
            await asyncio.sleep(0.01)
        return agent.last_usage

    first, second = await asyncio.gather(invoke("first"), invoke("second"))

    assert first == Usage(input_tokens=3, total_tokens=3)
    assert second == Usage(output_tokens=5, total_tokens=5)


@pytest.mark.asyncio
async def test_specialist_usage_resets_before_a_provider_failure(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed(
        {"specialist": "lookup", "evidence": [], "candidate_answer": "answer"},
        usage=Usage(input_tokens=9, total_tokens=9),
    )
    agent = LookupAgent(fake_provider)
    step = PlanStep(specialist="lookup", instruction="Find a source")

    await agent.run(task, step, (), fake_tools)
    with pytest.raises(IndexError):
        await agent.run(task, step, (), fake_tools)

    assert agent.last_usage == Usage()


@pytest.mark.asyncio
async def test_critic_repairs_one_malformed_response_locally(fake_provider, fake_tools, task) -> None:
    fake_provider.return_text("not valid critic output")
    fake_provider.return_parsed(
        {
            "verdict": "approve",
            "feedback": "Evidence supports the candidate",
            "checked_evidence_ids": ["source-1"],
            "corrected_answer": None,
        }
    )

    decision = await CriticAgent(fake_provider).verify(
        task,
        "Marie Curie",
        (
            Evidence(
                evidence_id="source-1",
                kind="source",
                summary="Observable result",
                source_url="https://primary.example/fact",
            ),
        ),
        fake_tools,
    )

    assert decision.verdict == "approve"
    assert len(fake_provider.requests) == 2
    assert fake_provider.executors == [fake_tools, None]


@pytest.mark.asyncio
async def test_critic_uses_an_evidence_preserving_fallback_after_two_malformed_responses(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_text("not valid critic output")
    fake_provider.return_text("still not valid critic output")
    evidence = Evidence(
        evidence_id="source-1",
        kind="source",
        summary="Observable result",
        source_url="https://primary.example/fact",
    )

    decision = await CriticAgent(fake_provider).verify(
        task,
        "Marie Curie",
        (evidence,),
        fake_tools,
    )

    assert decision.verdict == "approve"
    assert decision.checked_evidence_ids == ("source-1",)


@pytest.mark.asyncio
async def test_file_agent_routes_registered_attachment_without_exposing_gold(
    fake_provider, fake_tools, image_task
) -> None:
    fake_provider.return_parsed(
        {
            "specialist": "file",
            "evidence": [
                {
                    "evidence_id": "file-1",
                    "kind": "file",
                    "summary": "PNG attachment metadata",
                    "source_url": None,
                }
            ],
            "candidate_answer": None,
        }
    )
    step = PlanStep(specialist="file", instruction="Inspect the attachment")

    await FileAnalysisAgent(fake_provider).run(image_task, step, (), fake_tools)
    request = fake_provider.requests[0]
    context = json.loads(request.input_items[0]["content"][0]["text"])

    assert [tool.name for tool in request.allowed_tools] == [
        "execute_python",
        "read_attachment",
        "transcribe_audio",
    ]
    assert context["attachment"]["path"] == str(image_task.attachment_path)
    assert context["attachment"]["media_type"] == "image/png"
    assert "Hidden Image Answer" not in repr(request.input_items)


@pytest.mark.asyncio
async def test_visual_agent_sends_high_detail_image_and_retains_read_tool(
    fake_provider, fake_tools, image_task
) -> None:
    fake_provider.return_parsed(
        {
            "specialist": "visual",
            "evidence": [
                {
                    "evidence_id": "visual-1",
                    "kind": "visual",
                    "summary": "A blue synthetic object is visible",
                    "source_url": None,
                }
            ],
            "candidate_answer": "blue object",
        }
    )
    step = PlanStep(specialist="visual", instruction="Inspect the image")

    result = await VisualAgent(fake_provider).run(image_task, step, (), fake_tools)
    request = fake_provider.requests[0]
    content = request.input_items[0]["content"]
    image = next(item for item in content if item["type"] == "input_image")

    assert result.evidence[0].kind == "visual"
    assert image["detail"] == "high"
    assert image["image_url"].startswith("data:image/png;base64,")
    assert [tool.name for tool in request.allowed_tools] == ["read_attachment"]


@pytest.mark.asyncio
async def test_specialist_context_serializes_prior_evidence_in_order(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed(
        {"specialist": "lookup", "evidence": [], "candidate_answer": None}
    )
    evidence = (
        Evidence(
            evidence_id="source-2",
            kind="source",
            summary="Second",
            source_url="https://two.example",
        ),
        Evidence(
            evidence_id="source-1",
            kind="source",
            summary="First",
            source_url="https://one.example",
        ),
    )
    step = PlanStep(specialist="lookup", instruction="Cross-check")

    await LookupAgent(fake_provider).run(task, step, evidence, fake_tools)
    context = json.loads(fake_provider.requests[0].input_items[0]["content"][0]["text"])

    assert [item["evidence_id"] for item in context["evidence"]] == ["source-2", "source-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parsed", "message"),
    [
        (
            {
                "specialist": "lookup",
                "evidence": [
                    {
                        "evidence_id": "",
                        "kind": "source",
                        "summary": "Missing identity",
                        "source_url": "https://example.test",
                    }
                ],
                "candidate_answer": None,
            },
            "evidence IDs must be non-empty",
        ),
        (
            {
                "specialist": "lookup",
                "evidence": [
                    {
                        "evidence_id": "source-1",
                        "kind": "source",
                        "summary": "One",
                        "source_url": "https://example.test/one",
                    },
                    {
                        "evidence_id": "source-1",
                        "kind": "source",
                        "summary": "Two",
                        "source_url": "https://example.test/two",
                    },
                ],
                "candidate_answer": None,
            },
            "evidence IDs must be unique",
        ),
        (
            {
                "specialist": "lookup",
                "evidence": [
                    {
                        "evidence_id": "source-1",
                        "kind": "source",
                        "summary": "No source URL",
                        "source_url": None,
                    }
                ],
                "candidate_answer": None,
            },
            "source evidence must include a URL",
        ),
    ],
)
async def test_specialist_rejects_untraceable_evidence(
    fake_provider, fake_tools, task, parsed, message
) -> None:
    fake_provider.return_parsed(parsed)
    step = PlanStep(specialist="lookup", instruction="Find a source")

    with pytest.raises(ValueError, match=message):
        await LookupAgent(fake_provider).run(task, step, (), fake_tools)


@pytest.mark.asyncio
async def test_specialist_rejects_evidence_id_already_present_in_prior_evidence(
    fake_provider, fake_tools, task
) -> None:
    prior = (
        Evidence(
            evidence_id="source-1",
            kind="source",
            summary="Prior source",
            source_url="https://prior.example/fact",
        ),
    )
    fake_provider.return_parsed(
        {
            "specialist": "lookup",
            "evidence": [
                {
                    "evidence_id": "source-1",
                    "kind": "source",
                    "summary": "New source reuses the prior ID",
                    "source_url": "https://new.example/fact",
                }
            ],
            "candidate_answer": None,
        }
    )
    step = PlanStep(specialist="lookup", instruction="Find another source")

    with pytest.raises(
        ValueError,
        match=r"evidence IDs duplicate prior evidence: \['source-1'\]",
    ):
        await LookupAgent(fake_provider).run(task, step, prior, fake_tools)


@pytest.mark.asyncio
async def test_specialist_rejects_a_result_for_the_wrong_role(fake_provider, fake_tools, task) -> None:
    fake_provider.return_parsed(
        {"specialist": "computation", "evidence": [], "candidate_answer": "42"}
    )
    step = PlanStep(specialist="lookup", instruction="Find a source")

    with pytest.raises((ValueError, ValidationError), match="specialist"):
        await LookupAgent(fake_provider).run(task, step, (), fake_tools)


@pytest.mark.asyncio
async def test_critic_uses_only_verification_tools_and_returns_typed_decision(
    fake_provider, fake_tools, task
) -> None:
    evidence = (
        Evidence(
            evidence_id="source-1",
            kind="source",
            summary="Primary source supports the candidate",
            source_url="https://primary.example/fact",
        ),
    )
    fake_provider.return_parsed(
        {
            "verdict": "approve",
            "feedback": "The cited source supports the candidate.",
            "checked_evidence_ids": ["source-1"],
            "corrected_answer": None,
        }
    )

    decision = await CriticAgent(fake_provider).verify(
        task, "Marie Curie", evidence, fake_tools
    )
    request = fake_provider.requests[0]

    assert decision.verdict == "approve"
    assert decision.checked_evidence_ids == ("source-1",)
    assert [tool.name for tool in request.allowed_tools] == [
        "web_search",
        "read_webpage",
        "execute_python",
        "transcribe_audio",
    ]
    assert "transcribe_audio" in request.instructions
    assert "non-audio" in request.instructions.lower()
    assert fake_provider.executors == [fake_tools]
    assert fake_tools.calls == []
    _assert_all_object_schemas_are_strict(request.output_schema)
    assert request.output_schema["properties"]["verdict"]["enum"] == [
        "approve",
        "revise",
        "insufficient_evidence",
    ]
    assert "Hidden Reference Answer" not in repr(request.input_items)


@pytest.mark.asyncio
async def test_critic_exposes_only_the_current_provider_invocation_usage(
    fake_provider, fake_tools, task
) -> None:
    evidence = (
        Evidence(
            evidence_id="source-1",
            kind="source",
            summary="Primary source supports the candidate",
            source_url="https://primary.example/fact",
        ),
    )
    parsed = {
        "verdict": "approve",
        "feedback": "The source supports the candidate",
        "checked_evidence_ids": ["source-1"],
        "corrected_answer": None,
    }
    fake_provider.return_parsed(parsed, usage=Usage(input_tokens=7, total_tokens=7))
    fake_provider.return_parsed(parsed, usage=Usage(reasoning_tokens=11, total_tokens=11))
    critic = CriticAgent(fake_provider)

    await critic.verify(task, "Marie Curie", evidence, fake_tools)
    first = critic.last_usage
    await critic.verify(task, "Marie Curie", evidence, fake_tools)
    second = critic.last_usage

    assert first == Usage(input_tokens=7, total_tokens=7)
    assert second == Usage(reasoning_tokens=11, total_tokens=11)


@pytest.mark.asyncio
async def test_critic_approval_must_name_existing_checked_evidence(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed(
        {
            "verdict": "approve",
            "feedback": "Unsupported approval",
            "checked_evidence_ids": [],
            "corrected_answer": None,
        }
    )

    with pytest.raises(ValueError, match="approved critic decision must name checked evidence"):
        await CriticAgent(fake_provider).verify(task, "Marie Curie", (), fake_tools)


@pytest.mark.asyncio
async def test_critic_rejects_unknown_checked_evidence_id(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed(
        {
            "verdict": "approve",
            "feedback": "Claims to have checked missing evidence",
            "checked_evidence_ids": ["source-missing"],
            "corrected_answer": None,
        }
    )

    with pytest.raises(ValueError, match="unknown evidence IDs"):
        await CriticAgent(fake_provider).verify(task, "Marie Curie", (), fake_tools)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "corrected_answer", "message"),
    [
        ("revise", None, "revised critic decision must propose one correction"),
        ("approve", "different answer", "only a revised critic decision may correct"),
        (
            "insufficient_evidence",
            "free answer",
            "only a revised critic decision may correct",
        ),
    ],
)
async def test_critic_correction_is_single_and_verdict_scoped(
    fake_provider, fake_tools, task, verdict, corrected_answer, message
) -> None:
    fake_provider.return_parsed(
        {
            "verdict": verdict,
            "feedback": "Check the evidence",
            "checked_evidence_ids": [],
            "corrected_answer": corrected_answer,
        }
    )

    with pytest.raises(ValueError, match=message):
        await CriticAgent(fake_provider).verify(task, "candidate", (), fake_tools)


@pytest.mark.asyncio
async def test_critic_rejects_ungrounded_revision_without_evidence_or_tool_event(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed(
        {
            "verdict": "revise",
            "feedback": "Claims a correction without checking anything",
            "checked_evidence_ids": [],
            "corrected_answer": "42",
        }
    )

    with pytest.raises(
        ValueError,
        match="revised critic decision must be grounded in checked evidence or an allowed tool event",
    ):
        await CriticAgent(fake_provider).verify(task, "41", (), fake_tools)


@pytest.mark.asyncio
async def test_critic_accepts_revision_grounded_in_known_evidence(
    fake_provider, fake_tools, task
) -> None:
    evidence = (
        Evidence(
            evidence_id="computation-1",
            kind="computation",
            summary="The checked calculation returns 42",
        ),
    )
    fake_provider.return_parsed(
        {
            "verdict": "revise",
            "feedback": "The checked calculation contradicts the candidate",
            "checked_evidence_ids": ["computation-1"],
            "corrected_answer": "42",
        }
    )

    decision = await CriticAgent(fake_provider).verify(task, "41", evidence, fake_tools)

    assert decision.corrected_answer == "42"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "output"),
    [
        (
            "web_search",
            {"query": "independent check"},
            {"answer": "42", "sources": []},
        ),
        (
            "web_search",
            {"query": "independent check"},
            {
                "answer": "",
                "sources": [{"url": "https://source.example", "title": "Primary source"}],
            },
        ),
        (
            "read_webpage",
            {"url": "https://source.example"},
            {
                "url": "https://source.example",
                "status_code": 200,
                "content_type": "text/html",
                "text": "The independently verified result is 42.",
                "truncated": False,
            },
        ),
        (
            "execute_python",
            {"instruction": "Independently calculate the answer"},
            {"output_text": "42", "outputs": []},
        ),
        (
            "execute_python",
            {"instruction": "Independently calculate the answer"},
            {"output_text": "", "outputs": [{"type": "logs", "logs": "42"}]},
        ),
        (
            "transcribe_audio",
            {"path": "sample.mp3"},
            {"text": "The spoken answer is 42."},
        ),
    ],
)
async def test_critic_accepts_revision_grounded_in_tool_specific_observable_result(
    fake_provider, fake_tools, task, tool_name, arguments, output
) -> None:
    tool_event = ToolEvent(
        call_id="call-1",
        name=tool_name,
        arguments=arguments,
        output=output,
    )
    fake_provider.return_parsed(
        {
            "verdict": "revise",
            "feedback": "The independent calculation contradicts the candidate",
            "checked_evidence_ids": [],
            "corrected_answer": "42",
        },
        tool_events=(tool_event,),
    )

    decision = await CriticAgent(fake_provider).verify(task, "41", (), fake_tools)

    assert decision.corrected_answer == "42"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "output"),
    [
        ("execute_python", {}),
        (
            "execute_python",
            {"output_text": "  ", "outputs": [], "meta": {"items": []}},
        ),
        (
            "execute_python",
            {
                "error": "calculation failed",
                "meta": {"api_key": "redacted-secret"},
            },
        ),
        (
            "read_webpage",
            {
                "url": "https://source.example/empty",
                "status_code": 200,
                "content_type": "text/html; charset=utf-8",
                "text": "  ",
                "truncated": False,
            },
        ),
        (
            "execute_python",
            {"status": "error", "message": "calculation failed"},
        ),
        (
            "execute_python",
            {"request_id": "req-1", "latency_ms": 5},
        ),
        ("submit_answer", {"answer": "42"}),
    ],
)
async def test_critic_rejects_revision_grounded_only_by_non_evidence_tool_event(
    fake_provider, fake_tools, task, tool_name, output
) -> None:
    fake_provider.return_parsed(
        {
            "verdict": "revise",
            "feedback": "The event does not provide allowed observable result content",
            "checked_evidence_ids": [],
            "corrected_answer": "42",
        },
        tool_events=(
            ToolEvent(
                call_id="call-empty",
                name=tool_name,
                arguments={},
                output=output,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="revised critic decision must be grounded in checked evidence or an allowed tool event",
    ):
        await CriticAgent(fake_provider).verify(task, "41", (), fake_tools)


@pytest.mark.asyncio
async def test_critic_rejects_an_unknown_verdict(fake_provider, fake_tools, task) -> None:
    fake_provider.return_parsed(
        {
            "verdict": "maybe",
            "feedback": "Unstructured indecision",
            "checked_evidence_ids": [],
            "corrected_answer": None,
        }
    )

    with pytest.raises(ValidationError, match="verdict"):
        await CriticAgent(fake_provider).verify(task, "candidate", (), fake_tools)


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected_answer", ["FINAL ANSWER: 42", "42\nExplanation"])
async def test_critic_rejects_a_non_concise_correction(
    fake_provider, fake_tools, task, corrected_answer
) -> None:
    fake_provider.return_parsed(
        {
            "verdict": "revise",
            "feedback": "A correction is needed",
            "checked_evidence_ids": [],
            "corrected_answer": corrected_answer,
        }
    )

    with pytest.raises(ValueError, match="concise GAIA format"):
        await CriticAgent(fake_provider).verify(task, "candidate", (), fake_tools)


@pytest.mark.asyncio
async def test_critic_cannot_approve_a_missing_candidate(
    fake_provider, fake_tools, task
) -> None:
    evidence = (
        Evidence(
            evidence_id="source-1",
            kind="source",
            summary="A checked source",
            source_url="https://example.test",
        ),
    )
    fake_provider.return_parsed(
        {
            "verdict": "approve",
            "feedback": "No candidate exists",
            "checked_evidence_ids": ["source-1"],
            "corrected_answer": None,
        }
    )

    with pytest.raises(ValueError, match="cannot approve a missing candidate"):
        await CriticAgent(fake_provider).verify(task, None, evidence, fake_tools)


def test_role_prompts_enforce_grounded_observable_and_concise_behavior() -> None:
    planner_prompt = PlannerAgent.instructions.lower()
    assert "do not answer" in planner_prompt
    assert "observable" in planner_prompt
    for agent_type in (LookupAgent, ComputationAgent, FileAnalysisAgent, VisualAgent):
        prompt = agent_type.instructions.lower()
        assert "observable" in prompt
        assert "hidden reasoning" in prompt
        assert "invent" in prompt
        assert "concise" in prompt or "few words" in prompt
    critic_prompt = CriticAgent.instructions.lower()
    assert "insufficient or contradictory" in critic_prompt
    assert "exactly one concise correction" in critic_prompt
    assert "do not solve the task freely" in critic_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate", ["", "FINAL ANSWER: 42", "42\nExplanation"])
async def test_specialist_rejects_non_concise_candidate_format(
    fake_provider, fake_tools, task, candidate
) -> None:
    fake_provider.return_parsed(
        {"specialist": "computation", "evidence": [], "candidate_answer": candidate}
    )
    step = PlanStep(specialist="computation", instruction="Calculate")

    with pytest.raises(ValueError, match="concise GAIA format"):
        await ComputationAgent(fake_provider).run(task, step, (), fake_tools)


@pytest.mark.asyncio
async def test_visual_agent_rejects_non_image_attachment(
    fake_provider, fake_tools, task, tmp_path
) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not an image", encoding="utf-8")
    text_task = task.model_copy(update={"attachment_path": text_path})
    step = PlanStep(specialist="visual", instruction="Inspect")

    with pytest.raises(ValueError, match="requires an image attachment"):
        await VisualAgent(fake_provider).run(text_task, step, (), fake_tools)
    assert fake_provider.requests == []


@pytest.mark.asyncio
async def test_specialist_schema_is_role_specific_and_does_not_ask_model_for_usage(
    fake_provider, fake_tools, task
) -> None:
    fake_provider.return_parsed(
        {"specialist": "computation", "evidence": [], "candidate_answer": "42"}
    )
    step = PlanStep(specialist="computation", instruction="Calculate")

    await ComputationAgent(fake_provider).run(task, step, (), fake_tools)
    schema = fake_provider.requests[0].output_schema

    assert schema["properties"]["specialist"] == {
        "type": "string",
        "const": "computation",
    }
    assert schema["$defs"]["Evidence"]["properties"]["kind"]["enum"] == [
        "computation"
    ]
    assert "usage" not in schema["properties"]


def _assert_all_object_schemas_are_strict(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
            assert set(value.get("required", ())) == set(value.get("properties", ()))
        for child in value.values():
            _assert_all_object_schemas_are_strict(child)
    elif isinstance(value, list):
        for child in value:
            _assert_all_object_schemas_are_strict(child)
