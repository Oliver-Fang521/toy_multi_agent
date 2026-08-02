from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from gaia_agent.models import (
    AgentPlan,
    AgentResult,
    CriticDecision,
    Evidence,
    GaiaTask,
    PlanStep,
    SpecialistName,
)
from gaia_agent.providers import ModelProvider, ModelRequest

if TYPE_CHECKING:
    from gaia_agent.mcp_client import MCPToolClient

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    _close_object_schemas(schema)
    return schema


def _close_object_schemas(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object" and isinstance(value.get("properties"), dict):
            value["additionalProperties"] = False
            value["required"] = list(value["properties"])
        for child in value.values():
            _close_object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            _close_object_schemas(child)


class PlannerAgent:
    instructions = (
        "Restate the task and return only a short ordered plan using the available "
        "specialist roles. Plan how to gather observable evidence. Do not answer the task or "
        "request hidden reasoning."
    )

    def __init__(
        self,
        provider: ModelProvider,
        *,
        max_steps: int = 8,
        reasoning_effort: ReasoningEffort = "medium",
    ) -> None:
        self.provider = provider
        self.max_steps = max_steps
        self.reasoning_effort = reasoning_effort

    async def plan(self, task: GaiaTask) -> AgentPlan:
        output_schema = _strict_schema(AgentPlan)
        output_schema["properties"]["steps"].update(
            {"minItems": 1, "maxItems": self.max_steps}
        )
        request = ModelRequest(
            instructions=self.instructions,
            input_items=(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "task_id": task.task_id,
                                    "question": task.question,
                                    "level": task.level,
                                    "attachment": task.attachment_path.name
                                    if task.attachment_path
                                    else None,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                },
            ),
            output_schema=output_schema,
            allowed_tools=(),
            max_tool_calls=0,
            reasoning_effort=self.reasoning_effort,
        )
        response = await self.provider.complete(request)
        plan = AgentPlan.model_validate(response.parsed)
        if not 1 <= len(plan.steps) <= self.max_steps:
            raise ValueError(f"planner must return between 1 and {self.max_steps} steps")
        return plan


class SpecialistAgent:
    name: SpecialistName
    allowed_tool_names: frozenset[str]
    allowed_evidence_kinds: frozenset[str]
    instructions: str

    def __init__(
        self,
        provider: ModelProvider,
        *,
        reasoning_effort: ReasoningEffort = "medium",
        max_tool_calls: int = 8,
    ) -> None:
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.max_tool_calls = max_tool_calls

    async def run(
        self,
        task: GaiaTask,
        step: PlanStep,
        evidence: tuple[Evidence, ...],
        tools: MCPToolClient,
    ) -> AgentResult:
        if step.specialist != self.name:
            raise ValueError(
                f"step specialist {step.specialist.value} does not match agent {self.name.value}"
            )
        allowed_tools = tuple(
            tool for tool in tools.tool_specs if tool.name in self.allowed_tool_names
        )
        available_names = {tool.name for tool in allowed_tools}
        if missing := self.allowed_tool_names - available_names:
            raise ValueError(f"MCP client is missing required tools: {sorted(missing)}")
        output_schema = _agent_result_schema(self.name, self.allowed_evidence_kinds)
        request = ModelRequest(
            instructions=self.instructions,
            input_items=_specialist_input(task, step, evidence, visual=self.name == "visual"),
            output_schema=output_schema,
            allowed_tools=allowed_tools,
            max_tool_calls=self.max_tool_calls,
            reasoning_effort=self.reasoning_effort,
        )
        response = await self.provider.complete(request, tools)
        result = AgentResult.model_validate(response.parsed)
        if result.specialist != self.name:
            raise ValueError(
                f"result specialist {result.specialist.value} does not match agent {self.name.value}"
            )
        _validate_evidence(result.evidence, self.allowed_evidence_kinds)
        prior_ids = {item.evidence_id for item in evidence}
        duplicate_prior_ids = sorted(
            item.evidence_id for item in result.evidence if item.evidence_id in prior_ids
        )
        if duplicate_prior_ids:
            raise ValueError(
                f"evidence IDs duplicate prior evidence: {duplicate_prior_ids}"
            )
        candidate = _normalize_candidate_answer(result.candidate_answer)
        return result.model_copy(update={"candidate_answer": candidate, "usage": response.usage})


class LookupAgent(SpecialistAgent):
    name = SpecialistName.LOOKUP
    allowed_tool_names = frozenset({"web_search", "read_webpage"})
    allowed_evidence_kinds = frozenset({"source"})
    instructions = (
        "Use only the supplied web tools to find observable evidence, prioritizing primary "
        "sources. Return source URLs and concise summaries; do not invent unavailable facts or "
        "request hidden reasoning. A candidate answer must follow the GAIA format: a number, as "
        "few words as possible, or an ordered comma-separated list without prose."
    )


class ComputationAgent(SpecialistAgent):
    name = SpecialistName.COMPUTATION
    allowed_tool_names = frozenset({"execute_python"})
    allowed_evidence_kinds = frozenset({"computation"})
    instructions = (
        "Use only execute_python through the supplied tool plane for calculations and data "
        "transformation. Return observable computation evidence; do not invent results or request "
        "hidden reasoning. A candidate answer must be a number, as few words as possible, or an "
        "ordered comma-separated list without explanatory prose."
    )


class FileAnalysisAgent(SpecialistAgent):
    name = SpecialistName.FILE
    allowed_tool_names = frozenset({"read_attachment", "execute_python"})
    allowed_evidence_kinds = frozenset({"file", "computation"})
    instructions = (
        "Inspect only the registered task attachment through read_attachment or execute_python. "
        "Return observable file or computation evidence; do not invent missing content or request "
        "hidden reasoning. Keep any candidate answer in the concise GAIA answer format."
    )


class VisualAgent(SpecialistAgent):
    name = SpecialistName.VISUAL
    allowed_tool_names = frozenset({"read_attachment"})
    allowed_evidence_kinds = frozenset({"visual"})
    instructions = (
        "Inspect the supplied task image at high detail and use only read_attachment for its "
        "registered metadata. Report observable visual evidence without inventing obscured content "
        "or requesting hidden reasoning. Keep any candidate answer in the concise GAIA format."
    )


FileAgent = FileAnalysisAgent


class CriticAgent:
    allowed_tool_names = frozenset({"web_search", "read_webpage", "execute_python"})
    instructions = (
        "Independently verify the candidate against the observable evidence. Check sufficiency and "
        "contradictions, reject unsupported research answers, and use the supplied web or Python "
        "tools only when the existing evidence is insufficient or contradictory. Never request "
        "hidden reasoning and never invent facts. Approve only evidence-backed candidates. If a "
        "revision is necessary, propose exactly one concise correction directly supported by the "
        "checked evidence; do not solve the task freely."
    )

    def __init__(
        self,
        provider: ModelProvider,
        *,
        reasoning_effort: ReasoningEffort = "medium",
        max_tool_calls: int = 8,
    ) -> None:
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.max_tool_calls = max_tool_calls

    async def verify(
        self,
        task: GaiaTask,
        candidate: str | None,
        evidence: tuple[Evidence, ...],
        tools: MCPToolClient,
    ) -> CriticDecision:
        allowed_tools = tuple(
            tool for tool in tools.tool_specs if tool.name in self.allowed_tool_names
        )
        available_names = {tool.name for tool in allowed_tools}
        if missing := self.allowed_tool_names - available_names:
            raise ValueError(f"MCP client is missing required tools: {sorted(missing)}")
        context = {
            "task_id": task.task_id,
            "question": task.question,
            "level": task.level,
            "candidate_answer": candidate,
            "attachment": _attachment_metadata(task),
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
        output_schema = _strict_schema(CriticDecision)
        output_schema["properties"]["verdict"] = {
            "type": "string",
            "enum": ["approve", "revise", "insufficient_evidence"],
        }
        request = ModelRequest(
            instructions=self.instructions,
            input_items=(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(context, sort_keys=True, separators=(",", ":")),
                        }
                    ],
                },
            ),
            output_schema=output_schema,
            allowed_tools=allowed_tools,
            max_tool_calls=self.max_tool_calls,
            reasoning_effort=self.reasoning_effort,
        )
        response = await self.provider.complete(request, tools)
        decision = CriticDecision.model_validate(response.parsed)
        corrected_answer = _normalize_candidate_answer(decision.corrected_answer)
        if decision.verdict == "revise" and not corrected_answer:
            raise ValueError("revised critic decision must propose one correction")
        if decision.verdict != "revise" and corrected_answer is not None:
            raise ValueError("only a revised critic decision may correct the answer")
        if decision.verdict == "approve" and not decision.checked_evidence_ids:
            raise ValueError("approved critic decision must name checked evidence")
        if decision.verdict == "approve" and candidate is None:
            raise ValueError("critic cannot approve a missing candidate")
        known_ids = {item.evidence_id for item in evidence}
        unknown_ids = set(decision.checked_evidence_ids) - known_ids
        if unknown_ids:
            raise ValueError(f"critic checked unknown evidence IDs: {sorted(unknown_ids)}")
        if len(set(decision.checked_evidence_ids)) != len(decision.checked_evidence_ids):
            raise ValueError("critic checked evidence IDs must be unique")
        has_allowed_tool_evidence = any(
            event.name in self.allowed_tool_names and bool(event.output)
            for event in response.tool_events
        )
        if (
            decision.verdict == "revise"
            and not decision.checked_evidence_ids
            and not has_allowed_tool_evidence
        ):
            raise ValueError(
                "revised critic decision must be grounded in checked evidence "
                "or an allowed tool event"
            )
        return decision.model_copy(update={"corrected_answer": corrected_answer})


def _agent_result_schema(
    specialist: SpecialistName,
    allowed_evidence_kinds: frozenset[str],
) -> dict[str, Any]:
    schema = AgentResult.model_json_schema()
    schema["properties"].pop("usage")
    schema["properties"]["specialist"] = {"type": "string", "const": specialist.value}
    evidence_schema = schema.get("$defs", {}).get("Evidence")
    if isinstance(evidence_schema, dict):
        evidence_schema["properties"]["kind"] = {
            "type": "string",
            "enum": sorted(allowed_evidence_kinds),
        }
    _close_object_schemas(schema)
    return schema


def _specialist_input(
    task: GaiaTask,
    step: PlanStep,
    evidence: tuple[Evidence, ...],
    *,
    visual: bool,
) -> tuple[dict[str, Any], ...]:
    attachment = _attachment_metadata(task)
    context = {
        "task_id": task.task_id,
        "question": task.question,
        "level": task.level,
        "instruction": step.instruction,
        "attachment": attachment,
        "evidence": [item.model_dump(mode="json") for item in evidence],
    }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": json.dumps(context, sort_keys=True, separators=(",", ":")),
        }
    ]
    if visual:
        if task.attachment_path is None:
            raise ValueError("visual agent requires a task attachment")
        media_type = attachment["media_type"] if attachment is not None else "image/png"
        if not str(media_type).startswith("image/"):
            raise ValueError("visual agent requires an image attachment")
        encoded = b64encode(task.attachment_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{encoded}",
                "detail": "high",
            }
        )
    return ({"role": "user", "content": content},)


def _attachment_metadata(task: GaiaTask) -> dict[str, str] | None:
    if task.attachment_path is None:
        return None
    media_type = mimetypes.guess_type(task.attachment_path.name)[0]
    return {
        "path": str(task.attachment_path),
        "name": task.attachment_path.name,
        "media_type": media_type or "application/octet-stream",
    }


def _validate_evidence(
    evidence: tuple[Evidence, ...],
    allowed_evidence_kinds: frozenset[str],
) -> None:
    evidence_ids = [item.evidence_id for item in evidence]
    if any(not evidence_id.strip() for evidence_id in evidence_ids):
        raise ValueError("evidence IDs must be non-empty")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence IDs must be unique")
    for item in evidence:
        if item.kind not in allowed_evidence_kinds:
            raise ValueError(f"evidence kind {item.kind} is not allowed for this specialist")
        if item.kind == "source" and not item.source_url:
            raise ValueError("source evidence must include a URL")


def _normalize_candidate_answer(candidate: str | None) -> str | None:
    if candidate is None:
        return None
    normalized = candidate.strip()
    if (
        not normalized
        or "\n" in normalized
        or normalized.casefold().startswith("final answer:")
    ):
        raise ValueError("candidate answer must use the concise GAIA format")
    return normalized
