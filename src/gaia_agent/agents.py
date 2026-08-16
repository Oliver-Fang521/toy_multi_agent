from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from gaia_agent.mcp_client import AttachmentPolicyError
from gaia_agent.models import (
    AgentPlan,
    AgentResult,
    CriticDecision,
    Evidence,
    GaiaTask,
    PlanStep,
    SpecialistName,
    Usage,
)
from gaia_agent.providers import ModelProvider, ModelRequest
from gaia_agent.research_notebook import NotebookEntry, select_evidence_cards

if TYPE_CHECKING:
    from gaia_agent.mcp_client import MCPToolClient

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]

_MAX_TOOL_RESULT_ITEMS = 32
_MAX_TOOL_RESULT_DEPTH = 3
_WEB_RESULT_FIELDS = ("url", "title", "snippet", "text", "content", "answer")
_PYTHON_RESULT_FIELDS = (
    "logs",
    "text",
    "content",
    "data",
    "result",
    "value",
    "output",
    "image_url",
    "url",
)
_DEFAULT_MAX_OUTPUT_TOKENS = 2_000
_MAX_REASONING_OUTPUT_TOKENS = 8_000
_DEFAULT_MAX_CONTEXT_BYTES = 24_576
_MAX_CONTEXT_QUESTION_CHARS = 4_096
_MAX_CONTEXT_INSTRUCTION_CHARS = 2_048
_EVIDENCE_CARD_TIMESTAMP = datetime(1970, 1, 1, tzinfo=UTC)
_REPAIR_INSTRUCTIONS = (
    "Your previous response did not match the required JSON schema. "
    "Return only the JSON object exactly matching the schema, with no prose, "
    "no commentary, and no markdown code fences."
)
_NON_ANSWER_PREFIXES = (
    "i cannot",
    "i can't",
    "i do not know",
    "i don't know",
    "i'm unable",
    "unable to",
    "as an ai",
    "the answer is",
    "here is the answer",
    "not valid",
    "error:",
)

_Parsed = TypeVar("_Parsed")


class StageValidationCode(StrEnum):
    ROLE_MISMATCH = "role_mismatch"
    EVIDENCE_SHAPE = "evidence_shape"
    SOURCE_URL_MISSING = "source_url_missing"
    CANDIDATE_FORMAT = "candidate_format"
    CHECKED_EVIDENCE = "checked_evidence"
    SCHEMA = "schema"


class StageProtocolError(ValueError):
    def __init__(
        self,
        code: StageValidationCode,
        usage: Usage,
        repairable: bool = True,
        *,
        message: str | None = None,
        repair_attempted: bool = False,
    ) -> None:
        super().__init__(message or code.value)
        self.code = code
        self.usage = usage
        self.repairable = repairable
        self.repair_attempted = repair_attempted


def _max_output_tokens(reasoning_effort: ReasoningEffort) -> int:
    return (
        _MAX_REASONING_OUTPUT_TOKENS
        if reasoning_effort == "max"
        else _DEFAULT_MAX_OUTPUT_TOKENS
    )


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _repair_request(request: ModelRequest) -> ModelRequest:
    return request.model_copy(
        update={
            "instructions": f"{request.instructions}\n\n{_REPAIR_INSTRUCTIONS}",
            "allowed_tools": (),
            "max_tool_calls": 0,
            "max_output_tokens": min(request.max_output_tokens, _DEFAULT_MAX_OUTPUT_TOKENS),
        }
    )


def _semantic_repair_request(
    request: ModelRequest,
    error: StageProtocolError,
) -> ModelRequest:
    return request.model_copy(
        update={
            "instructions": (
                f"{request.instructions}\n\nRepair code: {error.code.value}. "
                "Return one valid object only."
            ),
            "allowed_tools": (),
            "max_tool_calls": 0,
        }
    )


def _parsed_repair_input(parsed: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return only provider-parsed JSON, never output text or tool event bodies."""
    return (
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": json.dumps(parsed, sort_keys=True, separators=(",", ":")),
                }
            ],
        },
    )


def _with_usage(
    error: StageProtocolError,
    usage: Usage,
    *,
    repair_attempted: bool | None = None,
) -> StageProtocolError:
    return StageProtocolError(
        error.code,
        usage,
        error.repairable,
        message=str(error),
        repair_attempted=(
            error.repair_attempted
            if repair_attempted is None
            else repair_attempted
        ),
    )


async def _complete_with_local_repair(
    provider: ModelProvider,
    request: ModelRequest,
    parser: Callable[[Any], _Parsed],
    *,
    executor: MCPToolClient | None = None,
) -> tuple[_Parsed, Usage, bool]:
    """Complete one stage, repairing one malformed structured response in place.

    The repair call is deliberately tool-free. A malformed response is a protocol
    failure, not evidence that the stage needs a second research trajectory.
    """
    response = await provider.complete(request, executor)
    try:
        return parser(response), response.usage, False
    except AttachmentPolicyError:
        raise
    except StageProtocolError as error:
        protocol_error = _with_usage(error, response.usage, repair_attempted=False)
    except (ValidationError, ValueError, TypeError, KeyError) as error:
        protocol_error = StageProtocolError(StageValidationCode.SCHEMA, response.usage)
        protocol_error.__cause__ = error

    if not protocol_error.repairable:
        raise protocol_error

    repair_request = (
        _semantic_repair_request(request, protocol_error).model_copy(
            update={"input_items": _parsed_repair_input(response.parsed)}
        )
        if response.parsed is not None
        else _repair_request(request)
    )
    repaired = await provider.complete(repair_request)
    aggregate_usage = _add_usage(response.usage, repaired.usage)
    try:
        parsed = parser(repaired)
    except AttachmentPolicyError:
        raise
    except StageProtocolError as repair_error:
        if not repair_error.repairable:
            raise _with_usage(repair_error, aggregate_usage, repair_attempted=True)
        raise StageProtocolError(
            repair_error.code,
            aggregate_usage,
            message=str(repair_error),
            repair_attempted=True,
        ) from repair_error
    except (ValidationError, ValueError, TypeError, KeyError) as repair_error:
        raise StageProtocolError(
            StageValidationCode.SCHEMA,
            aggregate_usage,
            message=str(repair_error),
            repair_attempted=True,
        ) from repair_error
    return parsed, aggregate_usage, True


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
        self._last_usage: ContextVar[Usage | None] = ContextVar(
            f"planner_usage_{id(self)}", default=None
        )
        self._last_repair: ContextVar[bool] = ContextVar(
            f"planner_repair_{id(self)}", default=False
        )
        self._last_fallback: ContextVar[bool] = ContextVar(
            f"planner_fallback_{id(self)}", default=False
        )

    @property
    def last_usage(self) -> Usage:
        return self._last_usage.get() or Usage()

    @property
    def last_repair_attempted(self) -> bool:
        return self._last_repair.get()

    @property
    def last_fallback_used(self) -> bool:
        return self._last_fallback.get()

    async def plan(self, task: GaiaTask) -> AgentPlan:
        self._last_usage.set(Usage())
        self._last_repair.set(False)
        self._last_fallback.set(False)
        attachment = _attachment_metadata(task)
        capabilities = {
            "attachment_present": task.attachment_path is not None,
            "attachment_media_type": attachment["media_type"] if attachment else None,
            "allowed_specialists": [
                item.value for item in _allowed_specialists(task, self.provider)
            ],
        }
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
                                    "capabilities": capabilities,
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
            max_output_tokens=_max_output_tokens(self.reasoning_effort),
            max_tool_calls=0,
            reasoning_effort=self.reasoning_effort,
        )
        def parse(response: Any) -> AgentPlan:
            try:
                plan = AgentPlan.model_validate(response.parsed)
            except ValidationError as error:
                raise StageProtocolError(StageValidationCode.SCHEMA, Usage()) from error
            if not 1 <= len(plan.steps) <= self.max_steps:
                raise StageProtocolError(
                    StageValidationCode.SCHEMA,
                    Usage(),
                    message=f"planner must return between 1 and {self.max_steps} steps",
                )
            return normalize_plan(plan, task, self.provider)

        try:
            plan, usage, repaired = await _complete_with_local_repair(
                self.provider,
                request,
                parse,
            )
        except StageProtocolError as error:
            self._last_usage.set(error.usage)
            if not error.repairable:
                raise
            # Planning is a routing concern. If the model cannot satisfy the
            # protocol twice, continue with a conservative, answer-free route.
            self._last_repair.set(True)
            self._last_fallback.set(True)
            return normalize_plan(_fallback_plan(task, self.provider), task, self.provider)
        self._last_repair.set(repaired)
        self._last_usage.set(usage)
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
        max_context_bytes: int = _DEFAULT_MAX_CONTEXT_BYTES,
    ) -> None:
        if max_context_bytes < 1:
            raise ValueError("max_context_bytes must be positive")
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.max_tool_calls = max_tool_calls
        self.max_context_bytes = max_context_bytes
        self._last_usage: ContextVar[Usage | None] = ContextVar(
            f"specialist_usage_{id(self)}", default=None
        )
        self._last_repair: ContextVar[bool] = ContextVar(
            f"specialist_repair_{id(self)}", default=False
        )

    @property
    def last_usage(self) -> Usage:
        return self._last_usage.get() or Usage()

    @property
    def last_repair_attempted(self) -> bool:
        return self._last_repair.get()

    async def run(
        self,
        task: GaiaTask,
        step: PlanStep,
        evidence: tuple[Evidence, ...],
        tools: MCPToolClient,
    ) -> AgentResult:
        self._last_usage.set(Usage())
        self._last_repair.set(False)
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
            input_items=await _specialist_input(
                task,
                step,
                evidence,
                visual=self.name == "visual",
                tools=tools,
                max_json_bytes=self.max_context_bytes,
            ),
            output_schema=output_schema,
            allowed_tools=allowed_tools,
            max_output_tokens=_max_output_tokens(self.reasoning_effort),
            max_tool_calls=self.max_tool_calls,
            reasoning_effort=self.reasoning_effort,
        )
        def parse(response: Any) -> AgentResult:
            if response.parsed is None:
                fallback = _extract_final_answer(response.output_text)
                if fallback is None:
                    raise StageProtocolError(
                        StageValidationCode.SCHEMA,
                        Usage(),
                        message="specialist response did not contain a structured or final answer",
                    )
                result = AgentResult(specialist=self.name, candidate_answer=fallback)
            else:
                try:
                    result = AgentResult.model_validate(response.parsed)
                except ValidationError as error:
                    raise StageProtocolError(StageValidationCode.SCHEMA, Usage()) from error
            if result.specialist != self.name:
                raise StageProtocolError(
                    StageValidationCode.ROLE_MISMATCH,
                    Usage(),
                    message=(
                        f"result specialist {result.specialist.value} does not match "
                        f"agent {self.name.value}"
                    ),
                )
            _validate_evidence(result.evidence, self.allowed_evidence_kinds)
            prior_ids = {item.evidence_id for item in evidence}
            duplicate_prior_ids = sorted(
                item.evidence_id for item in result.evidence if item.evidence_id in prior_ids
            )
            if duplicate_prior_ids:
                raise StageProtocolError(
                    StageValidationCode.EVIDENCE_SHAPE,
                    Usage(),
                    repairable=False,
                    message=f"evidence IDs duplicate prior evidence: {duplicate_prior_ids}",
                )
            candidate = _normalize_candidate_answer(result.candidate_answer)
            return result.model_copy(update={"candidate_answer": candidate})

        try:
            result, usage, repaired = await _complete_with_local_repair(
                self.provider,
                request,
                parse,
                executor=tools,
            )
        except StageProtocolError as error:
            self._last_usage.set(error.usage)
            self._last_repair.set(error.repairable)
            raise
        self._last_repair.set(repaired)
        self._last_usage.set(usage)
        return result.model_copy(update={"usage": usage})


class LookupAgent(SpecialistAgent):
    name = SpecialistName.LOOKUP
    allowed_tool_names = frozenset({"web_search", "read_webpage"})
    allowed_evidence_kinds = frozenset({"source"})
    instructions = (
        "Use only the supplied web tools to find observable evidence, prioritizing primary "
        "sources. Use at most three web searches and at most three webpage reads, then synthesize "
        "the evidence. Return source URLs and concise summaries; do not invent unavailable facts "
        "or request hidden reasoning. A candidate answer must follow the GAIA format: a number, "
        "as few words as possible, or an ordered comma-separated list without prose."
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
    allowed_tool_names = frozenset(
        {"read_attachment", "execute_python", "transcribe_audio"}
    )
    allowed_evidence_kinds = frozenset({"file", "computation"})
    instructions = (
        "Inspect only the registered task attachment through read_attachment, transcribe_audio, "
        "or execute_python. Use transcribe_audio for registered audio instead of uploading audio "
        "to Code Interpreter. Return observable file or computation evidence; do not invent "
        "missing content or request hidden reasoning. Keep any candidate answer in the concise "
        "GAIA answer format."
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
    allowed_tool_names = frozenset(
        {"web_search", "read_webpage", "execute_python", "transcribe_audio"}
    )
    instructions = (
        "Independently verify the candidate against the observable evidence. Check sufficiency and "
        "contradictions, reject unsupported research answers, and use the supplied web or Python "
        "tools only when the existing evidence is insufficient or contradictory. Use "
        "transcribe_audio for a registered audio attachment; use execute_python only for "
        "non-audio computation or files. Never request hidden reasoning and never invent facts. "
        "Approve only evidence-backed candidates. If a revision is necessary, propose exactly one "
        "concise correction directly supported by the checked evidence; do not solve the task "
        "freely."
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
        self._last_usage: ContextVar[Usage | None] = ContextVar(
            f"critic_usage_{id(self)}", default=None
        )
        self._last_repair: ContextVar[bool] = ContextVar(
            f"critic_repair_{id(self)}", default=False
        )

    @property
    def last_usage(self) -> Usage:
        return self._last_usage.get() or Usage()

    @property
    def last_repair_attempted(self) -> bool:
        return self._last_repair.get()

    async def verify(
        self,
        task: GaiaTask,
        candidate: str | None,
        evidence: tuple[Evidence, ...],
        tools: MCPToolClient,
    ) -> CriticDecision:
        self._last_usage.set(Usage())
        self._last_repair.set(False)
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
            max_output_tokens=_max_output_tokens(self.reasoning_effort),
            max_tool_calls=self.max_tool_calls,
            reasoning_effort=self.reasoning_effort,
        )
        known_ids = {item.evidence_id for item in evidence}

        def parse(response: Any) -> CriticDecision:
            try:
                decision = CriticDecision.model_validate(response.parsed)
            except ValidationError as error:
                raise StageProtocolError(StageValidationCode.SCHEMA, Usage()) from error
            corrected_answer = _normalize_candidate_answer(decision.corrected_answer)
            if decision.verdict == "revise" and not corrected_answer:
                raise StageProtocolError(
                    StageValidationCode.CANDIDATE_FORMAT,
                    Usage(),
                    message="revised critic decision must propose one correction",
                )
            if decision.verdict != "revise" and corrected_answer is not None:
                raise StageProtocolError(
                    StageValidationCode.CANDIDATE_FORMAT,
                    Usage(),
                    message="only a revised critic decision may correct the answer",
                )
            if decision.verdict == "approve" and not decision.checked_evidence_ids:
                raise StageProtocolError(
                    StageValidationCode.CHECKED_EVIDENCE,
                    Usage(),
                    message="approved critic decision must name checked evidence",
                )
            if decision.verdict == "approve" and candidate is None:
                raise StageProtocolError(
                    StageValidationCode.CANDIDATE_FORMAT,
                    Usage(),
                    message="critic cannot approve a missing candidate",
                )
            unknown_ids = set(decision.checked_evidence_ids) - known_ids
            if unknown_ids:
                raise StageProtocolError(
                    StageValidationCode.CHECKED_EVIDENCE,
                    Usage(),
                    repairable=False,
                    message=f"critic checked unknown evidence IDs: {sorted(unknown_ids)}",
                )
            if len(set(decision.checked_evidence_ids)) != len(decision.checked_evidence_ids):
                raise StageProtocolError(
                    StageValidationCode.CHECKED_EVIDENCE,
                    Usage(),
                    message="critic checked evidence IDs must be unique",
                )
            has_allowed_tool_evidence = any(
                event.name in self.allowed_tool_names
                and _has_tool_contract_evidence(event.name, event.output)
                for event in response.tool_events
            )
            if (
                decision.verdict == "revise"
                and not decision.checked_evidence_ids
                and not has_allowed_tool_evidence
            ):
                raise StageProtocolError(
                    StageValidationCode.CHECKED_EVIDENCE,
                    Usage(),
                    repairable=False,
                    message=(
                        "revised critic decision must be grounded in checked evidence "
                        "or an allowed tool event"
                    ),
                )
            return decision.model_copy(update={"corrected_answer": corrected_answer})

        try:
            decision, usage, repaired = await _complete_with_local_repair(
                self.provider,
                request,
                parse,
                executor=tools,
            )
        except StageProtocolError as error:
            self._last_usage.set(error.usage)
            if not error.repairable:
                raise
            if error.code != StageValidationCode.SCHEMA:
                raise
            self._last_repair.set(True)
            return _fallback_critic_decision(candidate, evidence)
        self._last_repair.set(repaired)
        self._last_usage.set(usage)
        return decision


def _fallback_critic_decision(
    candidate: str | None,
    evidence: tuple[Evidence, ...],
) -> CriticDecision:
    """Keep a useful attempt alive without inventing evidence after critic failure."""
    checked_ids = tuple(item.evidence_id for item in evidence)
    if candidate is not None and checked_ids:
        return CriticDecision(
            verdict="approve",
            feedback="Local critic fallback accepted the candidate with available evidence.",
            checked_evidence_ids=checked_ids,
        )
    return CriticDecision(
        verdict="insufficient_evidence",
        feedback="Critic output could not be repaired and no evidence-backed approval is safe.",
        checked_evidence_ids=checked_ids,
    )


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


async def _specialist_input(
    task: GaiaTask,
    step: PlanStep,
    evidence: tuple[Evidence, ...],
    *,
    visual: bool,
    tools: MCPToolClient,
    max_json_bytes: int = _DEFAULT_MAX_CONTEXT_BYTES,
) -> tuple[dict[str, Any], ...]:
    if max_json_bytes < 1:
        raise ValueError("max_json_bytes must be positive")
    attachment = _attachment_metadata(task)
    context = _bounded_specialist_context(task, step, evidence, max_json_bytes)
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
        encoded = b64encode(
            await tools.read_attachment_bytes(task.attachment_path)
        ).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{encoded}",
                "detail": "high",
            }
        )
    return ({"role": "user", "content": content},)


def context_size_bytes(input_items: Sequence[Mapping[str, Any]]) -> int:
    """Return the exact UTF-8 JSON size of model input items."""
    return len(
        json.dumps(
            list(input_items),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def specialist_context_size_bytes(
    task: GaiaTask,
    step: PlanStep,
    evidence: Sequence[Evidence],
    *,
    max_json_bytes: int = _DEFAULT_MAX_CONTEXT_BYTES,
) -> tuple[int, int]:
    """Return the bounded text-context size and selected evidence-card count."""
    context = _bounded_specialist_context(task, step, evidence, max_json_bytes)
    return context_size_bytes(_text_input_items(context)), len(context["evidence"])


def _bounded_specialist_context(
    task: GaiaTask,
    step: PlanStep,
    evidence: Sequence[Evidence],
    max_json_bytes: int,
) -> dict[str, Any]:
    attachment = _attachment_metadata(task)
    context: dict[str, Any] = {
        "task_id": task.task_id,
        "question": _bounded_context_text(task.question, _MAX_CONTEXT_QUESTION_CHARS),
        "level": task.level,
        "instruction": _bounded_context_text(step.instruction, _MAX_CONTEXT_INSTRUCTION_CHARS),
        "attachment": attachment,
        "evidence": [],
    }
    available_card_bytes = max(
        0,
        max_json_bytes - context_size_bytes(_text_input_items(context)),
    )
    context["evidence"] = list(
        select_evidence_cards(
            _notebook_entries_for_evidence(evidence),
            max_json_bytes=available_card_bytes,
        )
    )
    return context


def _text_input_items(context: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return (
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": json.dumps(
                        context,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                }
            ],
        },
    )


def _notebook_entries_for_evidence(evidence: Sequence[Evidence]) -> tuple[NotebookEntry, ...]:
    return tuple(
        NotebookEntry(
            signature=item.evidence_id,
            tool=item.kind,
            result={
                "evidence_id": item.evidence_id,
                "source_url": item.source_url,
                "summary": item.summary,
            },
            created_at=_EVIDENCE_CARD_TIMESTAMP,
        )
        for item in evidence
    )


def _bounded_context_text(value: str, limit: int) -> str:
    return " ".join(value.split())[:limit]


def _attachment_metadata(task: GaiaTask) -> dict[str, str] | None:
    if task.attachment_path is None:
        return None
    media_type = mimetypes.guess_type(task.attachment_path.name)[0]
    return {
        "path": str(task.attachment_path),
        "name": task.attachment_path.name,
        "media_type": media_type or "application/octet-stream",
    }


def _provider_supports_vision(provider: ModelProvider) -> bool:
    # Providers written before the capability flag are treated as vision-capable
    # for backwards compatibility with injected test providers.
    return bool(getattr(provider, "supports_vision", True))


def _allowed_specialists(
    task: GaiaTask,
    provider: ModelProvider,
) -> tuple[SpecialistName, ...]:
    attachment = _attachment_metadata(task)
    allowed = [SpecialistName.LOOKUP, SpecialistName.COMPUTATION]
    if attachment is not None:
        allowed.append(SpecialistName.FILE)
        if attachment["media_type"].startswith("image/") and _provider_supports_vision(provider):
            allowed.append(SpecialistName.VISUAL)
    return tuple(allowed)


def _fallback_specialist(task: GaiaTask) -> tuple[SpecialistName, str]:
    question = task.question.casefold()
    computation_cues = (
        "calculate",
        "compute",
        "how many",
        "how much",
        "times",
        "multiply",
        "divide",
        "percent",
        "percentage",
        "average",
        "ratio",
        "sum",
        "difference",
    )
    if any(cue in question for cue in computation_cues):
        return (
            SpecialistName.COMPUTATION,
            "Use the computation tool to calculate the requested value.",
        )
    return (
        SpecialistName.LOOKUP,
        "Use web tools to find observable primary-source evidence.",
    )


def normalize_plan(
    plan: AgentPlan,
    task: GaiaTask,
    provider: ModelProvider,
    max_initial_steps: int = 2,
) -> AgentPlan:
    """Rewrite impossible routes and keep only the initial executable work."""
    if max_initial_steps < 1:
        raise ValueError("max_initial_steps must be positive")
    allowed = set(_allowed_specialists(task, provider))
    attachment = _attachment_metadata(task)
    routed: list[PlanStep] = []
    for step in plan.steps:
        if step.specialist in allowed:
            routed_step = step
        elif step.specialist == SpecialistName.VISUAL and SpecialistName.FILE in allowed:
            reason = (
                "the attachment is not an image"
                if attachment is not None and not attachment["media_type"].startswith("image/")
                else "the configured provider has no native image-input capability"
            )
            routed_step = step.model_copy(
                update={
                    "specialist": SpecialistName.FILE,
                    "instruction": (
                        f"{step.instruction} Use the registered attachment with file tools; {reason}."
                    ),
                }
            )
        else:
            specialist, instruction = _fallback_specialist(task)
            routed_step = step.model_copy(
                update={
                    "specialist": specialist,
                    "instruction": f"{step.instruction} {instruction}",
                }
            )
        if routed and routed[-1].specialist == routed_step.specialist:
            routed[-1] = routed[-1].model_copy(
                update={
                    "instruction": f"{routed[-1].instruction}\n{routed_step.instruction}",
                }
            )
        else:
            routed.append(routed_step)
    return plan.model_copy(update={"steps": tuple(routed[:max_initial_steps])})


def _fallback_plan(task: GaiaTask, provider: ModelProvider) -> AgentPlan:
    """Build an answer-free route when the planner protocol cannot be repaired."""
    attachment = _attachment_metadata(task)
    if attachment is not None:
        if attachment["media_type"].startswith("image/") and _provider_supports_vision(provider):
            specialist = SpecialistName.VISUAL
            instruction = "Inspect the registered image and report only observable visual evidence."
        else:
            specialist = SpecialistName.FILE
            instruction = "Inspect the registered attachment with the appropriate file tool."
    else:
        specialist, instruction = _fallback_specialist(task)
    return AgentPlan(
        restatement="Use a conservative evidence-gathering route for the task.",
        steps=(PlanStep(specialist=specialist, instruction=instruction),),
    )


def _validate_evidence(
    evidence: tuple[Evidence, ...],
    allowed_evidence_kinds: frozenset[str],
) -> None:
    evidence_ids = [item.evidence_id for item in evidence]
    if any(not evidence_id.strip() for evidence_id in evidence_ids):
        raise StageProtocolError(
            StageValidationCode.EVIDENCE_SHAPE,
            Usage(),
            message="evidence IDs must be non-empty",
        )
    if len(set(evidence_ids)) != len(evidence_ids):
        raise StageProtocolError(
            StageValidationCode.EVIDENCE_SHAPE,
            Usage(),
            message="evidence IDs must be unique",
        )
    for item in evidence:
        if item.kind not in allowed_evidence_kinds:
            raise StageProtocolError(
                StageValidationCode.EVIDENCE_SHAPE,
                Usage(),
                message=f"evidence kind {item.kind} is not allowed for this specialist",
            )
        if item.kind == "source" and not item.source_url:
            raise StageProtocolError(
                StageValidationCode.SOURCE_URL_MISSING,
                Usage(),
                message="source evidence must include a URL",
            )


def _normalize_candidate_answer(candidate: str | None) -> str | None:
    if candidate is None:
        return None
    normalized = candidate.strip()
    if (
        not normalized
        or "\n" in normalized
        or normalized.casefold().startswith("final answer:")
    ):
        raise StageProtocolError(
            StageValidationCode.CANDIDATE_FORMAT,
            Usage(),
            message="candidate answer must use the concise GAIA format",
        )
    return normalized


def _extract_final_answer(text: str) -> str | None:
    """Recover a concise answer when a provider returns prose instead of JSON."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    marker = next(
        (line for line in reversed(lines) if line.casefold().startswith("final answer:")),
        None,
    )
    if marker is not None:
        candidate = marker.split(":", 1)[1].strip()
    elif len(lines) == 1:
        candidate = lines[0]
    else:
        return None
    candidate = candidate.strip().strip("`").strip()
    if not candidate or candidate.casefold().startswith(_NON_ANSWER_PREFIXES):
        return None
    return candidate


def _has_tool_contract_evidence(name: str, output: dict[str, Any]) -> bool:
    if name == "web_search":
        return _has_web_search_evidence(output)
    if name == "read_webpage":
        return any(_has_meaningful_text(output.get(field)) for field in ("text", "content", "body"))
    if name == "execute_python":
        return _has_meaningful_text(output.get("output_text")) or _has_python_output_content(
            output.get("outputs"), depth=0
        )
    if name == "transcribe_audio":
        return _has_meaningful_text(output.get("text"))
    return False


def _has_web_search_evidence(output: dict[str, Any]) -> bool:
    if any(_has_meaningful_text(output.get(field)) for field in ("answer", "text")):
        return True
    return any(
        _has_web_result_entries(output.get(field)) for field in ("sources", "results")
    )


def _has_web_result_entries(value: Any) -> bool:
    if _has_meaningful_text(value):
        return True
    entries = (value,) if isinstance(value, dict) else value
    if not isinstance(entries, (list, tuple)):
        return False
    for index, entry in enumerate(entries):
        if index >= _MAX_TOOL_RESULT_ITEMS:
            break
        if _has_meaningful_text(entry):
            return True
        if isinstance(entry, dict) and any(
            _has_meaningful_text(entry.get(field)) for field in _WEB_RESULT_FIELDS
        ):
            return True
    return False


def _has_python_output_content(value: Any, *, depth: int) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bool, int, float)):
        return True
    if depth >= _MAX_TOOL_RESULT_DEPTH:
        return False
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if index >= _MAX_TOOL_RESULT_ITEMS:
                break
            if _has_python_output_content(item, depth=depth + 1):
                return True
        return False
    if isinstance(value, dict):
        return any(
            field in value
            and _has_python_output_content(value[field], depth=depth + 1)
            for field in _PYTHON_RESULT_FIELDS
        )
    return False


def _has_meaningful_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
