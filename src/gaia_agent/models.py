import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class SpecialistName(StrEnum):
    LOOKUP = "lookup"
    COMPUTATION = "computation"
    FILE = "file"
    VISUAL = "visual"


class TerminalErrorCategory(StrEnum):
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    LOOP_DETECTED = "loop_detected"
    CRITIC_REJECTION = "critic_rejection"
    ATTACHMENT_ERROR = "attachment_error"
    RETRY_EXHAUSTED = "retry_exhausted"
    NO_CONSENSUS = "no_consensus"


class GaiaTask(FrozenModel):
    task_id: str
    question: str
    level: Literal[1, 2, 3]
    final_answer: str = Field(exclude=True)
    attachment_path: Path | None = None


class PlanStep(FrozenModel):
    specialist: SpecialistName
    instruction: str


class AgentPlan(FrozenModel):
    restatement: str
    steps: tuple[PlanStep, ...]


class Evidence(FrozenModel):
    evidence_id: str
    kind: Literal["source", "computation", "file", "visual"]
    summary: str
    source_url: str | None = None


class Usage(FrozenModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


HOSTED_USAGE_METADATA_KEY = "__gaia_usage__"


class AgentResult(FrozenModel):
    specialist: SpecialistName
    evidence: tuple[Evidence, ...] = ()
    candidate_answer: str | None = None
    usage: Usage = Field(default_factory=Usage)


class CriticDecision(FrozenModel):
    verdict: Literal["approve", "revise", "insufficient_evidence"]
    feedback: str
    checked_evidence_ids: tuple[str, ...]
    corrected_answer: str | None = None


StageCursor = Literal["act", "evidence_repair", "verify", "correct", "submit"]

_MAX_CHECKPOINT_COUNTER = 1_000_000
_MAX_CHECKPOINT_USAGE = 1_000_000_000_000
_TRUSTED_EVIDENCE_ID = re.compile(r"evidence-[0-9]{4}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


class StageToolCallCounts(FrozenModel):
    """Cumulative logical calls by budget bucket, never raw tool content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner: int = Field(default=0, ge=0, le=_MAX_CHECKPOINT_COUNTER)
    specialist: int = Field(default=0, ge=0, le=_MAX_CHECKPOINT_COUNTER)
    critic: int = Field(default=0, ge=0, le=_MAX_CHECKPOINT_COUNTER)

    @property
    def total(self) -> int:
        return self.planner + self.specialist + self.critic


class StageCheckpoint(FrozenModel):
    """Authenticated private state from the latest stable graph transition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cursor: StageCursor
    plan: AgentPlan
    evidence: tuple[Evidence, ...]
    evidence_origins: dict[str, int]
    raw_evidence_id_digests: tuple[str, ...]
    candidate_origins: dict[int, str]
    candidate: str | None
    usage: Usage
    step_count: int = Field(ge=0, le=_MAX_CHECKPOINT_COUNTER)
    verify_count: int = Field(ge=0, le=_MAX_CHECKPOINT_COUNTER)
    correction_count: int = Field(ge=0, le=1)
    next_step_index: int = Field(ge=0, le=_MAX_CHECKPOINT_COUNTER)
    soft_step_failures: int = Field(ge=0, le=1)
    logical_tool_calls: int = Field(ge=0, le=_MAX_CHECKPOINT_COUNTER)
    stage_logical_tool_calls: StageToolCallCounts
    critic_verdict: Literal[
        "approve", "revise", "insufficient_evidence", "not_run"
    ]
    decision: CriticDecision | None = None

    @model_validator(mode="after")
    def validate_resume_invariants(self) -> "StageCheckpoint":
        if not self.plan.steps:
            raise ValueError("checkpoint plan must contain at least one step")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        expected_ids = tuple(
            f"evidence-{index:04d}" for index in range(1, len(self.evidence) + 1)
        )
        if len(self.evidence) > 9_999 or evidence_ids != expected_ids:
            raise ValueError("checkpoint evidence IDs must be unique trusted sequential IDs")
        if any(
            not _TRUSTED_EVIDENCE_ID.fullmatch(item.evidence_id)
            or (item.kind == "source" and not item.source_url)
            for item in self.evidence
        ):
            raise ValueError("checkpoint evidence is not trusted typed evidence")
        if set(self.evidence_origins) != set(evidence_ids):
            raise ValueError("checkpoint evidence lineage must exactly match evidence")
        if any(
            isinstance(origin, bool) or origin < 1 or origin > self.step_count
            for origin in self.evidence_origins.values()
        ):
            raise ValueError("checkpoint evidence lineage contains an invalid step")
        if (
            len(self.raw_evidence_id_digests) != len(self.evidence)
            or len(set(self.raw_evidence_id_digests)) != len(self.raw_evidence_id_digests)
            or tuple(sorted(self.raw_evidence_id_digests))
            != self.raw_evidence_id_digests
            or any(
                not _SHA256_HEX.fullmatch(digest)
                for digest in self.raw_evidence_id_digests
            )
        ):
            raise ValueError("checkpoint raw evidence digests are invalid")
        if any(
            isinstance(origin, bool)
            or origin < 1
            or origin > self.step_count
            or not value.strip()
            or "\n" in value
            for origin, value in self.candidate_origins.items()
        ):
            raise ValueError("checkpoint candidate lineage is invalid")
        if self.candidate is not None and (
            not self.candidate.strip() or "\n" in self.candidate
        ):
            raise ValueError("checkpoint candidate must use the concise GAIA format")
        if self.cursor == "act" and self.next_step_index >= len(self.plan.steps):
            raise ValueError("checkpoint act cursor requires an executable next step")
        if self.cursor != "act" and self.next_step_index > len(self.plan.steps):
            raise ValueError("checkpoint next step exceeds the plan")
        if self.logical_tool_calls != self.stage_logical_tool_calls.total:
            raise ValueError("checkpoint logical tool counters do not reconcile")
        usage_values = (
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.usage.reasoning_tokens,
            self.usage.total_tokens,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > _MAX_CHECKPOINT_USAGE
            for value in usage_values
        ):
            raise ValueError("checkpoint usage counters are invalid")
        if self.decision is not None:
            checked_ids = self.decision.checked_evidence_ids
            if len(checked_ids) != len(set(checked_ids)) or set(checked_ids) - set(
                evidence_ids
            ):
                raise ValueError("checkpoint critic decision references invalid evidence")
            if self.decision.verdict != self.critic_verdict:
                raise ValueError("checkpoint critic verdict does not match its decision")
            if self.decision.verdict == "approve" and self.candidate is None:
                raise ValueError("checkpoint critic approved a missing candidate")
            if self.decision.verdict == "revise" and not (
                self.decision.corrected_answer
                and self.decision.corrected_answer.strip()
                and "\n" not in self.decision.corrected_answer
            ):
                raise ValueError("checkpoint critic revision requires a concise answer")
            if (
                self.decision.verdict != "revise"
                and self.decision.corrected_answer is not None
            ):
                raise ValueError("checkpoint non-revision has a corrected answer")
        if self.cursor in {"correct", "submit"} and self.decision is None:
            raise ValueError("checkpoint cursor requires a critic decision")
        if self.cursor == "submit" and (
            self.candidate is None
            or not self.evidence
            or self.decision is None
            or self.decision.verdict != "approve"
            or not self.decision.checked_evidence_ids
        ):
            raise ValueError("checkpoint submit cursor requires checked evidence and approval")
        if (
            self.cursor == "correct"
            and self.decision is not None
            and self.decision.verdict == "approve"
        ):
            raise ValueError("checkpoint correct cursor requires rejection")
        return self


class TaskPrediction(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    level: Literal[1, 2, 3]
    answer: str | None
    correct: bool = False
    duration_seconds: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    critic_verdict: str
    evidence_ids: tuple[str, ...] = ()
    error_category: TerminalErrorCategory | None = None
    run_index: int = 0
    vote_count: int = 1
    attempt_number: int = Field(default=1, ge=1, le=1_000_000)
    recovery_count: int = Field(default=0, ge=0, le=1_000_000)
    recovery_succeeded: bool = False
    retry_reason: Literal[
        "provider_transient_exhausted", "timeout", "task_transient"
    ] | None = None
    retry_category: str | None = None
    provider_attempt_count: int = Field(default=0, ge=0, le=1_000_000)


_SENSITIVE_KEY_PARTS = ("api_key", "authorization", "credential", "password", "secret", "token")


def _redact_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
            else _redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_payload(item) for item in value)
    return value


class AuditEvent(FrozenModel):
    sequence: int
    event_type: str
    timestamp: datetime
    payload: dict[str, Any]

    @model_validator(mode="after")
    def redact_payload(self) -> "AuditEvent":
        object.__setattr__(self, "payload", _redact_payload(self.payload))
        return self
