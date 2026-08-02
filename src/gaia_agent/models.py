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


class TaskPrediction(FrozenModel):
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
