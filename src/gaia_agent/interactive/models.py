from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from gaia_agent.config import RunConfig
from gaia_agent.models import Evidence, Usage


class InteractiveConfig(RunConfig):
    hf_token: SecretStr = SecretStr("")
    model: Literal["gpt-6.1-sol"] = "gpt-6.1-sol"
    reasoning_effort: Literal["low"] = "low"
    timeout_seconds: float = 500
    planner_timeout_seconds: float = 30
    specialist_timeout_seconds: float = 410
    critic_timeout_seconds: float = 60
    provider_max_attempts: int = 3
    output_dir: Path = Path("interactive-data")


class InteractiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    question: str = Field(min_length=1, max_length=12_000)
    attachment_id: str | None = Field(default=None, max_length=64)


class InteractiveResult(BaseModel):
    answer: str
    evidence: list[Evidence]
    usage: Usage
