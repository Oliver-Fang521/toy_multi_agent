import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr, model_validator


class RunConfig(BaseModel):
    openai_api_key: SecretStr
    hf_token: SecretStr
    model: str = "gpt-5.6-sol"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    sample_size: int = 15
    seed: int = 42
    max_steps: int = 8
    max_tool_calls: int = 24
    timeout_seconds: float = 300.0
    votes: int = 1
    output_dir: Path = Path("results/gaia-15-seed42")
    resume: bool = False

    @model_validator(mode="after")
    def validate_run_shape(self) -> "RunConfig":
        if self.sample_size < 3 or self.sample_size % 3:
            raise ValueError("sample_size must be at least 3 and divisible by 3")
        if self.votes < 1 or self.votes % 2 == 0:
            raise ValueError("votes must be a positive odd integer")
        return self

    @classmethod
    def from_env(cls, **overrides: object) -> "RunConfig":
        values: dict[str, object] = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY"),
            "hf_token": os.environ.get("HF_TOKEN"),
            "model": os.environ.get("GAIA_MODEL", "gpt-5.6-sol"),
            "reasoning_effort": os.environ.get("GAIA_REASONING_EFFORT", "medium"),
            "max_steps": os.environ.get("GAIA_MAX_STEPS", "8"),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)
