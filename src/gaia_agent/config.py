import os
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr, field_validator, model_validator


class RunConfig(BaseModel):
    openai_api_key: SecretStr
    hf_token: SecretStr
    model: str = "deepseek-v4-flash"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    quality_policy: Literal["benchmark", "strict"] = "benchmark"
    sample_size: int = 15
    seed: int = 42
    max_steps: int = 8
    max_tool_calls: int = 24
    timeout_seconds: float = 300.0
    votes: int = 1
    output_dir: Path = Path("results/gaia-15-seed42")
    resume: bool = False
    langfuse_input_cost_per_million: Decimal | None = None
    langfuse_output_cost_per_million: Decimal | None = None

    @field_validator("langfuse_input_cost_per_million", "langfuse_output_cost_per_million")
    @classmethod
    def validate_langfuse_cost_rate(cls, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        if not value.is_finite() or value < 0:
            raise ValueError("Langfuse cost rates must be non-negative finite decimals")
        return value

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
            "model": os.environ.get("GAIA_MODEL", "deepseek-v4-flash"),
            "reasoning_effort": os.environ.get("GAIA_REASONING_EFFORT", "medium"),
            "quality_policy": os.environ.get("GAIA_QUALITY_POLICY", "benchmark"),
            "sample_size": os.environ.get("GAIA_SAMPLE_SIZE", "15"),
            "seed": os.environ.get("GAIA_SEED", "42"),
            "max_steps": os.environ.get("GAIA_MAX_STEPS", "8"),
            "max_tool_calls": os.environ.get("GAIA_MAX_TOOL_CALLS", "24"),
            "timeout_seconds": os.environ.get("GAIA_TIMEOUT_SECONDS", "300"),
            "votes": os.environ.get("GAIA_VOTES", "1"),
            "output_dir": os.environ.get("GAIA_OUTPUT_DIR", "results/gaia-15-seed42"),
            "resume": os.environ.get("GAIA_RESUME", "false"),
            "langfuse_input_cost_per_million": os.environ.get(
                "LANGFUSE_INPUT_COST_PER_MILLION"
            ),
            "langfuse_output_cost_per_million": os.environ.get(
                "LANGFUSE_OUTPUT_COST_PER_MILLION"
            ),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)
