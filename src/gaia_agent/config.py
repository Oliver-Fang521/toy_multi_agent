import os
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

MAX_STAGE_CONTEXT_BYTES = 1_048_576


class RunConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    openai_api_key: SecretStr
    hf_token: SecretStr
    model: str = "gpt-5.6-luna"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "max"
    quality_policy: Literal["benchmark", "strict"] = "benchmark"
    sample_size: int = 15
    seed: int = 42
    max_steps: int = 8
    max_tool_calls: int = 24
    timeout_seconds: float = 300.0
    votes: int = 1
    output_dir: Path = Path("results/gaia-15-seed42")
    resume: bool = False
    provider_max_attempts: int = Field(default=4, ge=1, le=10)
    provider_backoff_base: float = Field(default=0.5, gt=0, le=30, allow_inf_nan=False)
    private_notebook_max_bytes: int = Field(default=65_536, ge=1, le=1_048_576)
    stage_context_max_bytes: int = Field(default=24_576, ge=1, le=MAX_STAGE_CONTEXT_BYTES)
    search_circuit_cooldown: float = Field(default=30.0, ge=0, le=3_600, allow_inf_nan=False)
    langfuse_input_cost_per_million: Decimal | None = Field(
        default=None, ge=0, le=Decimal(1_000_000), allow_inf_nan=False
    )
    langfuse_output_cost_per_million: Decimal | None = Field(
        default=None, ge=0, le=Decimal(1_000_000), allow_inf_nan=False
    )

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
            "model": os.environ.get("GAIA_MODEL", "gpt-5.6-luna"),
            "reasoning_effort": os.environ.get("GAIA_REASONING_EFFORT", "max"),
            "quality_policy": os.environ.get("GAIA_QUALITY_POLICY", "benchmark"),
            "sample_size": os.environ.get("GAIA_SAMPLE_SIZE", "15"),
            "seed": os.environ.get("GAIA_SEED", "42"),
            "max_steps": os.environ.get("GAIA_MAX_STEPS", "8"),
            "max_tool_calls": os.environ.get("GAIA_MAX_TOOL_CALLS", "24"),
            "timeout_seconds": os.environ.get("GAIA_TIMEOUT_SECONDS", "300"),
            "votes": os.environ.get("GAIA_VOTES", "1"),
            "output_dir": os.environ.get("GAIA_OUTPUT_DIR", "results/gaia-15-seed42"),
            "resume": os.environ.get("GAIA_RESUME", "false"),
            "provider_max_attempts": os.environ.get("GAIA_PROVIDER_MAX_ATTEMPTS", "4"),
            "provider_backoff_base": os.environ.get("GAIA_PROVIDER_BACKOFF_BASE", "0.5"),
            "private_notebook_max_bytes": os.environ.get(
                "GAIA_PRIVATE_NOTEBOOK_MAX_BYTES", "65536"
            ),
            "stage_context_max_bytes": os.environ.get("GAIA_STAGE_CONTEXT_MAX_BYTES", "24576"),
            "search_circuit_cooldown": os.environ.get("GAIA_SEARCH_CIRCUIT_COOLDOWN", "30"),
            "langfuse_input_cost_per_million": os.environ.get(
                "LANGFUSE_INPUT_COST_PER_MILLION"
            ),
            "langfuse_output_cost_per_million": os.environ.get(
                "LANGFUSE_OUTPUT_COST_PER_MILLION"
            ),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)
