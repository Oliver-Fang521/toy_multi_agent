import pytest
from pydantic import ValidationError

from gaia_agent.config import RunConfig
from gaia_agent.models import AuditEvent, GaiaTask, TaskPrediction, TerminalErrorCategory


def test_run_config_requires_balanced_sample_size() -> None:
    with pytest.raises(ValidationError, match="divisible by 3"):
        RunConfig(openai_api_key="sk-test", hf_token="hf_test", sample_size=10)


def test_run_config_defaults_to_luna_max_in_direct_and_environment_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment")
    monkeypatch.setenv("HF_TOKEN", "hf-environment")
    monkeypatch.delenv("GAIA_MODEL", raising=False)
    monkeypatch.delenv("GAIA_REASONING_EFFORT", raising=False)

    direct = RunConfig(openai_api_key="sk-test", hf_token="hf-test")
    from_env = RunConfig.from_env()

    assert (direct.model, direct.reasoning_effort) == ("gpt-5.6-luna", "max")
    assert (from_env.model, from_env.reasoning_effort) == ("gpt-5.6-luna", "max")


def test_run_config_redacts_secrets() -> None:
    config = RunConfig(openai_api_key="sk-secret", hf_token="hf_secret")
    assert "sk-secret" not in repr(config)
    assert "hf_secret" not in repr(config)


def test_gaia_task_rejects_unknown_level() -> None:
    with pytest.raises(ValidationError):
        GaiaTask(task_id="x", question="q", level=4, final_answer="a")


def test_run_config_from_env_allows_non_none_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment")
    monkeypatch.setenv("HF_TOKEN", "hf-environment")
    monkeypatch.setenv("GAIA_MAX_STEPS", "12")

    config = RunConfig.from_env(model="test-model", max_steps=None)

    assert config.model == "test-model"
    assert config.max_steps == 12


def test_run_config_reads_benchmark_quality_policy_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment")
    monkeypatch.setenv("HF_TOKEN", "hf-environment")
    monkeypatch.setenv("GAIA_QUALITY_POLICY", "benchmark")

    config = RunConfig.from_env()

    assert config.quality_policy == "benchmark"


def test_run_config_accepts_bounded_reliability_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment")
    monkeypatch.setenv("HF_TOKEN", "hf-environment")
    monkeypatch.setenv("GAIA_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("GAIA_REASONING_EFFORT", "max")
    monkeypatch.setenv("GAIA_PROVIDER_MAX_ATTEMPTS", "6")
    monkeypatch.setenv("GAIA_PROVIDER_BACKOFF_BASE", "1.25")
    monkeypatch.setenv("GAIA_PRIVATE_NOTEBOOK_MAX_BYTES", "80000")
    monkeypatch.setenv("GAIA_STAGE_CONTEXT_MAX_BYTES", "32768")
    monkeypatch.setenv("GAIA_SEARCH_CIRCUIT_COOLDOWN", "90")
    monkeypatch.setenv("LANGFUSE_INPUT_COST_PER_MILLION", "0.25")
    monkeypatch.setenv("LANGFUSE_OUTPUT_COST_PER_MILLION", "1.50")

    config = RunConfig.from_env()

    assert config.model == "gpt-5.6-luna"
    assert config.reasoning_effort == "max"
    assert config.provider_max_attempts == 6
    assert config.provider_backoff_base == 1.25
    assert config.private_notebook_max_bytes == 80_000
    assert config.stage_context_max_bytes == 32_768
    assert config.search_circuit_cooldown == 90
    assert str(config.langfuse_input_cost_per_million) == "0.25"
    assert str(config.langfuse_output_cost_per_million) == "1.50"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_max_attempts", 0),
        ("provider_max_attempts", 11),
        ("provider_backoff_base", -0.1),
        ("provider_backoff_base", float("nan")),
        ("provider_backoff_base", float("inf")),
        ("provider_backoff_base", 31.0),
        ("private_notebook_max_bytes", 0),
        ("private_notebook_max_bytes", 1_048_577),
        ("stage_context_max_bytes", 0),
        ("stage_context_max_bytes", 1_048_577),
        ("search_circuit_cooldown", -1.0),
        ("search_circuit_cooldown", float("nan")),
        ("search_circuit_cooldown", 3_601.0),
        ("langfuse_input_cost_per_million", "-0.01"),
        ("langfuse_output_cost_per_million", "NaN"),
        ("langfuse_output_cost_per_million", "1000000.01"),
    ],
)
def test_run_config_rejects_unsafe_reliability_settings(field: str, value: object) -> None:
    values = {"openai_api_key": "sk-test", "hf_token": "hf-test", field: value}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(values)


def test_run_config_redacts_invalid_reliability_setting_from_validation_error() -> None:
    secret = "sk-this-must-not-appear-in-validation-output"

    with pytest.raises(ValidationError) as error:
        RunConfig.model_validate(
            {
                "openai_api_key": "sk-test",
                "hf_token": "hf-test",
                "langfuse_input_cost_per_million": secret,
            }
        )

    assert secret not in str(error.value)


def test_domain_models_are_immutable() -> None:
    task = GaiaTask(task_id="x", question="q", level=1, final_answer="a")

    with pytest.raises(ValidationError):
        task.level = 2


def test_audit_event_redacts_secret_values_and_prediction_serializes_error_category() -> None:
    event = AuditEvent(
        sequence=1,
        event_type="tool_failure",
        timestamp="2026-08-02T00:00:00Z",
        payload={"api_key": "sk-secret", "nested": {"token": "hf-secret"}},
    )
    prediction = TaskPrediction(
        task_id="x",
        level=1,
        answer=None,
        critic_verdict="insufficient_evidence",
        error_category=TerminalErrorCategory.TIMEOUT,
    )

    assert event.payload == {"api_key": "[REDACTED]", "nested": {"token": "[REDACTED]"}}
    assert prediction.model_dump(mode="json")["error_category"] == "timeout"
