import pytest
from pydantic import ValidationError

from gaia_agent.config import RunConfig
from gaia_agent.models import AuditEvent, GaiaTask, TaskPrediction, TerminalErrorCategory


def test_run_config_requires_balanced_sample_size() -> None:
    with pytest.raises(ValidationError, match="divisible by 3"):
        RunConfig(openai_api_key="sk-test", hf_token="hf_test", sample_size=10)


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
