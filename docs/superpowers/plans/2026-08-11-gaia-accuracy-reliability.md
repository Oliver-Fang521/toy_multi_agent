# GAIA Agent Accuracy and Reliability Implementation Plan

> Execute in this order. Keep the current worktree changes intact and run the
> focused tests after each stage.

## 1. Add failing tests

- Extend `tests/test_agents.py` with a fake provider sequence proving that a
  malformed planner response is repaired locally and that a deterministic plan
  is returned when repair also fails.
- Add critic/specialist malformed-response coverage where useful, ensuring a
  candidate is not accepted without the existing evidence policy.
- Extend provider tests to assert OpenAI vision support and DeepSeek's lack of
  native vision.
- Extend `tests/test_config.py` for `GAIA_QUALITY_POLICY` and persisted config.
- Extend `tests/test_orchestrator.py` to compare `benchmark` and `strict` policy
  behavior using the same candidate/evidence.

## 2. Implement stage repair and planner fallback

- Add a small shared helper in `src/gaia_agent/agents.py` (or a focused module)
  that performs one schema-preserving repair call and returns the validated
  object plus usage.
- Use it from `PlannerAgent`, `SpecialistAgent`, and `CriticAgent`.
- Add deterministic plan construction based only on task text and validated
  attachment metadata; expose a traceable fallback marker.

## 3. Implement capability-aware routing

- Add `supports_vision` to the provider protocol and concrete providers.
- Make visual routing consult that capability and fail early when no safe text
  route exists.
- Keep attachment resolution and local path restrictions unchanged.

## 4. Make the quality gate explicit

- Add `quality_policy: benchmark | strict` to `RunConfig`, environment parsing,
  manifests, and CLI-visible configuration.
- Preserve the current evidence gate in strict mode; in benchmark mode do not
  reject a critic-approved, scorer-valid candidate solely for internal evidence
  density.

## 5. Verify and report

- Run focused tests, then `pytest -q`, `ruff check .`, and `uv build`.
- Report exactly which checks passed and how to select strict mode. Do not claim
  benchmark accuracy improved until a controlled evaluation is run.
