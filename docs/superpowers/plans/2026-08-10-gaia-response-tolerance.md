# GAIA Response Tolerance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GAIA agent responses tolerant enough for `gpt-5.6-luna` with `reasoning=max` while retaining strict validation where it protects scoring and audit integrity.

**Architecture:** Keep the provider-neutral `ModelResponse` boundary. Add a small, deterministic final-answer extraction helper used only after structured parsing fails; make Specialist optional fields defaultable at the model/schema boundary; choose a larger output budget only for max reasoning requests.

**Tech Stack:** Python 3.12+, Pydantic, OpenAI Responses API, pytest, Ruff, uv.

## Global Constraints

- Do not alter the GAIA answer scorer or reference-answer handling.
- Do not weaken Planner, Critic, tool-argument, evidence-ID, or attachment safety validation.
- Preserve the existing 2,000-token budget for reasoning levels other than `max`.
- Do not include raw provider responses, questions, or reference answers in public artifacts.
- Preserve unrelated existing worktree changes.

---

### Task 1: Tolerant final-answer extraction

**Files:**
- Modify: `src/gaia_agent/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Add a deterministic helper that receives provider output text and returns either a concise answer string or `None`.
- Structured JSON remains the first-choice path; fallback extraction is only used when structured parsing is absent.

- [x] **Step 1: Write failing tests** for `FINAL ANSWER:` extraction and plain one-line fallback.
- [x] **Step 2: Run the focused agent tests and confirm the new tests fail.**
- [x] **Step 3: Implement the minimal extraction helper and wire it after structured parsing.**
- [x] **Step 4: Run the focused agent tests and confirm they pass.**

### Task 2: Defaultable Specialist fields

**Files:**
- Modify: `src/gaia_agent/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Specialist parsing accepts omitted optional evidence, candidate, and usage fields with safe defaults.
- The outbound schema remains closed and fully required for OpenAI Structured Outputs; nullable fields represent semantically optional values.
- Planner and Critic continue using strict schemas and existing validation.

- [x] **Step 1: Write failing tests** for an omitted Specialist `evidence` and `candidate_answer` being defaulted, while an invalid specialist identity still fails.
- [x] **Step 2: Run the focused tests and confirm the new tests fail.**
- [x] **Step 3: Update only Specialist parsing defaults; retain the provider's strict schema and evidence validation.**
- [x] **Step 4: Run the focused tests and confirm they pass.**

### Task 3: Max-only output budget

**Files:**
- Modify: `src/gaia_agent/agents.py`.
- Test: `tests/test_agents.py`.

**Interfaces:**
- A request with `reasoning_effort="max"` uses `max_output_tokens=8_000`.
- Requests with `none`, `low`, `medium`, `high`, or `xhigh` continue using `2_000` unless explicitly configured elsewhere.

- [x] **Step 1: Write failing tests** asserting the max/non-max token budgets.
- [x] **Step 2: Run the focused tests and confirm the new tests fail.**
- [x] **Step 3: Implement the max-only budget selection.**
- [x] **Step 4: Run the focused tests and confirm they pass.**

### Task 4: Full verification

**Files:**
- No additional production files unless required by the focused tests.

- [x] **Step 1: Run the full test suite with `uv run pytest`.**
- [x] **Step 2: Run Ruff.**
- [x] **Step 3: Run `uv build`.**
- [x] **Step 4: Inspect the diff and verify unrelated worktree changes were not staged.**
