# Langfuse Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, privacy-preserving Langfuse traces to the GAIA agent without changing evaluation results.

**Architecture:** Add a narrow `Observability` adapter with real Langfuse and no-op implementations. Inject it at the runner/orchestrator/provider/MCP boundaries, passing only safe metadata and usage; all transport failures are isolated from the evaluation path.

**Tech Stack:** Python 3.11+, Langfuse SDK, Pydantic configuration, pytest, pytest-asyncio, Ruff.

## Global Constraints

- Tracing is disabled when `LANGFUSE_ENABLED=false` or either required key is absent.
- Never send questions, reference answers, attachment content, full prompts, model output text, tool arguments, or tool results.
- Never write Langfuse credentials to source, reports, audit logs, or trace metadata.
- Langfuse failures must not change predictions, retries, scoring, or exit status.
- Preserve all existing worktree changes and the current provider-neutral architecture.

---

### Task 1: Configuration and observability adapter

**Files:**
- Create: `src/gaia_agent/observability.py`
- Modify: `src/gaia_agent/config.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Test: `tests/test_observability.py`

**Interfaces:**
- `ObservabilityConfig.from_env() -> ObservabilityConfig` reads enablement, keys, base URL, and project name.
- `Observability.start_task(task_id: str, *, level: int, model: str, reasoning_effort: str, run_index: int, evaluation_mode: str) -> TaskTrace` returns a no-op or Langfuse-backed task trace.
- `TaskTrace.span(name: str, metadata: dict[str, object]) -> Span` and `TaskTrace.generation(name: str, metadata: dict[str, object]) -> Span` create safe child observations.
- `Span.end(*, status: str, metadata: dict[str, object] | None = None, usage: Usage | None = None)` closes an observation.
- `Observability.flush()` is best-effort and never raises.

- [ ] **Step 1: Write failing tests** for disabled no-op tracing, key-based enablement, safe metadata filtering, usage serialization, and swallowed flush errors.
- [ ] **Step 2: Run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_observability.py -q` and confirm the tests fail because the adapter is absent.**
- [ ] **Step 3: Add the `langfuse` dependency, implement configuration, no-op objects, and the Langfuse wrapper with guarded imports and best-effort calls.**
- [ ] **Step 4: Add Langfuse variables to `.env.example` without values and run the focused tests again.**

### Task 2: Task and stage lifecycle instrumentation

**Files:**
- Modify: `src/gaia_agent/runner.py`
- Modify: `src/gaia_agent/orchestrator.py`
- Test: `tests/test_observability.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Runner owns one `Observability` instance and flushes it after each task and at run completion.
- Orchestrator receives the adapter or task trace and creates stage spans without passing task content.

- [ ] **Step 1: Write failing tests** asserting task traces include only stable task metadata and stage spans close on success and terminal failure.
- [ ] **Step 2: Run the focused tests and confirm they fail before instrumentation.**
- [ ] **Step 3: Wrap `_run_task`, planner/act/verify/submit, and terminal-failure paths with spans; attach only status, counts, names, and sanitized exception types.**
- [ ] **Step 4: Run focused runner/orchestrator tests and confirm prediction values and usage are unchanged.**

### Task 3: Provider and MCP call instrumentation

**Files:**
- Modify: `src/gaia_agent/providers/openai.py`
- Modify: `src/gaia_agent/providers/deepseek.py`
- Modify: `src/gaia_agent/mcp_client.py`
- Modify: `src/gaia_agent/mcp_server.py`
- Test: `tests/test_openai_provider.py`
- Test: `tests/test_deepseek_provider.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Provider calls use the active trace context and record model, reasoning effort, parsed/refusal status, and `Usage` only.
- MCP client calls create a span with tool name, call number, success/failure, and safe result/status keys only.

- [ ] **Step 1: Write failing tests** for provider usage spans, parsed/refusal flags, MCP tool success/error spans, and no payload leakage.
- [ ] **Step 2: Run the focused provider/MCP tests and confirm the tests fail.**
- [ ] **Step 3: Add guarded instrumentation around provider and MCP boundaries, preserving current exception and usage behavior.**
- [ ] **Step 4: Run focused tests and verify trace failures do not affect provider results.**

### Task 4: Documentation and full verification

**Files:**
- Modify: `README.md`
- Test: full suite

- [ ] **Step 1: Document environment variables, enablement behavior, dashboard trace shape, and privacy exclusions.**
- [ ] **Step 2: Run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`.**
- [ ] **Step 3: Run `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`.**
- [ ] **Step 4: Run `UV_CACHE_DIR=/tmp/uv-cache uv build`.**
