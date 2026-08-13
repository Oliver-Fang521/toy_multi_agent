# LangGraph Orchestrator Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing GAIA Plan -> Act -> Verify -> Correct -> Submit workflow through a compiled LangGraph `StateGraph` without changing agent, tool, scoring, persistence, or Langfuse behavior.

**Architecture:** Keep `Orchestrator.run()` as the timeout, resource, budget, and terminal-error boundary. Replace only `_execute()` with an async invocation of a compiled graph whose nodes delegate to the existing planner, `_act()`, `_verify()`, `_correct_once()`, and `_submit()` methods.

**Tech Stack:** Python 3.11+, LangGraph 1.x, Pydantic 2, pytest, pytest-asyncio, Ruff, Hatchling/uv.

## Global Constraints

- Use `langgraph>=1,<2` and the documented `StateGraph`, `START`, and `END` API.
- Preserve the public `Orchestrator.run(task) -> TaskPrediction` contract.
- Preserve existing GAIA answer formatting, quality policy, terminal categories, audit events, counters, and Langfuse spans.
- Do not add checkpointing, parallel execution, prompt changes, or new trace payloads.
- Work only in the existing `codex/gaia-multi-agent` worktree and preserve unrelated dirty changes.

---

### Task 1: Require LangGraph execution at the Orchestrator boundary

**Files:**
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: existing `_orchestrator(...) -> Orchestrator` test helper.
- Produces: a behavioral test that fails if `_execute()` bypasses the compiled graph.

- [ ] **Step 1: Write the failing behavioral test**

Add a test that replaces `orchestrator._graph` with a recording fake exposing
`async ainvoke(input, config)`. The fake returns a literal prediction and the
test asserts `run()` returns it, receives the sanitized task, existing
`_TaskState`, budgeted tool client, `next_step_index == 0`, and a recursion
limit greater than the configured `max_steps`.

- [ ] **Step 2: Run the new test to verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py::test_orchestrator_executes_compiled_graph -q
```

Expected: FAIL because `Orchestrator` has no `_graph` and `_execute()` still
runs the hand-written loop.

- [ ] **Step 3: Do not change production code yet**

Confirm the failure names the missing graph execution boundary rather than a
fixture or syntax error.

---

### Task 2: Compile and execute the StateGraph

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/gaia_agent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `_TaskState`, `_BudgetedToolClient`, `GaiaTask`, `CriticDecision`, and `TaskPrediction`.
- Produces: `_GraphState`, `Orchestrator._build_graph()`, graph node methods, conditional routers, and `_execute()` delegating to `_graph.ainvoke(...)`.

- [ ] **Step 1: Add the runtime dependency**

Add this line in `[project].dependencies`:

```toml
"langgraph>=1,<2",
```

Regenerate the lockfile with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv lock
```

- [ ] **Step 2: Define graph state**

Add a private `TypedDict` with these fields:

```python
class _GraphState(TypedDict, total=False):
    task: Required[GaiaTask]
    runtime: Required[_TaskState]
    tools: Required[_BudgetedToolClient]
    next_step_index: int
    decision: CriticDecision
    prediction: TaskPrediction
```

- [ ] **Step 3: Compile the graph in `Orchestrator.__init__`**

Build this topology with named nodes and conditional edges:

```text
START -> plan -> act -> act|verify
verify -> correct|submit|reject
correct -> verify
submit -> END
reject -> END (the node raises the preserved rejection exception)
```

- [ ] **Step 4: Implement thin graph nodes**

Move the current planner block into `_graph_plan()`. Make `_graph_act()` call
`_act()` for exactly one indexed plan step. Make `_graph_verify()`,
`_graph_correct()`, and `_graph_submit()` delegate to the existing methods.
Make `_graph_reject()` raise `_CorrectionUnavailable`.

- [ ] **Step 5: Replace `_execute()` with graph invocation**

Invoke the compiled graph asynchronously:

```python
result = await self._graph.ainvoke(
    {
        "task": task,
        "runtime": state,
        "tools": tools,
        "next_step_index": 0,
    },
    config={"recursion_limit": self._graph_recursion_limit()},
)
prediction = result.get("prediction")
if not isinstance(prediction, TaskPrediction):
    raise ValueError("graph completed without a prediction")
return prediction
```

- [ ] **Step 6: Run the RED test to verify GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py::test_orchestrator_executes_compiled_graph -q
```

Expected: PASS.

- [ ] **Step 7: Run the focused orchestrator suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py -q
```

Expected: all tests pass, proving success, correction, budgets, attachment
errors, malformed responses, and Langfuse summaries retain their behavior.

---

### Task 3: Verify dependency, code quality, and complete regression suite

**Files:**
- Review: `pyproject.toml`
- Review: `uv.lock`
- Review: `src/gaia_agent/orchestrator.py`
- Review: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: the completed LangGraph migration.
- Produces: fresh verification evidence for tests, lint, importability, and package build.

- [ ] **Step 1: Run Ruff checks**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests
```

- [ ] **Step 2: Run the full test suite**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

- [ ] **Step 3: Verify the package imports and graph compiles**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from langgraph.graph import StateGraph; from gaia_agent.orchestrator import Orchestrator; print(StateGraph.__name__, Orchestrator.__name__)"
```

- [ ] **Step 4: Build the package**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
```

- [ ] **Step 5: Review the final diff**

Confirm the diff contains only the dependency, graph orchestration, tests, and
the two migration documents. Do not stage, commit, or alter unrelated existing
changes in this shared worktree.
