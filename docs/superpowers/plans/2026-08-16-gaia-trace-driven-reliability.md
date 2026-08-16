# GAIA Trace-Driven Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Luna/max GAIA workflow survive transient provider/tool failures, reuse completed research, repair semantic protocol errors, avoid impossible routes, reduce context waste, and emit complete Langfuse task outcomes.

**Architecture:** Preserve the current LangGraph Planner → Specialist → Critic → Submit graph and GAIA scorer. Add typed reliability helpers at the provider, tool, stage, plan, persistence, and telemetry boundaries; keep private research data outside public run artifacts and make every recovery bounded and observable.

**Tech Stack:** Python 3.11+, asyncio, OpenAI Responses API, LangGraph, MCP v2, httpx, Pydantic v2, Langfuse v4, pytest/pytest-asyncio, Ruff.

## Global Constraints

- Every Planner, Specialist, Critic, synthesis, and repair request uses `gpt-5.6-luna` with `reasoning_effort=max` in production.
- Do not change GAIA answer normalization, exact-match scoring, reference-answer handling, or selected-task sampling.
- Do not publish prompts, questions, candidate text, reference answers, hidden reasoning, attachment contents, webpage bodies, full tool arguments/results, or credentials.
- Attachment registration, path containment, symlink, MIME, and size checks remain fail-closed.
- Every retry, repair, fallback, cache reuse, route rewrite, and early stop is bounded and recorded with safe metadata.
- No paid GAIA evaluation runs during implementation or verification; a paid run requires separate user authorization.
- Existing untracked `.DS_Store` files are not modified, staged, or committed.

---

## File map

- Create `src/gaia_agent/reliability.py`: provider retry policy, typed transient classification, safe tool-error envelope, backoff helper.
- Create `src/gaia_agent/research_notebook.py`: private atomic per-task tool-result checkpoint and bounded evidence-card selection.
- Create `src/gaia_agent/search.py`: OpenAI/Serper/SerpAPI search adapters, normalization, retries, and circuit breaker.
- Modify `src/gaia_agent/providers/base.py`: typed safe recoverable tool result and provider retry metadata.
- Modify `src/gaia_agent/providers/openai.py`: retry every Responses call in place, preserve successful tool events, and expose attempt metadata.
- Modify `src/gaia_agent/agents.py`: semantic repair, capability-aware plan normalization, and bounded specialist context.
- Modify `src/gaia_agent/orchestrator.py`: notebook integration, early stop, soft optional-step failure, and richer counters.
- Modify `src/gaia_agent/mcp_server.py`: search-provider chain and typed webpage errors.
- Modify `src/gaia_agent/mcp_client.py`: propagate configured search credentials and safe error categories.
- Modify `src/gaia_agent/observability.py`: task outcome, retry/tool counters, cost details, and visible flush status.
- Modify `src/gaia_agent/runner.py`: task-scoped root trace through scoring and private notebook composition.
- Modify `src/gaia_agent/config.py`, `.env.example`, `.gitignore`, and `README.md`: bounded reliability configuration and operational documentation.
- Add focused tests to existing suites plus `tests/test_reliability.py`, `tests/test_research_notebook.py`, and `tests/test_search.py`.

---

### Task 1: Typed Provider Retry at the Failed Continuation

**Files:**
- Create: `src/gaia_agent/reliability.py`
- Modify: `src/gaia_agent/providers/openai.py`
- Modify: `src/gaia_agent/observability.py`
- Test: `tests/test_reliability.py`
- Test: `tests/test_openai_provider.py`

**Interfaces:**
- Produces: `RetryPolicy(max_attempts: int = 4, base_delay_seconds: float = 0.5, max_delay_seconds: float = 8.0)`.
- Produces: `is_transient_provider_error(error: BaseException) -> bool`.
- Produces: `retry_delay(policy: RetryPolicy, failed_attempt: int, jitter: Callable[[], float]) -> float`.
- Changes: `OpenAIProvider(..., retry_policy: RetryPolicy | None = None, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, jitter: Callable[[], float] = random.random)`.

- [ ] **Step 1: Write failing retry classification and backoff tests**

```python
@pytest.mark.parametrize(
    "error",
    [APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
     APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")),
     RateLimitError("limited", response=_response(429), body=None),
     InternalServerError("server", response=_response(500), body=None)],
)
def test_transient_provider_errors_are_classified(error: Exception) -> None:
    assert is_transient_provider_error(error)

def test_retry_delay_is_bounded() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=4.0)
    assert retry_delay(policy, 1, lambda: 0.0) == 1.0
    assert retry_delay(policy, 2, lambda: 0.0) == 2.0
    assert retry_delay(policy, 3, lambda: 1.0) <= 4.0
```

- [ ] **Step 2: Run the new tests and verify they fail because the reliability module does not exist**

Run: `uv run pytest tests/test_reliability.py -q`

- [ ] **Step 3: Implement frozen retry policy and MRO-safe transient classification**

```python
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0

def is_transient_provider_error(error: BaseException) -> bool:
    return any(cls.__name__ in TRANSIENT_PROVIDER_ERROR_NAMES for cls in type(error).__mro__)

def retry_delay(policy: RetryPolicy, failed_attempt: int, jitter: Callable[[], float]) -> float:
    exponential = policy.base_delay_seconds * (2 ** (failed_attempt - 1))
    return min(policy.max_delay_seconds, exponential * (1.0 + max(0.0, min(1.0, jitter())) * 0.25))
```

- [ ] **Step 4: Write failing provider tests for initial and continuation `APIConnectionError`**

The continuation test must return a function call, execute the fake tool once, fail the first
continuation request, then succeed. Assert three API calls, one tool execution, one final parsed
response, and sleep delays `[0.5]`.

- [ ] **Step 5: Replace `_create_with_retry` with four-attempt in-place retry**

```python
async def _create_with_retry(self, **kwargs: Any) -> Any:
    for attempt in range(1, self._retry_policy.max_attempts + 1):
        try:
            return await self._client.responses.create(**kwargs)
        except Exception as error:
            if not is_transient_provider_error(error) or attempt == self._retry_policy.max_attempts:
                raise
            self._record_retry(error, attempt)
            await self._sleep(retry_delay(self._retry_policy, attempt, self._jitter))
    raise AssertionError("bounded provider retry did not return or raise")
```

- [ ] **Step 6: Record safe generation retry metadata without exception messages**

Add `provider_attempt_count`, `provider_retry_count`, and `retry_category` to the observability
allowlist and generation summary. Use class/category names only.

- [ ] **Step 7: Run focused provider tests**

Run: `uv run pytest tests/test_reliability.py tests/test_openai_provider.py -q`

- [ ] **Step 8: Commit**

```bash
git add src/gaia_agent/reliability.py src/gaia_agent/providers/openai.py src/gaia_agent/observability.py tests/test_reliability.py tests/test_openai_provider.py
git commit -m "fix: retry transient provider continuations"
```

---

### Task 2: Semantic Stage Repair and Partial-Candidate Survival

**Files:**
- Modify: `src/gaia_agent/agents.py`
- Modify: `src/gaia_agent/orchestrator.py`
- Test: `tests/test_agents.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `StageValidationCode` enum with bounded public values.
- Produces: `StageProtocolError(code: StageValidationCode, usage: Usage, repairable: bool = True)`.
- Changes: `_complete_with_local_repair(..., parser)` retries both unparsed output and repairable semantic validation once.
- Changes: `_GraphState` carries `soft_step_failures: int` and routes a prior valid candidate/evidence to verification.

- [ ] **Step 1: Write failing specialist tests for parsed-but-invalid repair**

Use a first parsed `AgentResult` with the wrong specialist and a second tool-free valid result. Assert
two provider calls, the second request has no tools, usage is aggregated, and
`last_repair_attempted` is true.

- [ ] **Step 2: Write failing tests proving security/policy errors are not repaired**

An attachment policy error and unknown evidence provenance must make exactly one provider call and
must not enter repair.

- [ ] **Step 3: Add validation-code mapping and semantic repair request**

```python
class StageValidationCode(StrEnum):
    ROLE_MISMATCH = "role_mismatch"
    EVIDENCE_SHAPE = "evidence_shape"
    SOURCE_URL_MISSING = "source_url_missing"
    CANDIDATE_FORMAT = "candidate_format"
    CHECKED_EVIDENCE = "checked_evidence"
    SCHEMA = "schema"

def _semantic_repair_request(request: ModelRequest, error: StageProtocolError) -> ModelRequest:
    return request.model_copy(update={
        "instructions": f"{request.instructions}\n\nRepair code: {error.code.value}. Return one valid object only.",
        "allowed_tools": (),
        "max_tool_calls": 0,
    })
```

The repair input includes only the already-sanitized parsed object, not raw hidden reasoning or
tool bodies.

- [ ] **Step 4: Make Planner, Specialist, and Critic parsers raise typed protocol errors**

Map role/evidence/candidate/checked-ID failures to `StageProtocolError`; retain attachment and
provenance policy errors as non-repairable exceptions.

- [ ] **Step 5: Write failing orchestrator test for optional enrichment failure**

Create a two-step plan where step one yields candidate plus source evidence and step two raises a
repair-exhausted `StageProtocolError`. Assert `soft_step_failure` audit, Critic execution, and final
submission of the earlier candidate when approved.

- [ ] **Step 6: Implement bounded soft failure in `_graph_act`**

Catch only exhausted repairable `StageProtocolError`. Continue to verify only when
`state.candidate is not None and state.evidence`; otherwise re-raise. Record step number and error
code, never the message or candidate.

- [ ] **Step 7: Run focused stage tests**

Run: `uv run pytest tests/test_agents.py tests/test_orchestrator.py -q`

- [ ] **Step 8: Commit**

```bash
git add src/gaia_agent/agents.py src/gaia_agent/orchestrator.py tests/test_agents.py tests/test_orchestrator.py
git commit -m "fix: repair semantic stage results"
```

---

### Task 3: Capability-Aware Short Plans and Early Stop

**Files:**
- Modify: `src/gaia_agent/agents.py`
- Modify: `src/gaia_agent/orchestrator.py`
- Test: `tests/test_agents.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `normalize_plan(plan: AgentPlan, task: GaiaTask, provider: ModelProvider, max_initial_steps: int = 2) -> AgentPlan`.
- Produces: `_ready_for_critic(state: _TaskState) -> bool`.

- [ ] **Step 1: Write failing no-attachment and non-image routing tests**

Assert Visual/File steps become Lookup or Computation when `attachment_path=None`; Visual becomes
File for a registered PDF; no normalized plan contains an ineligible specialist.

- [ ] **Step 2: Write failing duplicate and plan-length tests**

Assert adjacent duplicate specialists are coalesced and the initial normalized plan has at most two
steps while keeping the original order of the first distinct routes.

- [ ] **Step 3: Add task capabilities to Planner context and normalize every plan**

```python
capabilities = {
    "attachment_present": task.attachment_path is not None,
    "attachment_media_type": attachment["media_type"] if attachment else None,
    "allowed_specialists": [item.value for item in _allowed_specialists(task, self.provider)],
}
```

Apply `normalize_plan` to both model plans and deterministic fallback plans.

- [ ] **Step 4: Write failing graph early-stop test**

Use a three-step plan whose first step produces candidate and source evidence. Assert only the first
specialist executes before Critic and Submit.

- [ ] **Step 5: Route candidate plus evidence to verification**

Change `_route_after_act` to return `verify` before the next planned step when `_ready_for_critic`
is true. Emit `plan_early_stop` with completed/remaining counts.

- [ ] **Step 6: Run focused plan tests**

Run: `uv run pytest tests/test_agents.py tests/test_orchestrator.py -q`

- [ ] **Step 7: Commit**

```bash
git add src/gaia_agent/agents.py src/gaia_agent/orchestrator.py tests/test_agents.py tests/test_orchestrator.py
git commit -m "feat: normalize and shorten GAIA plans"
```

---

### Task 4: Private Persistent Research Notebook

**Files:**
- Create: `src/gaia_agent/research_notebook.py`
- Modify: `src/gaia_agent/orchestrator.py`
- Modify: `src/gaia_agent/runner.py`
- Modify: `.gitignore`
- Test: `tests/test_research_notebook.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- Produces: `NotebookEntry(signature: str, tool: str, result: dict[str, Any], created_at: datetime)`.
- Produces: `ResearchNotebook(root: Path, task_id: str, max_entry_bytes: int = 65_536)`.
- Produces: `ResearchNotebook.lookup(tool: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None`.
- Produces: `ResearchNotebook.record(tool: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> None`.
- Changes: `_BudgetedToolClient(..., notebook: ResearchNotebook | None = None)`.

- [ ] **Step 1: Write failing atomic persistence and permission tests**

```python
def test_notebook_persists_private_bounded_result(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1", max_entry_bytes=1024)
    notebook.record("web_search", {"query": "q"}, {"answer": "a", "sources": []})
    assert notebook.lookup("web_search", {"query": "q"})["answer"] == "a"
    assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
    assert stat.S_IMODE(next((tmp_path / "private").rglob("*.json")).stat().st_mode) == 0o600
```

Also test corruption as cache miss, task-ID containment, argument-order normalization, oversized
entry rejection, and deep-copy returns.

- [ ] **Step 2: Implement canonical signature and atomic private writes**

Use SHA-256 over canonical JSON `{tool, arguments}`. Write a temporary file in the same directory,
`fsync`, `chmod(0o600)`, and `os.replace`; create directories with `0o700`. Never log arguments or
results.

- [ ] **Step 3: Write failing budget-client cross-scope reuse test**

Populate a notebook, create a fresh `_BudgetedToolClient`, and assert an identical call returns the
persisted deep copy without invoking the MCP client or incrementing paid-call count. Record
`tool_call_restored` separately from same-scope loop detection.

- [ ] **Step 4: Integrate notebook after successful MCP calls**

Lookup the notebook after global budget validation and before the MCP call. Record only successful
results. Keep failed calls uncached. Maintain separate counters for logical calls, executed calls,
and restored calls.

- [ ] **Step 5: Compose a sibling private root in Runner**

```python
def _private_run_root(output_dir: Path) -> Path:
    absolute = output_dir.absolute()
    return absolute.parent / f".{absolute.name}.private"
```

Create one task notebook in `orchestrator_factory(task)`. Add `.*.private/` to `.gitignore`. Do not
add private files to `RunStore`, report generation, or `verify_run_directory`.

- [ ] **Step 6: Add public-artifact exclusion tests**

Run a fake task with private notebook content containing a sentinel URL/text. Assert the sentinel is
absent from every file under the public output directory and public verification still passes.

- [ ] **Step 7: Run notebook and integration tests**

Run: `uv run pytest tests/test_research_notebook.py tests/test_orchestrator.py tests/test_cli_e2e.py -q`

- [ ] **Step 8: Commit**

```bash
git add .gitignore src/gaia_agent/research_notebook.py src/gaia_agent/orchestrator.py src/gaia_agent/runner.py tests/test_research_notebook.py tests/test_orchestrator.py tests/test_cli_e2e.py
git commit -m "feat: persist private GAIA research checkpoints"
```

---

### Task 5: Structured Tool Errors and Search Provider Fallback

**Files:**
- Create: `src/gaia_agent/search.py`
- Modify: `src/gaia_agent/providers/base.py`
- Modify: `src/gaia_agent/providers/openai.py`
- Modify: `src/gaia_agent/mcp_server.py`
- Modify: `src/gaia_agent/mcp_client.py`
- Modify: `.env.example`
- Test: `tests/test_search.py`
- Test: `tests/test_mcp.py`
- Test: `tests/test_openai_provider.py`

**Interfaces:**
- Produces: `SafeToolFailure(tool: str, category: str, retryable: bool, provider: str | None, attempt_count: int, http_status: int | None)`.
- Produces: `SearchProvider.search(query: str) -> dict[str, Any]` protocol.
- Produces: `SearchProviderChain(providers: Sequence[SearchProvider], retry_policy: RetryPolicy, clock: Callable[[], float])`.
- Normalized search output remains `{"answer": str, "sources": [{"url": str, "title": str | None}]}`.

- [ ] **Step 1: Write failing safe-envelope tests**

Assert a recoverable tool exception reaches Luna as JSON with exactly `error`, `category`,
`retryable`, `provider`, `attempt_count`, and optional `http_status`; assert raw exception messages,
keys, URLs, and bodies are absent.

- [ ] **Step 2: Replace generic `tool_failed` output**

```python
except ToolCallError as error:
    failure = SafeToolFailure.from_exception(call.name, error)
    result = {"error": "tool_failed", **failure.model_dump(exclude_none=True)}
```

- [ ] **Step 3: Write failing Serper and SerpAPI normalization tests**

Mock httpx responses and assert both adapters produce the same bounded answer/source schema,
respect timeouts, reject malformed payloads, and never include API keys in exceptions.

- [ ] **Step 4: Implement provider adapters with injected `httpx.AsyncClient`**

Serper calls `POST https://google.serper.dev/search` with `X-API-KEY`; SerpAPI calls
`GET https://serpapi.com/search.json` with the key in request params. Extract answer boxes and
organic results, then reuse `_bounded_web_search_result` at the MCP boundary.

- [ ] **Step 5: Write failing fallback and circuit-breaker tests**

Assert provider order OpenAI → Serper → SerpAPI, two consecutive transient failures open the
circuit, an open provider is skipped, cooldown permits a probe, and success closes the circuit.

- [ ] **Step 6: Implement `SearchProviderChain`**

Try only configured providers, retry transient failures in place, fall through on bounded
retry exhaustion, and raise one safe `SearchUnavailableError` only after every configured provider
fails. Do not concatenate secret-bearing messages.

- [ ] **Step 7: Propagate optional keys to the MCP subprocess**

Add only `SERPER_API_KEY` and `SERPAPI_API_KEY` to `_minimal_server_env`; add them to the MCP server
redaction secret set. Update `.env.example` with empty placeholders.

- [ ] **Step 8: Categorize webpage failures**

Normalize timeout, HTTP status, policy rejection, parse failure, and empty content to safe categories.
Retry only transient transport/status failures and return search snippets to the model when direct
page access fails.

- [ ] **Step 9: Run focused tool tests**

Run: `uv run pytest tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py -q`

- [ ] **Step 10: Commit**

```bash
git add .env.example src/gaia_agent/search.py src/gaia_agent/providers/base.py src/gaia_agent/providers/openai.py src/gaia_agent/mcp_server.py src/gaia_agent/mcp_client.py tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py
git commit -m "feat: add resilient search provider fallback"
```

---

### Task 6: Bound Context and Reuse Evidence Cards

**Files:**
- Modify: `src/gaia_agent/research_notebook.py`
- Modify: `src/gaia_agent/agents.py`
- Modify: `src/gaia_agent/orchestrator.py`
- Test: `tests/test_research_notebook.py`
- Test: `tests/test_agents.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `select_evidence_cards(entries: Sequence[NotebookEntry], *, max_items: int = 12, max_json_bytes: int = 24_576) -> tuple[dict[str, Any], ...]`.
- Produces: `context_size_bytes(input_items: Sequence[Mapping[str, Any]]) -> int`.

- [ ] **Step 1: Write failing evidence selection tests**

Assert URL/content-hash duplicates collapse, source-bearing cards outrank empty/error cards, total
serialized size stays within 24 KiB, and original order breaks equal-score ties deterministically.

- [ ] **Step 2: Implement bounded evidence-card selection**

Keep source title/domain, relevant excerpt, tool kind, and opaque entry hash. Bound each text field,
drop full webpage bodies, and never truncate JSON into invalid syntax.

- [ ] **Step 3: Write failing Specialist context tests**

Build oversized prior evidence and notebook cards, then assert `_specialist_input` contains the
highest-value bounded cards, the question once, no duplicate source bodies, and serialized size
under the configured limit.

- [ ] **Step 4: Add budget-pressure and forced-synthesis routing**

When context reaches 70 percent of the per-stage limit, audit `context_budget_pressure` with byte
and item counts and stop issuing new enrichment steps. Keep Luna/max and route to Critic/synthesis.

- [ ] **Step 5: Run context tests**

Run: `uv run pytest tests/test_research_notebook.py tests/test_agents.py tests/test_orchestrator.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/gaia_agent/research_notebook.py src/gaia_agent/agents.py src/gaia_agent/orchestrator.py tests/test_research_notebook.py tests/test_agents.py tests/test_orchestrator.py
git commit -m "perf: bound GAIA research context"
```

---

### Task 7: Complete Task-Outcome Langfuse Traces

**Files:**
- Modify: `src/gaia_agent/observability.py`
- Modify: `src/gaia_agent/runner.py`
- Modify: `src/gaia_agent/config.py`
- Test: `tests/test_observability.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- Changes: one `gaia-task` root spans every vote, local scoring, persistence decision, and terminal outcome.
- Produces: `TaskTrace.finish(outcome: TaskTraceOutcome) -> None`.
- Produces: `Observability.flush() -> bool`.
- Produces: optional `LANGFUSE_INPUT_COST_PER_MILLION` and `LANGFUSE_OUTPUT_COST_PER_MILLION` decimal configuration.

- [ ] **Step 1: Write failing root-trace lifecycle test**

Assert root trace starts before `_run_task`, remains open while `question_scorer` runs, and ends with
safe output fields `answer_present`, `evidence_count`, `critic_verdict`, `error_category`, `correct`,
`provider_retry_count`, `tool_success_count`, `tool_failure_count`, and `checkpoint_restored`.

- [ ] **Step 2: Move root ownership from vote to selected task**

Start the root in `EvaluationRunner._run` immediately before `_run_task`; create safe `vote` child
spans for each `run_index`; score the selected prediction; then finish the root in `finally` so every
path has one terminal record.

- [ ] **Step 3: Add safe outcome models and allowlists**

Use a frozen `TaskTraceOutcome` with bounded booleans/counts/categories. Add only its exact keys to
safe output/metadata allowlists. Do not include answer text or reference answer.

- [ ] **Step 4: Add per-stage usage, latency, retry, and tool counters**

Aggregate existing provider and MCP spans into explicit safe fields. Add `run_id` and attempt number
to root metadata. Preserve existing reasoning summaries.

- [ ] **Step 5: Write failing cost-detail and flush-warning tests**

With configured decimal rates, assert generation update receives bounded `cost_details`. Assert
successful `flush()` returns true and a fake SDK exception returns false plus one warning that
contains no credential or payload.

- [ ] **Step 6: Implement model-rate mapping and visible startup/flush health**

Parse non-negative finite decimal rates from the environment, compute input/output cost from usage,
and omit cost details when rates are absent. CLI startup prints only `langfuse=enabled|disabled`; run
completion prints `langfuse_flush=ok|warning` without keys or endpoint details.

- [ ] **Step 7: Run observability tests**

Run: `uv run pytest tests/test_observability.py tests/test_cli_e2e.py -q`

- [ ] **Step 8: Commit**

```bash
git add src/gaia_agent/observability.py src/gaia_agent/runner.py src/gaia_agent/config.py tests/test_observability.py tests/test_cli_e2e.py
git commit -m "feat: record complete GAIA task outcomes"
```

---

### Task 8: Configuration, Documentation, Regression, and Offline Trace Simulation

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `tests/test_config.py`
- Modify: `tests/test_cli_e2e.py`
- Create: `tests/test_trace_reliability_e2e.py`

**Interfaces:**
- Documents all reliability environment variables and Luna/max invocation.
- Produces an offline ten-task-equivalent deterministic simulation covering the observed category mix without network or paid API calls.

- [ ] **Step 1: Write failing config tests**

Assert retry attempts, retry delays, private notebook size, context limit, search cooldown, and cost
rates accept valid bounded values and reject negative, non-finite, excessive, or secret-printing
inputs. Assert production test configuration remains `model=gpt-5.6-luna` and
`reasoning_effort=max` when supplied.

- [ ] **Step 2: Add bounded configuration fields and environment parsing**

Add explicit `GAIA_PROVIDER_MAX_ATTEMPTS`, `GAIA_PROVIDER_BACKOFF_BASE`,
`GAIA_PRIVATE_NOTEBOOK_MAX_BYTES`, `GAIA_STAGE_CONTEXT_MAX_BYTES`,
`GAIA_SEARCH_CIRCUIT_COOLDOWN`, `LANGFUSE_INPUT_COST_PER_MILLION`, and
`LANGFUSE_OUTPUT_COST_PER_MILLION`. Persist non-secret run-defining values in the manifest so resume
cannot silently change behavior.

- [ ] **Step 3: Build the offline trace reliability simulation**

Script ten fake tasks: six inject one continuation `APIConnectionError`, two inject one repairable
semantic result, and two return impossible visual plans without attachments. Assert all provider
errors recover, semantic payloads repair, routes normalize, ten root traces close, at least eight
reach Critic, and no public artifact contains private sentinel content.

- [ ] **Step 4: Update README and `.env.example`**

Document the provider chain, private sibling checkpoint directory, structured recovery order,
Langfuse outcome fields, cost-rate configuration, and this production command without embedding
credentials:

```console
GAIA_MODEL=gpt-5.6-luna GAIA_REASONING_EFFORT=max \
uv run gaia-agent evaluate --sample-size 15 --seed 42 --output-dir results/gaia-luna-max
```

- [ ] **Step 5: Run focused config and offline E2E tests**

Run: `uv run pytest tests/test_config.py tests/test_trace_reliability_e2e.py tests/test_cli_e2e.py -q`

- [ ] **Step 6: Run complete verification**

Run:

```console
uv run pytest -q
uv run ruff check .
uv build
```

Expected: all tests pass, Ruff reports no violations, and both sdist and wheel build successfully.

- [ ] **Step 7: Verify worktree privacy and diff scope**

Run:

```console
git status --short
git diff --check
git diff --name-only
```

Confirm no `.env`, private notebook directory, result directory, credential, `.DS_Store`, or paid-run
artifact is staged.

- [ ] **Step 8: Commit**

```bash
git add README.md .env.example src/gaia_agent/config.py tests/test_config.py tests/test_cli_e2e.py tests/test_trace_reliability_e2e.py
git commit -m "test: verify trace-driven GAIA reliability"
```

- [ ] **Step 9: Request code review and address only evidence-backed findings**

Use the `superpowers:requesting-code-review` workflow against the implementation-base commit, then
rerun the complete verification commands after any accepted change.

---

## Paid evaluation gate

After all eight tasks pass and the user separately approves spending API credits:

1. verify Langfuse configuration and flush with a no-model connectivity check;
2. run a three-task Luna/max smoke evaluation in a new output directory;
3. inspect all three root traces and confirm scorer fields;
4. only with another explicit approval, run the requested larger GAIA subset;
5. compare completion rate, failure categories, median tokens, tool failure rate, Critic reach rate,
   and accuracy against the August 11 run.
