# GAIA Stage Recovery and Search Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent hosted-search timeouts from bypassing fallback or causing an expensive whole-task replay, while preserving completed research and reserving resources for Critic verification.

**Architecture:** Normalize timeout errors at the search-provider boundary, enforce deadline-aware stage and tool budgets in the orchestrator, and store authenticated private graph checkpoints beside the existing Research Notebook. Resume from the latest stable graph cursor, repair missing evidence links from notebook cards without tools, and expose only bounded operational metadata to Langfuse.

**Tech Stack:** Python 3.12, asyncio, Pydantic v2, LangGraph, OpenAI Responses API, MCP over stdio, httpx, Langfuse SDK, pytest/pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-24-gaia-stage-recovery-design.md`

## Global Constraints

- Production evaluation remains `model=gpt-5.6-luna` with `reasoning_effort=max`.
- GAIA scoring and final-answer normalization are unchanged.
- The global task deadline remains a hard upper bound; recovery does not create a second full deadline.
- Private candidate, evidence, URLs, webpage content, prompts, and credentials never enter public results or Langfuse summaries.
- Stage checkpoints use bounded reads, authenticated content, atomic replacement, and owner-only permissions.
- Serper and SerpAPI are optional and are enabled only by their existing environment keys.
- Existing `benchmark` and `strict` quality policies retain their submission rules.

---

### Task 1: Normalize hosted-search timeouts and make fallback deadline-aware

**Files:**
- Modify: `src/gaia_agent/search.py:217-309`
- Modify: `src/gaia_agent/mcp_server.py:655-695`
- Test: `tests/test_search.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `RetryPolicy`, `SearchProviderError`, and the existing ordered provider list.
- Produces: `SearchDeadlineExceeded`, an `operation_timeout_seconds` argument on
  `SearchProviderChain.__init__`, and a
  bounded hosted-provider timeout that leaves time for configured fallback providers.

- [ ] **Step 1: Write a failing regression test for built-in timeout fallback**

```python
@pytest.mark.asyncio
async def test_builtin_timeout_is_retryable_and_falls_back_to_serper() -> None:
    class TimedOutOpenAI:
        provider = "openai"
        async def search(self, query: str) -> dict[str, Any]:
            raise TimeoutError

    class HealthySerper:
        provider = "serper"
        async def search(self, query: str) -> dict[str, Any]:
            return {"answer": "fallback", "sources": []}

    chain = SearchProviderChain(
        [OpenAISearchProvider(TimedOutOpenAI()), HealthySerper()],
        RetryPolicy(max_attempts=1),
        clock=lambda: 0.0,
    )
    result = await chain.search("question")
    assert result.provider == "serper"
    assert result.fallback_used is True
```

- [ ] **Step 2: Run the regression test and confirm RED**

Run: `uv run pytest tests/test_search.py::test_builtin_timeout_is_retryable_and_falls_back_to_serper -q`

Expected: FAIL because built-in `TimeoutError` escapes `OpenAISearchProvider`.

- [ ] **Step 3: Implement the minimal timeout normalization**

```python
except TimeoutError:
    raise SearchProviderError(
        provider=self.provider,
        category="timeout",
        retryable=True,
    ) from None
```

- [ ] **Step 4: Add failing tests for remaining-time checks**

```python
@pytest.mark.asyncio
async def test_search_chain_does_not_retry_when_deadline_cannot_cover_backoff() -> None:
    provider = AlwaysTimeoutProvider()
    sleeps: list[float] = []
    chain = SearchProviderChain(
        [provider],
        RetryPolicy(max_attempts=2, base_delay_seconds=0.5),
        clock=lambda: 10.0,
        operation_timeout_seconds=0.1,
        sleep=lambda delay: sleeps.append(delay),
    )
    with pytest.raises(SearchDeadlineExceeded):
        await chain.search("question")
    assert provider.calls == 1
    assert sleeps == []
```

Add a hosted-backend test asserting that its `asyncio.wait_for` timeout is the configured provider
timeout, which must be lower than the whole search-operation timeout whenever a fallback provider is
configured.

- [ ] **Step 5: Run the deadline tests and confirm RED**

Run: `uv run pytest tests/test_search.py tests/test_mcp.py -k 'deadline or builtin_timeout' -q`

Expected: FAIL because no deadline callback or dynamic request timeout exists.

- [ ] **Step 6: Add deadline-aware search interfaces**

```python
class SearchDeadlineExceeded(RuntimeError):
    pass

class SearchProviderChain:
    def __init__(
        self,
        providers: Sequence[SearchProvider],
        retry_policy: RetryPolicy,
        clock: Callable[[], float],
        *,
        operation_timeout_seconds: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._operation_timeout_seconds = operation_timeout_seconds

    def _can_wait(self, deadline: float | None, delay: float) -> bool:
        return deadline is None or deadline - self._clock() > delay
```

At the start of `search`, compute `deadline = clock() + operation_timeout_seconds`. Before sleep,
retry, and fallback, fail with `SearchDeadlineExceeded` if there is no usable time.
The MCP server creates one operation deadline for each `web_search` call. The parent
`_BudgetedToolClient` independently wraps the complete MCP call with the active stage's remaining
time in Task 3. This two-level bound works across the stdio process boundary without exposing a
deadline argument to the model's tool schema.

- [ ] **Step 7: Run search and MCP suites**

Run: `uv run pytest tests/test_search.py tests/test_mcp.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/gaia_agent/search.py src/gaia_agent/mcp_server.py tests/test_search.py tests/test_mcp.py
git commit -m "fix: normalize hosted search timeouts"
```

### Task 2: Add behavior-defining stage budget configuration

**Files:**
- Modify: `src/gaia_agent/config.py`
- Modify: `src/gaia_agent/cli.py`
- Modify: `src/gaia_agent/mcp_client.py:398-430`
- Modify: `src/gaia_agent/mcp_server.py:1148-1198`
- Modify: `src/gaia_agent/runner.py:662-704`
- Modify: `src/gaia_agent/persistence.py:190-260`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_config.py`
- Test: `tests/test_cli_e2e.py`
- Test: `tests/test_mcp.py`
- Test: `tests/test_persistence.py`

**Interfaces:**
- Produces `RunConfig.planner_timeout_seconds`, `specialist_timeout_seconds`, `critic_timeout_seconds`, `specialist_max_tool_calls`, and `critic_max_tool_calls`.
- Environment names are `GAIA_PLANNER_TIMEOUT_SECONDS`, `GAIA_SPECIALIST_TIMEOUT_SECONDS`, `GAIA_CRITIC_TIMEOUT_SECONDS`, `GAIA_SPECIALIST_MAX_TOOL_CALLS`, and `GAIA_CRITIC_MAX_TOOL_CALLS`.

- [ ] **Step 1: Write failing config validation and manifest tests**

```python
def test_stage_budgets_must_fit_global_limits(config_values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({
            **config_values,
            "timeout_seconds": 300,
            "planner_timeout_seconds": 30,
            "specialist_timeout_seconds": 240,
            "critic_timeout_seconds": 60,
        })

def test_manifest_persists_stage_budgets_for_safe_resume(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        planner_timeout_seconds=12,
        specialist_timeout_seconds=70,
        critic_timeout_seconds=18,
        specialist_max_tool_calls=7,
        critic_max_tool_calls=2,
    )
    manifest = _manifest_for_test_config(config)
    assert manifest.limits["planner_timeout_seconds"] == 12
    assert manifest.limits["specialist_timeout_seconds"] == 70
    assert manifest.limits["critic_timeout_seconds"] == 18
    assert manifest.limits["specialist_max_tool_calls"] == 7
    assert manifest.limits["critic_max_tool_calls"] == 2
```

Require each timeout to be positive and their sum to be `<= timeout_seconds`. Stage tool limits may
be zero for deliberately tiny test configurations, must otherwise be non-negative, and their sum
must be `<= max_tool_calls`.

- [ ] **Step 2: Run config tests and confirm RED**

Run: `uv run pytest tests/test_config.py tests/test_cli_e2e.py -k 'stage_budget or stage_budgets' -q`

Expected: FAIL because the fields do not exist.

- [ ] **Step 3: Implement fields, defaults, env, CLI, manifest, and MCP propagation**

Derive defaults in a `mode="before"` validator so existing short-timeout tests remain valid. Explicit
environment or CLI values always win:

```python
timeout = float(values.get("timeout_seconds", 300.0))
values.setdefault("planner_timeout_seconds", timeout * 0.10)
values.setdefault("specialist_timeout_seconds", timeout * 0.70)
values.setdefault("critic_timeout_seconds", timeout * 0.20)

calls = int(values.get("max_tool_calls", 24))
critic_calls = 0 if calls == 1 else max(1, calls // 4)
values.setdefault("specialist_max_tool_calls", calls - critic_calls)
values.setdefault("critic_max_tool_calls", critic_calls)
```

For the current production defaults this resolves to `30/210/60` seconds and `18/6` calls. Add an
after-validator enforcing the total relationships. Pass the MCP search-operation/provider timeout
derived from the active stage configuration; API keys continue through the existing allowlist.

- [ ] **Step 4: Document configuration without including live key values**

Add the five settings to `.env.example` and the README configuration table. Document that
`SERPER_API_KEY` activates the first fallback provider.

- [ ] **Step 5: Run config, CLI, and MCP propagation suites**

Run: `uv run pytest tests/test_config.py tests/test_cli_e2e.py tests/test_mcp.py tests/test_persistence.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gaia_agent/config.py src/gaia_agent/cli.py src/gaia_agent/mcp_client.py src/gaia_agent/mcp_server.py src/gaia_agent/runner.py src/gaia_agent/persistence.py .env.example README.md tests/test_config.py tests/test_cli_e2e.py tests/test_mcp.py tests/test_persistence.py
git commit -m "feat: configure GAIA stage budgets"
```

### Task 3: Enforce separate Specialist and Critic budgets

**Files:**
- Modify: `src/gaia_agent/orchestrator.py:130-230`
- Modify: `src/gaia_agent/orchestrator.py:450-760`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- `StageBudget` tracks stage name, deadline, and logical call limit.
- `_BudgetedToolClient.start_scope(scope, *, stage, max_calls, deadline)` activates a stage allowance.
- `StageBudgetExceeded(stage, reason)` distinguishes `time` from `tool_calls` without private text.

- [ ] **Step 1: Write failing tool-budget isolation tests**

```python
@pytest.mark.asyncio
async def test_specialist_cannot_consume_reserved_critic_calls(fake_tools) -> None:
    client = _BudgetedToolClient(fake_tools, max_calls=8, audit=lambda *_: None)
    client.start_scope("act:lookup:1", stage="specialist", max_calls=6, deadline=None)
    for index in range(6):
        await client.call("web_search", {"query": str(index)})
    with pytest.raises(StageBudgetExceeded, match="specialist"):
        await client.call("web_search", {"query": "extra"})

    client.start_scope("verify:1", stage="critic", max_calls=2, deadline=None)
    await client.call("web_search", {"query": "verify"})
```

Add a test that the call is rejected before invoking the underlying MCP client when the stage deadline
has expired.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `uv run pytest tests/test_orchestrator.py -k 'reserved_critic or stage_deadline' -q`

Expected: FAIL because scopes do not carry budgets.

- [ ] **Step 3: Implement `StageBudget` and pre-call enforcement**

```python
@dataclass(frozen=True)
class StageBudget:
    name: Literal["planner", "specialist", "critic"]
    deadline: float | None
    max_calls: int

class StageBudgetExceeded(RuntimeError):
    def __init__(self, stage: str, reason: Literal["time", "tool_calls"]) -> None:
        super().__init__(f"{stage} stage budget exhausted")
        self.stage = stage
        self.reason = reason
```

Track per-stage logical counts separately from the global logical count. Remove the duplicated
`self.executed_call_count += 1` currently present before MCP execution.

- [ ] **Step 4: Write failing orchestrator time-reservation tests**

```python
@pytest.mark.asyncio
async def test_specialist_timeout_preserves_reserved_critic_deadline(task) -> None:
    specialist = BlockingSpecialist()
    critic = RecordingCritic(verdict="insufficient_evidence")
    orchestrator = _orchestrator(
        task,
        specialist=specialist,
        critic=critic,
        config=_config(
            timeout_seconds=1.0,
            planner_timeout_seconds=0.1,
            specialist_timeout_seconds=0.2,
            critic_timeout_seconds=0.7,
        ),
    )
    prediction = await orchestrator.run(task)
    assert specialist.cancelled is True
    assert critic.calls == 1
    assert prediction.duration_seconds < 1.0
```

Assert the audit payload contains only stage, reason, and numeric budget summaries.

- [ ] **Step 5: Implement stage wrappers**

Wrap Planner, each Specialist invocation, and Critic in `asyncio.timeout_at` using deadlines calculated
once from the task start. Convert stage timeout into `StageBudgetExceeded` and route completed stable
state according to Task 6; do not change scoring behavior here.

- [ ] **Step 6: Run orchestrator and composition tests**

Run: `uv run pytest tests/test_orchestrator.py tests/test_cli_e2e.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/gaia_agent/orchestrator.py tests/test_orchestrator.py tests/test_cli_e2e.py
git commit -m "feat: reserve GAIA critic budgets"
```

### Task 4: Persist authenticated private stage checkpoints

**Files:**
- Modify: `src/gaia_agent/research_notebook.py`
- Modify: `src/gaia_agent/models.py`
- Test: `tests/test_research_notebook.py`
- Test: `tests/test_persistence.py`

**Interfaces:**
- `StageCursor = Literal["act", "evidence_repair", "verify", "correct", "submit"]`.
- `StageCheckpoint` contains cursor, plan, evidence, lineage, candidate, usage, counters, and next step.
- `ResearchNotebook.save_stage_checkpoint(checkpoint)` and `load_stage_checkpoint()` use the notebook trust anchor.

- [ ] **Step 1: Write failing authenticated round-trip tests**

```python
def test_stage_checkpoint_round_trip_is_private_authenticated_and_bounded(tmp_path: Path) -> None:
    notebook = ResearchNotebook(tmp_path / "private", "task-1", run_generation="run-1")
    checkpoint = StageCheckpoint(
        cursor="verify",
        plan=AgentPlan(restatement="r", steps=(PlanStep(specialist="lookup", instruction="i"),)),
        evidence=(Evidence(evidence_id="evidence-0001", kind="source", summary="s", source_url="https://example.test"),),
        evidence_origins={"evidence-0001": 1},
        candidate="answer",
        usage=Usage(total_tokens=10),
        step_count=1,
        verify_count=0,
        next_step_index=1,
        logical_tool_calls=2,
    )
    notebook.save_stage_checkpoint(checkpoint)
    assert notebook.load_stage_checkpoint() == checkpoint
```

Add tampering, wrong run generation, symlink, oversized payload, and mode `0o600` tests.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/test_research_notebook.py -k stage_checkpoint -q`

Expected: FAIL because the API does not exist.

- [ ] **Step 3: Implement typed checkpoint and authenticated storage**

Use a separate contract and schema version:

```python
_STAGE_CHECKPOINT_CONTRACT = "gaia.stage-checkpoint"
_STAGE_CHECKPOINT_SCHEMA_VERSION = 1
_STAGE_CHECKPOINT_FILENAME = ".stage-checkpoint.json"
```

Canonicalize the checkpoint JSON, sign it using the existing notebook HMAC key, enforce the existing
maximum private entry size, and use the existing atomic-write and private-directory checks. Parsing
must validate exact keys and return `None` on any integrity or containment failure.

- [ ] **Step 4: Ensure public result verification rejects leaked private fields**

Extend persistence tests so candidate/checkpoint fields remain forbidden in manifest, prediction notes,
and Langfuse-safe public output structures.

- [ ] **Step 5: Run notebook and persistence suites**

Run: `uv run pytest tests/test_research_notebook.py tests/test_persistence.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gaia_agent/research_notebook.py src/gaia_agent/models.py tests/test_research_notebook.py tests/test_persistence.py
git commit -m "feat: persist private GAIA stage checkpoints"
```

### Task 5: Repair missing evidence links without another web call

**Files:**
- Modify: `src/gaia_agent/agents.py`
- Modify: `src/gaia_agent/research_notebook.py`
- Modify: `src/gaia_agent/runner.py:418-570`
- Test: `tests/test_agents.py`
- Test: `tests/test_research_notebook.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- `ResearchNotebook.linkable_evidence_cards()` returns bounded cards keyed by authenticated entry hash.
- `EvidenceLinkerAgent.link(task, candidate, cards) -> tuple[str, ...]` makes one tool-free structured provider request.
- `ResearchNotebook.evidence_for_hashes(hashes) -> tuple[Evidence, ...]` resolves only known hashes.

- [ ] **Step 1: Write failing allowlist and no-tools tests**

```python
@pytest.mark.asyncio
async def test_evidence_linker_can_select_only_known_card_hashes(fake_provider, task) -> None:
    fake_provider.return_json({"evidence_entry_hashes": ["known"]})
    linker = EvidenceLinkerAgent(fake_provider, reasoning_effort="max")
    selected = await linker.link(task, "answer", ({"entry_hash": "known", "excerpt": "x"},))
    assert selected == ("known",)
    assert fake_provider.requests[0].tools == ()
    assert fake_provider.requests[0].max_tool_calls == 0
```

Add unknown-hash rejection, one local repair attempt, bounded context, and no-card short-circuit tests.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/test_agents.py tests/test_research_notebook.py -k evidence_link -q`

Expected: FAIL because the linker and card-resolution API do not exist.

- [ ] **Step 3: Implement bounded notebook card resolution**

Reuse `select_evidence_cards`; expose entry hashes, source metadata, and excerpts only to the local model
request. Resolve selected hashes back to typed `Evidence` with orchestrator-assigned trusted IDs. Never
accept URLs, summaries, or IDs invented by the linker response.

- [ ] **Step 4: Implement `EvidenceLinkerAgent` using the existing stage repair parser pattern**

The response schema is exactly:

```json
{"evidence_entry_hashes":["known-hash"]}
```

One malformed response degrades to no links; it does not trigger another provider request or web tool.

- [ ] **Step 5: Compose the linker and run targeted suites**

Run: `uv run pytest tests/test_agents.py tests/test_research_notebook.py tests/test_cli_e2e.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gaia_agent/agents.py src/gaia_agent/research_notebook.py src/gaia_agent/runner.py tests/test_agents.py tests/test_research_notebook.py tests/test_cli_e2e.py
git commit -m "feat: repair GAIA evidence links locally"
```

### Task 6: Resume the LangGraph from the latest stable stage and bound task recovery

**Files:**
- Modify: `src/gaia_agent/orchestrator.py`
- Modify: `src/gaia_agent/runner.py:315-388`
- Modify: `src/gaia_agent/models.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- `Orchestrator.run` restores `StageCheckpoint` before graph execution.
- A new graph entry router chooses `plan`, `act`, `evidence_repair`, `verify`, `correct`, or `submit`.
- `TaskPrediction` carries only public recovery metadata: attempt number, recovery count, recovery success, and retry category.

- [ ] **Step 1: Write failing resume-after-Specialist test**

```python
@pytest.mark.asyncio
async def test_restore_specialist_checkpoint_skips_planner_and_specialist(tmp_path: Path, task) -> None:
    notebook = ResearchNotebook(tmp_path / "private", task.task_id, run_generation="run-1")
    notebook.save_stage_checkpoint(_checkpoint(cursor="verify", candidate="restored answer"))
    planner = ForbiddenPlanner()
    specialist = ForbiddenSpecialist()
    critic = ApprovingCritic(checked_evidence_ids=("evidence-0001",))
    orchestrator = _orchestrator(
        task,
        planner=planner,
        specialist=specialist,
        critic=critic,
        notebook=notebook,
    )
    prediction = await orchestrator.run(task)
    assert prediction.answer == "restored answer"
    assert critic.calls == 1
    assert planner.calls == 0
    assert specialist.calls == 0
```

Add tests for Planner checkpoint -> Act, Critic checkpoint -> Submit, and invalid checkpoint -> normal
Plan. Assert checkpoints are saved after each stable transition.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/test_orchestrator.py -k 'checkpoint and (restore or stable)' -q`

Expected: FAIL because graph execution always starts at Plan.

- [ ] **Step 3: Add graph entry routing and checkpoint conversion helpers**

Add a `restore` node after `START`, hydrate `_TaskState`, and route using the authenticated cursor.
Persist a new checkpoint only after a node has produced fully validated state.

- [ ] **Step 4: Write failing bounded-recovery tests**

```python
@pytest.mark.asyncio
async def test_near_deadline_candidate_is_not_replayed_from_plan() -> None:
    clock = FakeClock()
    factory = CheckpointingTimeoutFactory(clock=clock, first_duration=9.0)
    runner = _runner(
        timeout_seconds=10.0,
        orchestrator_factory=factory,
        clock=clock,
    )
    result = await runner._run_vote(_task())
    assert factory.planner_calls == 1
    assert result.prediction.attempt_number == 1
    assert result.prediction.error_category == TerminalErrorCategory.TIMEOUT
```

Also test that an early transient provider failure without a checkpoint gets one retry inside the
remaining global envelope.

- [ ] **Step 5: Replace unconditional two-attempt task replay**

Track one vote deadline in `_run_vote`. Retry only when all are true:

```python
is_early_transient
and not stable_checkpoint_present
and remaining_seconds > minimum_recovery_seconds
```

When a stable checkpoint exists, create the recovery orchestrator with the same notebook generation
and remaining deadline; its restore node skips completed stages. Never sum two full 300-second
allowances.

- [ ] **Step 6: Route Specialist budget exhaustion safely**

If a previously completed Specialist state contains a candidate, route to evidence repair/Verify.
If the active Specialist timed out before producing validated state and no earlier candidate exists,
return typed `timeout`. Critic exhaustion preserves the checkpoint but never submits an unverified
answer.

- [ ] **Step 7: Run orchestrator and CLI recovery suites**

Run: `uv run pytest tests/test_orchestrator.py tests/test_cli_e2e.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/gaia_agent/orchestrator.py src/gaia_agent/runner.py src/gaia_agent/models.py tests/test_orchestrator.py tests/test_cli_e2e.py
git commit -m "feat: resume GAIA tasks from stable stages"
```

### Task 7: Add safe observability, configure Serper, and verify end to end

**Files:**
- Modify: `src/gaia_agent/observability.py`
- Modify: `src/gaia_agent/orchestrator.py`
- Modify: `src/gaia_agent/runner.py`
- Modify locally, ignored: `.env`
- Test: `tests/test_observability.py`
- Test: `tests/test_trace_reliability_e2e.py`

**Interfaces:**
- Stage summaries include `budget_seconds`, `elapsed_seconds`, `remaining_bucket`, and call counts.
- Search summaries include provider, safe timeout category, attempt count, and `fallback_used`.
- Root output includes checkpoint-restored and recovery-source enums, never checkpoint content.

- [ ] **Step 1: Write failing Langfuse metadata tests**

Assert successful fallback and checkpoint recovery produce the required operational fields. Recursively
scan every trace payload and assert absence of task question, answer, URLs, evidence excerpts, prompts,
raw errors, and all configured secret values.

- [ ] **Step 2: Run observability tests and confirm RED**

Run: `uv run pytest tests/test_observability.py tests/test_trace_reliability_e2e.py -k 'stage_budget or fallback or recovery' -q`

Expected: FAIL because the new summaries are missing.

- [ ] **Step 3: Add bounded operational summaries**

Use enums and numeric counts only. Bucket remaining time as `ample`, `low`, or `exhausted`; do not emit
absolute deadlines or exception messages. Close every span in `finally` and preserve the existing
flush behavior.

- [ ] **Step 4: Configure the supplied Serper key safely**

Update only the ignored `.env` entry `SERPER_API_KEY`. Set file mode `0o600`. Verify with value-free
checks:

```bash
git check-ignore -q .env
test "$(stat -f '%Lp' .env)" = "600"
```

Do not print, commit, trace, or copy the key into shell history, reports, fixtures, or documentation.

- [ ] **Step 5: Run the full automated suite**

Run: `uv run pytest -q`

Expected: all tests PASS with no unhandled task or telemetry warnings.

- [ ] **Step 6: Run static checks and public-result privacy checks**

Run: `uv run ruff check src tests`

Run: `git diff --check`

Run a fake-backend 3-task evaluation and `verify-results`; recursively scan the output directory for
secret values, question text, checkpoint fields, and private notebook files. Expected: no matches and
`verify-results` succeeds.

- [ ] **Step 7: Run one bounded live Level 2 validation only after automated checks pass**

Use a new output directory, one balanced 3-task smoke only if the CLI requires balanced sampling, Luna
max, one vote, and the newly bounded budgets. Confirm:

- the Level 2 trace either succeeds or ends within one task envelope;
- a hosted-search timeout shows retry/fallback rather than generic `tool_error`;
- no task repeats Planner after a restored Specialist checkpoint;
- Langfuse root output and public report contain no private content.

- [ ] **Step 8: Commit observability and documentation changes**

```bash
git add src/gaia_agent/observability.py src/gaia_agent/orchestrator.py src/gaia_agent/runner.py tests/test_observability.py tests/test_trace_reliability_e2e.py
git commit -m "feat: trace GAIA stage recovery safely"
```

`.env` must remain untracked.
