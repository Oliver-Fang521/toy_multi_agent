# Task 6 Report — Stable LangGraph Resume and Bounded Vote Recovery

## Status

Complete. Task 6 restores authenticated `StageCheckpoint` state into LangGraph at the
persisted stable cursor, saves each stable transition before downstream work, invokes notebook-only
evidence repair exactly at `evidence_repair`, restores logical budgets and cumulative usage, and
replaces unconditional two-attempt replay with one vote-wide deadline.

## RED evidence

Checkpoint entry routing, stable saves, digest restoration, and counter hydration:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py -k 'checkpoint and (restore or stable)' -q
FFF.FFF                                                                  [100%]
6 failed
```

The corrupt-checkpoint fallback passed; all six resume/save tests failed because the graph still
started at Planner and never wrote stable-transition checkpoints.

One-shot evidence repair and trusted evidence-ID assignment:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py::test_evidence_repair_checkpoint_links_once_through_trusted_id_assignment -q
F                                                                        [100%]
```

The candidate reached Critic without linked evidence because the composed Task 5 linker had no graph
node.

Vote-wide deadline and retry gates:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_e2e.py -k 'early_transient or late_transient or near_deadline or vote_deadline or checkpoint_transient' -q
FFFF
```

Four fake-clock tests failed on the absent `clock` interface. The real blocking recovery test then
remained live beyond 30 seconds and was interrupted, directly reproducing the missing vote-wide
deadline.

Recovery accounting:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py::test_failed_critic_refreshes_checkpoint_counters_without_partial_decision tests/test_cli_e2e.py::test_runner_checkpoint_transient_resumes_once_with_remaining_vote_budget -q
FF                                                                       [100%]
```

The checkpoint retained zero Critic calls after one completed logical call, and runner reported 13
tokens instead of the restored cumulative 8.

Task 3 terminal Specialist-timeout ruling:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_e2e.py::test_runner_specialist_timeout_without_stable_candidate_is_terminal -q
F                                                                        [100%]
IndexError: pop from an empty deque
```

The empty deque proved runner incorrectly created a second orchestrator from a Planner-only `act`
checkpoint after Specialist timed out without validated candidate/evidence.

## GREEN evidence

Focused graph checkpoint and evidence-repair slice:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py -k 'checkpoint and (restore or stable or evidence_repair)' -q
........                                                                 [100%]
```

Focused vote-wide recovery slice:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_e2e.py -k 'early_transient or late_transient or near_deadline or vote_deadline or checkpoint_transient' -q
.....                                                                    [100%]
```

Accounting and Task 3 terminal-route regressions:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_e2e.py::test_runner_specialist_timeout_without_stable_candidate_is_terminal tests/test_cli_e2e.py::test_runner_checkpoint_transient_resumes_once_with_remaining_vote_budget tests/test_orchestrator.py::test_failed_critic_refreshes_checkpoint_counters_without_partial_decision -q
...                                                                      [100%]
```

Required impacted suites:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py tests/test_cli_e2e.py -q
........................................................................ [ 55%]
..........................................................               [100%]
```

Fresh full suite:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
........................................................................ [ 11%]
........................................................................ [ 23%]
........................................................................ [ 35%]
........................................................................ [ 46%]
........................................................................ [ 58%]
........................................................................ [ 70%]
........................................................................ [ 82%]
........................................................................ [ 93%]
.....................................                                    [100%]
```

Static and whitespace verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
All checks passed!

git diff --check
# no output
```

## Behavior implemented

- `START` now enters `restore`, which loads only the current `ResearchNotebook`'s authenticated
  checkpoint. Missing, stale, tampered, or invalid checkpoint data routes to normal `plan`.
- Restore hydrates plan, trusted typed evidence, lineage, candidate, usage, graph indices, correction
  state, Critic decision, and global/per-stage logical call counters. Raw SHA-256 hex digests are
  restored with `bytes.fromhex`, preserving duplicate raw-evidence detection.
- Stable checkpoints are saved after valid Plan, each valid Specialist transition, evidence repair,
  Critic decision, and correction, before the next provider/tool stage. Timed-out or malformed active
  Specialist steps roll back their unvalidated step count before checkpointing prior stable state.
- After a terminal stage failure, only accounting on the last stable checkpoint is refreshed. Its
  cursor, plan, candidate, evidence, lineage, and decision are unchanged; no partial provider result
  or partial tool result is checkpointed.
- `evidence_repair` is a single graph node. It calls the linker at most once for that traversal, only
  with authenticated bounded notebook cards, resolves selected hashes through the notebook, and feeds
  typed `Evidence` through `_record_result` for trusted sequential IDs and digest lineage.
- One vote deadline is computed before attempt 1. Each orchestrator await is capped by remaining vote
  time. A no-checkpoint transient gets one fresh retry only in the first half of the envelope and with
  a minimum recovery reserve. A transient with a stable checkpoint gets one resume only with that
  reserve. A near-deadline result returns without another attempt.
- A Specialist time exhaustion is terminal only when the authenticated checkpoint has no validated
  candidate. A candidate without typed evidence routes to evidence repair/Critic. Specialist
  tool-call exhaustion remains terminal. Critic exhaustion keeps the prior stable checkpoint and
  never submits an unverified answer.
- Restored predictions already carry cumulative checkpoint usage, so runner does not sum the first
  prediction into them again. Provider-attempt and public recovery counters remain additive.
- Runtime checkpoint/evidence-repair audit events stay in memory for recovery decisions but are not
  persisted into the public audit allowlist; Task 7 owns the final safe observability expansion.

## Files changed

- `src/gaia_agent/orchestrator.py`
  - Adds restore/evidence-repair graph nodes and stable-cursor routing.
  - Adds checkpoint conversion, stable persistence, failure-accounting refresh, digest restoration,
    and logical counter hydration.
- `src/gaia_agent/runner.py`
  - Adds an injectable monotonic clock, one vote deadline, early fresh-retry and checkpoint-resume
    gates, terminal Specialist-timeout detection, cumulative restored-usage handling, and public audit
    filtering for Task 6 runtime events.
- `tests/test_orchestrator.py`
  - Covers act/verify/submit restore, corrupt fallback, exact stable cursor order, digest bytes,
    logical counters, failure accounting, and one-shot trusted evidence repair.
- `tests/test_cli_e2e.py`
  - Covers early and late no-checkpoint transients, near-deadline checkpoint behavior, checkpoint
    resume, one vote deadline cancellation, cumulative usage, and terminal no-state Specialist
    timeout.

`src/gaia_agent/models.py` required no change: the current Task 4/earlier public contract already
contains the complete private `StageCheckpoint` fields and the bounded public attempt/recovery/retry
metadata consumed by Task 6. Removing or duplicating those fields would add contract churn without
behavior.

## Proposed rulings / cost if wrong

- **Evidence repair uses half of the Critic time remaining at node entry.** It runs after Specialist
  has produced a stable candidate and may be entered precisely because Specialist time is exhausted;
  charging it to Specialist would make recovery impossible at that cursor. The half-window leaves a
  deterministic Critic reserve. Cost if wrong: a slow linker that needed more than half of the
  remaining Critic window fails closed and Critic receives the existing candidate/evidence.
- **A useful retry requires more than `min(critic_timeout, 10% of vote_timeout)`, with a 1 ms floor;
  no-checkpoint retry is “early” only in the first half of the vote.** The plan names
  `minimum_recovery_seconds` and `is_early_transient` but does not specify values. This conservative
  choice makes the 10-second near-deadline boundary deterministic and gives production recovery up to
  a 30-second minimum reserve under the 300-second defaults. Cost if wrong: some recoveries with less
  time could have succeeded, or a different benchmark-specific threshold may be preferable.
- **Terminal failure may refresh accounting on the prior stable checkpoint.** Only cumulative usage
  and logical counters change; stable graph content does not. This reconciles “no partial provider/tool
  state” with “no budget reset.” Cost if wrong: the checkpoint file is rewritten after a failed stage
  even though its stable cursor and semantic state are unchanged.

## Privacy and composition review

- No `.env`, secret, credential, prompt, question, candidate, URL, evidence excerpt, or raw provider
  error was added to public results, public audits, or Langfuse metadata.
- No production model or reasoning setting changed. Runner still composes all agents, including the
  evidence linker, from the shared configured provider; production guards remain
  `gpt-5.6-luna`/`reasoning_effort=max`.
- Checkpoints are loaded only from `ResearchNotebook`; no Langfuse or public metadata reconstructs
  private state.
- Existing untracked `.superpowers/sdd-tools/` content was left untouched and excluded from the
  commit.

## Concerns / handoff

- No open Task 6 blocker.
- Task 7 can map the in-memory `checkpoint_saved`, `checkpoint_restored`, and evidence-repair events
  into its bounded safe Langfuse summary fields without persisting their private checkpoint content.

## Commit

`124f483 feat: resume GAIA tasks from stable stages`

---

## Fix round 1 — adversarial recovery hardening

### RED evidence

The six findings were first encoded as focused regressions (the evidence-repair and Critic findings
each include both of their typed failure variants):

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_runner_scopes_checkpoints_per_vote_but_reuses_scope_for_recovery \
  tests/test_cli_e2e.py::test_runner_cancelled_restored_attempt_retains_prior_cumulative_usage \
  tests/test_cli_e2e.py::test_runner_critic_budget_exhaustion_at_verify_is_terminal \
  tests/test_cli_e2e.py::test_runner_critic_provider_retry_exhaustion_at_verify_is_terminal \
  tests/test_cli_e2e.py::test_runner_specialist_timeout_with_checkpoint_candidate_resumes_from_act \
  tests/test_cli_e2e.py::test_runner_deadline_does_not_wait_for_cancellation_suppression \
  tests/test_cli_e2e.py::test_runner_deadline_starts_before_synchronous_factory_work \
  tests/test_orchestrator.py::test_evidence_repair_transient_failure_is_charged_once_and_routes_to_critic \
  tests/test_orchestrator.py::test_specialist_timeout_with_candidate_and_no_evidence_repairs_then_critic \
  tests/test_orchestrator.py::test_failed_critic_refreshes_checkpoint_counters_without_partial_decision
FFFFFFFFFF                                                               [100%]
10 failed
```

The exact failures were:

- runner called the factory with only `task`, so no vote-stable namespace existed and direct
  namespaced `_run_vote` calls were unsupported;
- evidence-linker transient failure escaped before Critic, candidate-only Specialist stability never
  reached the linker, and Critic provider exhaustion emitted no stage-terminal marker;
- the restored-cancel accounting path selected synthetic zero usage;
- Critic-local timeout/retry exhaustion was eligible for checkpoint recovery;
- `asyncio.timeout` waited through cancellation suppression, and factory drift still allowed the
  expired orchestrator to run.

### GREEN evidence

Final focused adversarial slice:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_runner_scopes_checkpoints_per_vote_but_reuses_scope_for_recovery \
  tests/test_cli_e2e.py::test_runner_cancelled_restored_attempt_retains_prior_cumulative_usage \
  tests/test_cli_e2e.py::test_runner_critic_budget_exhaustion_at_verify_is_terminal \
  tests/test_cli_e2e.py::test_runner_critic_provider_retry_exhaustion_at_verify_is_terminal \
  tests/test_cli_e2e.py::test_runner_specialist_timeout_with_checkpoint_candidate_resumes_from_act \
  tests/test_cli_e2e.py::test_runner_deadline_does_not_wait_for_cancellation_suppression \
  tests/test_cli_e2e.py::test_runner_deadline_starts_before_synchronous_factory_work \
  tests/test_orchestrator.py::test_evidence_repair_failure_is_charged_once_and_routes_to_critic \
  tests/test_orchestrator.py::test_specialist_timeout_with_candidate_and_no_evidence_repairs_then_critic \
  tests/test_orchestrator.py::test_failed_critic_refreshes_checkpoint_counters_without_partial_decision
...........                                                              [100%]
```

Impacted suites:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_orchestrator.py tests/test_cli_e2e.py
........................................................................ [ 51%]
....................................................................     [100%]
```

Fresh full suite:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
........................................................................ [ 11%]
........................................................................ [ 23%]
........................................................................ [ 34%]
........................................................................ [ 46%]
........................................................................ [ 57%]
........................................................................ [ 69%]
........................................................................ [ 80%]
........................................................................ [ 92%]
...............................................                          [100%]
```

Static and whitespace verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
All checks passed!

git diff --check
# no output
```

### Behavior added in fix round 1

- Runner assigns deterministic private namespaces (`vote-0001`, `vote-0002`, …) per majority vote.
  Every recovery attempt inside one vote reuses that namespace; different votes use distinct
  `ResearchNotebook` roots while preserving the public GAIA `task_id`. Production notebook roots
  reject non-internal namespace shapes. Legacy one-argument injected factories remain supported by
  signature inspection; the production/internal interface receives `(task, vote_namespace)`.
- Restored-attempt usage is component-wise bounded below by all prior attempt accounting. A canceled
  resume with synthetic zero usage therefore retains the authenticated prior cumulative charge,
  while a successful cumulative resume is not double-counted.
- Evidence repair owns linker failures. It charges linker usage once, records only a safe private
  failure category, saves the `verify` cursor, and routes directly to Critic. Linker time is capped at
  half of the remaining Critic window so a timeout cannot consume the entire verification reserve.
- A validated candidate is the stable Specialist-timeout criterion even when typed evidence is
  absent. Checkpoint runtime events expose only `candidate_present`; no candidate text enters public
  audit or Langfuse. Candidate-only state routes through authenticated evidence repair when eligible,
  then Critic, while Specialist logical-call exhaustion remains terminal.
- Critic stage time exhaustion and typed provider retry exhaustion at `verify` are terminal to runner
  recovery. Critic provider exhaustion emits a private safe stage marker; raw outer transient
  exceptions remain governed by the existing transient recovery policy.
- Factory creation runs off-loop in a child task under the vote's existing absolute deadline; late
  factory completion cannot start an orchestrator. Each orchestration attempt then runs in its own
  child task. At either deadline, the child is canceled, observed for a bounded 5 ms cleanup window,
  then safely detached with a callback that consumes its eventual outcome. Cancellation suppression
  and synchronous factory work therefore cannot hold the vote open.

### Fix-round files

- `src/gaia_agent/runner.py`: vote namespaces/private roots, compatible factory dispatch, bounded
  child attempts, terminal checkpoint policy, and cumulative usage floor.
- `src/gaia_agent/orchestrator.py`: candidate-presence checkpoint metadata, candidate-only timeout
  routing, contained evidence repair, reserved repair time, and Critic retry terminal marker.
- `tests/test_cli_e2e.py`: cross-vote isolation, restored cancellation accounting, runner terminal
  gates, hard deadline/cancellation suppression, factory drift, and production notebook-root tests.
- `tests/test_orchestrator.py`: linker transient/time containment, candidate-only evidence repair, and
  Critic retry-marker regressions.

### Fix-round proposed ruling / cost if wrong

- **Evidence repair receives half of the Critic time remaining when the node starts.** This leaves a
  deterministic verification reserve while keeping both operations inside the original Critic/vote
  deadlines. Cost if wrong: a slow linker that could have succeeded using more than half of the
  remaining Critic window now fails closed and sends the existing candidate/evidence to Critic.

### Fix-round concerns

- No open blocker. Detached cancellation-suppressing attempts retain a strong reference only until
  their done callback consumes the result/exception; the runner never accepts their late prediction.
- Python cannot stop already-running synchronous worker code. A timed-out factory's asyncio wrapper
  is canceled and observed, its eventual return value is ignored, and it can never launch the
  orchestrator; the worker itself may finish its bounded synchronous cleanup after runner returns.
- All new checkpoint/linker/retry events remain private runtime events. Task 7 still owns final safe
  observability expansion.

### Review-fix RED/GREEN evidence

Independent review found that the first implementation counted synchronous factory time but still
ran the factory on the event loop. The strengthened real-clock test made that envelope failure RED:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_cli_e2e.py::test_runner_deadline_starts_before_synchronous_factory_work
F                                                                        [100%]
assert 0.08555841608904302 < 0.06
1 failed
```

After moving factory creation into the bounded worker phase, both hard-deadline variants were GREEN:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_runner_deadline_starts_before_synchronous_factory_work \
  tests/test_cli_e2e.py::test_runner_deadline_does_not_wait_for_cancellation_suppression
..                                                                       [100%]
```

Read-only re-review result:

```text
Critical/Important issues: none remaining.
Ready: Yes.
```

Fix-round commit: `616d8fd853c8fbaabe86da7d9359e5c013859d58 fix: harden GAIA stage recovery boundaries`

---

## Fix round 2 — cancellation accounting and invocation boundary

### RED evidence

All three Important findings were encoded before production changes. The factory-compatibility
case is parameterized across both supported legacy arities, and the invocation-boundary finding is
split across real tool/provider paths plus bounded-registry cleanup:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_real_cancelled_resume_refreshes_authenticated_usage_before_reraise \
  tests/test_cli_e2e.py::test_runner_legacy_sync_factory_stays_on_caller_event_loop \
  tests/test_cli_e2e.py::test_runner_production_offloaded_factory_stays_inside_vote_deadline \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_tool_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_provider_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_detached_attempt_registry_is_bounded_and_cleans_up
FFFFFFF                                                                  [100%]
7 failed
```

The exact failures established that:

- a restored 7-token checkpoint stayed at 7 after Critic had finalized another known 5 tokens;
- both legacy factory arities ran in a worker without a running event loop;
- no explicit production offload mode existed;
- cancellation-suppressing real orchestration could begin one provider call and one tool call after
  the runner had returned; and
- a second detached attempt started after the test reduced registry capacity to one.

### GREEN evidence

Final focused round-two slice:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_real_cancelled_resume_refreshes_authenticated_usage_before_reraise \
  tests/test_cli_e2e.py::test_runner_legacy_sync_factory_stays_on_caller_event_loop \
  tests/test_cli_e2e.py::test_runner_production_offloaded_factory_stays_inside_vote_deadline \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_tool_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_provider_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_detached_attempt_registry_is_bounded_and_cleans_up
.......                                                                  [100%]
```

Required impacted suites:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py tests/test_orchestrator.py \
  tests/test_openai_provider.py tests/test_deepseek_provider.py
........................................................................ [ 37%]
........................................................................ [ 74%]
..................................................                       [100%]
```

Fresh full suite:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 34%]
........................................................................ [ 45%]
........................................................................ [ 57%]
........................................................................ [ 68%]
........................................................................ [ 80%]
........................................................................ [ 91%]
.....................................................                    [100%]
```

Static and whitespace verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
All checks passed!

git diff --check
# no output
```

### Behavior added in fix round 2

- `Orchestrator.run` now treats `CancelledError` as control flow: stage/provider `finally` blocks
  first add known usage, the authenticated checkpoint receives accounting-only usage and logical
  counter refresh, then cancellation is re-raised. Stable cursor and semantic checkpoint content are
  unchanged. Runner reloads that authenticated cumulative usage after bounded cancellation and uses
  a component-wise floor, producing exactly 12 rather than synthetic 7 or double-charged 19 in the
  real restored-Critic probe.
- Runner creates one mutable absolute invocation deadline per vote and attaches it to every real
  Orchestrator attempt. The deadline is closed before timeout cancellation. Context propagation makes
  that same closed guard visible even to detached cancellation-suppressing control flow.
- New provider completions, every OpenAI/DeepSeek SDK retry attempt, MCP tool calls, and attachment
  reads check the guard immediately before external invocation. A detached task may finish internal
  cleanup, but cannot begin new provider/tool cost after the vote boundary.
- Detached orchestration starts require a capacity reservation. At most 64 running/detached attempts
  are globally retained; capacity exhaustion fails closed without starting another coroutine. Late
  tasks remain strongly referenced, their result/exception is consumed, and the callback removes
  them on completion. A bounded diagnostic exposes tracked/reserved/overflow counts without private
  task data.
- Legacy synchronous one- and two-argument factories execute on the caller event loop, preserving
  loop-affine behavior. Production composition explicitly opts into worker offload, so its internal
  synchronous notebook/orchestrator construction remains inside the absolute vote envelope and a
  late factory result can never start `Orchestrator.run`.

### Fix-round files

- `src/gaia_agent/orchestrator.py`: cancellation accounting refresh, authenticated usage accessor,
  invocation-deadline scope, and tool/attachment boundary checks.
- `src/gaia_agent/reliability.py`: shared absolute invocation guard and task-local propagation.
- `src/gaia_agent/runner.py`: authenticated cancellation accounting floor, provider guard,
  production-only factory offload, and bounded detached-attempt registry.
- `src/gaia_agent/providers/openai.py` and `src/gaia_agent/providers/deepseek.py`: reject each new SDK
  retry attempt after the vote deadline.
- `tests/test_cli_e2e.py`: real cancellation accounting, legacy loop affinity, production slow-factory
  deadline, post-deadline provider/tool rejection, and registry bound/cleanup regressions.

### Fix-round compatibility ruling / cost if wrong

- **Legacy injected synchronous factories retain caller-loop execution; only production composition
  opts into worker offload.** This preserves the existing one-/two-argument injection contract and
  permits loop-affine setup such as `asyncio.get_running_loop()`. Cost if wrong: an untrusted legacy
  synchronous factory can still block its caller beyond the configured vote deadline. This is the
  explicit compatibility limitation requested by the review; production construction remains
  bounded.

### Fix-round concerns and fundamental limit

- No open blocker. Python cannot forcibly terminate an arbitrary coroutine that suppresses
  cancellation, nor prevent arbitrary non-agent side effects already written into that coroutine.
  The enforceable production boundary is therefore external cost initiation: after deadline, no new
  GAIA provider call, provider retry, MCP tool call, or attachment read may begin. In-flight bounded
  operations still receive cancellation/stage deadlines, and lingering internal cleanup is observed
  under bounded registry capacity.
- A timed-out production factory's underlying worker may complete synchronous construction after the
  runner returns, but its returned Orchestrator is discarded and never run. Legacy sync factory
  blocking remains the documented compatibility tradeoff above.
- Deadline and registry diagnostics contain only counts/timing state. No model change, secret,
  prompt, question, candidate, evidence, URL, raw provider error, public audit expansion, or Langfuse
  metadata expansion was introduced.

### Review-fix RED/GREEN evidence

Read-only adversarial review found that production factory wrappers were not yet part of bounded
detached observation. External cancellation could also leave the vote guard open. Both cases were
converted to regressions before the review fix:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_runner_cancellation_during_production_factory_is_observed_and_closes_guard \
  tests/test_cli_e2e.py::test_slow_production_factories_share_bounded_detached_capacity
FF                                                                       [100%]

assert created_deadlines[0].closed is True
assert factory_calls == 1
2 failed
```

After adding an outer vote `finally` and reserving/tracking offloaded production factory wrappers
without canceling away their underlying thread lifetime:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_runner_cancellation_during_production_factory_is_observed_and_closes_guard \
  tests/test_cli_e2e.py::test_slow_production_factories_share_bounded_detached_capacity
..                                                                       [100%]
```

Final expanded focused slice, including all original round-two and review regressions:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_real_cancelled_resume_refreshes_authenticated_usage_before_reraise \
  tests/test_cli_e2e.py::test_runner_legacy_sync_factory_stays_on_caller_event_loop \
  tests/test_cli_e2e.py::test_runner_production_offloaded_factory_stays_inside_vote_deadline \
  tests/test_cli_e2e.py::test_runner_cancellation_during_production_factory_is_observed_and_closes_guard \
  tests/test_cli_e2e.py::test_slow_production_factories_share_bounded_detached_capacity \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_tool_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_provider_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_detached_attempt_registry_is_bounded_and_cleans_up
.........                                                                [100%]
```

The shared capacity now covers both orchestration attempts and offloaded production construction.
Factory timeout/cancellation leaves the worker wrapper strongly observed until the underlying thread
finishes, and late construction results are consumed without ever starting the Orchestrator. An
outer `finally` closes the absolute invocation boundary on every normal, exceptional, and caller-
cancellation exit from `_run_vote`.

Post-review impacted suites and full-suite verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py tests/test_orchestrator.py \
  tests/test_openai_provider.py tests/test_deepseek_provider.py
........................................................................ [ 36%]
........................................................................ [ 73%]
....................................................                     [100%]

UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 34%]
........................................................................ [ 45%]
........................................................................ [ 57%]
........................................................................ [ 68%]
........................................................................ [ 79%]
........................................................................ [ 91%]
.......................................................                  [100%]
```

Final review also reproduced a test-only scheduling race: the 20 ms boundary cases constructed a
real LangGraph inside the timed factory, so a loaded suite could expire before Critic entered. The
Orchestrators are now preconstructed, each test asserts a Critic-start handshake, and the runner
envelope is 50 ms while cancellation is still suppressed for 20 ms before the forbidden call.

```text
for run in 1 2 3 4 5; do
  UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
    tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_tool_call_after_vote_deadline \
    tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_provider_call_after_vote_deadline
done
..                                                                       [100%]
..                                                                       [100%]
..                                                                       [100%]
..                                                                       [100%]
..                                                                       [100%]
```

Final read-only re-review: **no Critical or Important findings; Ready: Yes.**

---

## Fix round 3 — pre-teardown accounting and cancellation-safe ownership

### RED evidence

Both controller findings were reproduced against `5085272` before production changes:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_real_cancelled_resume_refreshes_before_slow_tool_teardown \
  tests/test_cli_e2e.py::test_repeated_cleanup_cancellation_never_orphans_reserved_attempts
FF                                                                       [100%]
```

The real restored Orchestrator returned while its 50 ms tool-context teardown was still running,
but immediate final/checkpoint usage was still 7 instead of 12. The repeated cleanup probe showed
reservation poisoning exactly as `[0, 1, 2]` before three successive attempts instead of returning
to `[0, 0, 0]`; its children also lacked the exception-consuming done callback.

### GREEN evidence

Focused controller regressions:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_real_cancelled_resume_refreshes_before_slow_tool_teardown \
  tests/test_cli_e2e.py::test_repeated_cleanup_cancellation_never_orphans_reserved_attempts
..                                                                       [100%]
```

Expanded recovery/deadline slice, including all round-two and round-three regressions:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_real_cancelled_resume_refreshes_authenticated_usage_before_reraise \
  tests/test_cli_e2e.py::test_real_cancelled_resume_refreshes_before_slow_tool_teardown \
  tests/test_cli_e2e.py::test_runner_legacy_sync_factory_stays_on_caller_event_loop \
  tests/test_cli_e2e.py::test_runner_production_offloaded_factory_stays_inside_vote_deadline \
  tests/test_cli_e2e.py::test_runner_cancellation_during_production_factory_is_observed_and_closes_guard \
  tests/test_cli_e2e.py::test_slow_production_factories_share_bounded_detached_capacity \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_tool_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_real_detached_orchestrator_cannot_start_provider_call_after_vote_deadline \
  tests/test_cli_e2e.py::test_detached_attempt_registry_is_bounded_and_cleans_up \
  tests/test_cli_e2e.py::test_repeated_cleanup_cancellation_never_orphans_reserved_attempts
...........                                                              [100%]
```

Impacted suites and fresh full suite:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py tests/test_orchestrator.py \
  tests/test_openai_provider.py tests/test_deepseek_provider.py
........................................................................ [ 36%]
........................................................................ [ 72%]
......................................................                   [100%]

UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 34%]
........................................................................ [ 45%]
........................................................................ [ 56%]
........................................................................ [ 68%]
........................................................................ [ 79%]
........................................................................ [ 90%]
.........................................................                [100%]
```

```text
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
All checks passed!

git diff --check
# no output
```

### Behavior added in fix round 3

- Cancellation propagated by the active provider/graph is now caught inside the tool-context body.
  Completed provider usage, raw tool usage, global logical calls, and per-stage logical calls are
  persisted to the prior authenticated semantic checkpoint before `__aexit__` begins. Cancellation
  is immediately re-raised so normal resource teardown still owns its lifecycle.
- Raw tool usage is marked collected on that early path, and the later cancellation boundary knows
  the accounting checkpoint was already refreshed. Slow teardown therefore cannot delay accounting,
  and eventual completion neither repeats the refresh nor double-charges usage.
- `_cancel_and_observe` transfers its existing reservation into strong tracked ownership and installs
  the exception-consuming done callback synchronously before calling `cancel()` or awaiting cleanup.
  A second cancellation can abort only the optional bounded wait; it cannot strand a reserved slot
  or orphan the child outcome.
- The deterministic ownership test repeats the second-cancellation race three times. Each cycle
  observes reserved/tracked `0/1` while the child is pending, then `0/0` after its late exception is
  consumed. The separate production-factory cancellation test continues to prove its non-awaiting
  ownership-transfer path also moves reserved `1` to tracked `1`.

### Fix-round files and concerns

- `src/gaia_agent/orchestrator.py`: pre-teardown cancellation accounting and single-collection/
  single-refresh guards.
- `src/gaia_agent/runner.py`: ownership-first bounded cancellation cleanup.
- `tests/test_cli_e2e.py`: real slow-`__aexit__` accounting and repeated second-cancellation ownership
  regressions, plus the production-factory reserved-state assertion.

No new model, provider, public audit, Langfuse, secret, prompt, candidate, evidence, or URL behavior
was introduced. The existing documented Python limit remains: arbitrary cancellation-suppressing
code cannot be forcibly terminated, but every admitted task stays within bounded tracked ownership
and cannot begin new guarded provider/tool work after deadline.

Final independent read-only review found no Critical or Important findings and returned
**Ready: Yes**. Its focused cancellation slice passed 3/3; Ruff and `git diff --check` were clean.

---

## Fix round 4 — immediate completed-turn usage publication

### RED evidence

The production `CriticAgent` cancellation probe and the provider/agent usage probes were added
before production changes. All seven failed against `e0ef3a5`:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cli_e2e.py::test_real_critic_publishes_malformed_turn_usage_before_cancelled_local_repair \
  tests/test_agents.py::test_specialist_repairs_a_parsed_result_with_the_wrong_role \
  tests/test_openai_provider.py::test_openai_provider_publishes_completed_usage_before_cancelled_tool_call \
  tests/test_openai_provider.py::test_openai_provider_aggregates_usage_and_extracts_sources \
  tests/test_openai_provider.py::test_openai_provider_usage_observers_are_invocation_local_under_concurrency \
  tests/test_deepseek_provider.py::test_deepseek_provider_publishes_completed_usage_before_cancelled_tool_call \
  tests/test_deepseek_provider.py::test_deepseek_provider_executes_tool_loop_and_echoes_reasoning
FFFFFFF                                                                  [100%]
```

The exact real-Critic probe reached the second, tool-free repair request, canceled it, and then
observed `critic.last_usage == 0` instead of the completed malformed turn's 5 tokens. After fixing
the test's independent stage-budget setup, its focused RED was:

```text
assert critic.last_usage == Usage(input_tokens=5, total_tokens=5)
E assert Usage(total_tokens=0) == Usage(input_tokens=5, total_tokens=5)
1 failed
```

The agent local-repair fake received no usage observer, and both production adapters rejected the
new per-invocation `usage_observer` interface, so successful tool-loop totals, cancellation after a
completed turn, and concurrent observer isolation were all independently RED.

### GREEN evidence

Required cancellation, success/no-double, local-repair, concurrency, and SDK-retry probes:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_openai_provider.py::test_openai_provider_retries_transient_failures_once \
  tests/test_deepseek_provider.py::test_deepseek_provider_retries_transient_error_once \
  tests/test_cli_e2e.py::test_real_critic_publishes_malformed_turn_usage_before_cancelled_local_repair \
  tests/test_agents.py::test_specialist_repairs_a_parsed_result_with_the_wrong_role \
  tests/test_openai_provider.py::test_openai_provider_publishes_completed_usage_before_cancelled_tool_call \
  tests/test_openai_provider.py::test_openai_provider_aggregates_usage_and_extracts_sources \
  tests/test_openai_provider.py::test_openai_provider_usage_observers_are_invocation_local_under_concurrency \
  tests/test_deepseek_provider.py::test_deepseek_provider_publishes_completed_usage_before_cancelled_tool_call \
  tests/test_deepseek_provider.py::test_deepseek_provider_executes_tool_loop_and_echoes_reasoning
...........                                                              [100%]
```

Impacted agent, provider, graph, runner/CLI, and trace-reliability suites:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_agents.py tests/test_openai_provider.py tests/test_deepseek_provider.py \
  tests/test_orchestrator.py tests/test_cli_e2e.py tests/test_trace_reliability_e2e.py
........................................................................ [ 24%]
........................................................................ [ 49%]
........................................................................ [ 74%]
........................................................................ [ 99%]
.                                                                        [100%]
```

Fresh full suite:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 33%]
........................................................................ [ 45%]
........................................................................ [ 56%]
........................................................................ [ 67%]
........................................................................ [ 79%]
........................................................................ [ 90%]
.............................................................            [100%]
```

```text
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
All checks passed!

git diff --check
# no output
```

### Behavior added in fix round 4

- `ModelProvider.complete` now accepts an optional synchronous, per-invocation usage observer. The
  deadline guard forwards that same observer without storing it, so concurrent votes cannot share
  provider-global accounting state.
- OpenAI Responses and DeepSeek Chat adapters publish a delta immediately after every completed SDK
  response and before any following tool execution, schema recovery, tool-loop continuation, or
  local stage repair can await. A failed SDK attempt that yields no response publishes nothing.
- Provider return values retain the original aggregate usage. Agents overwrite their invocation
  accumulator with that aggregate only on success, so already-published deltas are not added again.
- Planner, Specialist, Critic, and Evidence Linker create an invocation-local observer backed by
  their existing `ContextVar`. Each also exposes the latest published cumulative value outside the
  child graph context, allowing cancellation accounting to be read before slow teardown completes;
  concurrent active invocations continue to read their own task-local values.
- The real restored-Critic regression starts from 7 authenticated tokens, completes a malformed
  5-token response, and is canceled in the second repair request. Its immediate agent/result/
  checkpoint totals are now 5/12/12; later cleanup leaves them unchanged.
- Successful local repair and both production tool loops retain exact prior totals rather than
  double-counting. Transient SDK retry probes publish only the one response that actually completed.

### Fix-round files and concerns

- `src/gaia_agent/providers/base.py`, `providers/__init__.py`: optional observer protocol/export.
- `src/gaia_agent/providers/openai.py`, `providers/deepseek.py`: completed-response delta publication.
- `src/gaia_agent/agents.py`: invocation-local accumulation and cancellation-visible publication.
- `src/gaia_agent/runner.py`: deadline-guard observer forwarding.
- Agent/provider/CLI/trace test doubles were migrated to the optional keyword interface; focused
  regressions live in `tests/test_agents.py`, `tests/test_openai_provider.py`,
  `tests/test_deepseek_provider.py`, and `tests/test_cli_e2e.py`.

No model selection, reasoning effort, checkpoint semantic state, vote namespace, retry deadline,
tool budget, private/public observability boundary, secret, prompt, candidate, evidence, or URL
behavior changed. The observer addition is an internal compatible extension: callers may omit it,
while implementations used through the internal `ModelProvider` protocol must accept the optional
keyword. Outside an active invocation context, a shared agent reports its latest published usage;
production constructs independent agents per vote, while concurrent active calls remain isolated
by `ContextVar` and were regression-tested.

Final independent read-only review found no Critical or Important findings and returned
**Ready: Yes**. It confirmed response-before-await publication across both production providers,
equivalent aggregate replacement on success, cancellation preservation, guarded forwarding, fake
compatibility, and concurrent observer isolation; its provider/agent and focused slices passed, and
Ruff plus `git diff --check` were clean.

---

## Fix round 5 — legacy provider observer compatibility

### RED evidence

The fix-four observer call sites passed `usage_observer=` unconditionally, which rejected legacy
structural providers that still implement `complete(request, executor=None)`. The focused legacy
Agent, local-repair cancellation, guarded ProviderFactory, `**kwargs`, and body-`TypeError` tests
were added around a pre-invocation signature-binding dispatcher. Additional compatibility tests
first exposed three binding holes in the inherited dispatcher:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_agents.py::test_specialist_accepts_keyword_only_request_name_and_executor \
  tests/test_agents.py::test_provider_positional_only_observer_keeps_keyword_only_executor \
  tests/test_agents.py::test_uninspectable_provider_uses_historical_call_once_for_local_repair
FFF
```

- a keyword-only request parameter named `payload` was never supplied;
- a positional-only `usage_observer` caused a declared keyword-only executor to be lost; and
- an uninspectable tool-free provider was called as `complete(request, None)` rather than the
  historical `complete(request)` form.

The independent read-only review then found one more binding regression. Its focused test was RED:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_agents.py::test_provider_varargs_keeps_declared_keyword_executor
F
```

The positional candidate bound the executor into `*args`, leaving the declared keyword executor as
`None`.

### GREEN evidence

The final compatibility slice, including the review regression, passed:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_agents.py::test_specialist_accepts_legacy_provider_with_keyword_only_executor \
  tests/test_agents.py::test_specialist_accepts_keyword_only_request_name_and_executor \
  tests/test_agents.py::test_provider_positional_only_observer_keeps_keyword_only_executor \
  tests/test_agents.py::test_provider_varargs_keeps_declared_keyword_executor \
  tests/test_agents.py::test_uninspectable_provider_uses_historical_call_once_for_local_repair \
  tests/test_agents.py::test_provider_compatibility_dispatch_does_not_retain_provider_instance \
  tests/test_agents.py::test_legacy_provider_first_call_usage_survives_cancelled_local_repair \
  tests/test_agents.py::test_provider_var_keyword_signature_receives_usage_observer \
  tests/test_agents.py::test_provider_body_type_error_is_not_retried \
  tests/test_cli_e2e.py::test_provider_factory_deadline_guard_accepts_legacy_complete_signature
..........
```

The existing exact per-turn built-in OpenAI/DeepSeek usage, cancellation, graph, runner/CLI, and
trace-reliability suites were also rerun, followed by the full suite and static checks.

### Compatibility behavior

- Agents and `_DeadlineGuardedProvider` use one exported `complete_with_usage_observer` invocation
  helper. It inspects and binds a supported structural signature before the only provider call, so a
  `TypeError` raised from provider code is never interpreted as a binding failure or retried.
- Legacy providers that do not accept the observer are called once and publish their successful
  aggregate response usage once. Consequently, their first completed local-repair turn remains
  available if the later repair is cancelled, while no legacy provider can expose an unfinished
  internal turn.
- Explicit observer parameters and `**kwargs` receive the per-invocation observer. Named keyword
  executors are preferred over a variadic positional match; positional-only observer/executor forms
  and keyword-only request names are bound before invocation.
- The uninspectable fallback makes exactly the prior call shape: `complete(request)` without an
  executor and `complete(request, executor)` with one. It does not speculate or retry alternative
  shapes, so it retains only successful aggregate accounting until the provider adopts the observer
  interface.
- The helper has no provider registry or cache. The direct weak-reference regression proves an
  ephemeral provider is collectable after completion.

### Files

- `src/gaia_agent/providers/base.py`: documented structural-provider protocol and one-shot,
  signature-compatible invocation helper.
- `src/gaia_agent/providers/__init__.py`: helper export.
- `src/gaia_agent/agents.py`, `src/gaia_agent/runner.py`: route all observer-aware calls through the
  helper.
- `tests/test_agents.py`, `tests/test_cli_e2e.py`: Agent, cancellation, signature, ProviderFactory,
  TypeError, observer, and lifetime regressions.
