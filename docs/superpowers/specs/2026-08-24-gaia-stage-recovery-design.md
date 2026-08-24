# GAIA Stage Recovery and Search Resilience Design

## Objective

Improve GAIA accuracy and cost control when hosted web search is slow or unavailable. A task that
has already produced useful research or a candidate answer must not discard that work and restart
from planning. The design preserves the existing GAIA final-answer contract and keeps all research
content private.

## Scope

This change covers:

- classification and fallback for hosted-search timeouts;
- deadline-aware provider retries;
- separate Specialist and Critic time/tool budgets;
- durable private checkpoints after stable graph stages;
- recovery from the latest stable stage instead of whole-task replay;
- evidence linkage repair using existing notebook cards before new web research;
- safe Langfuse summaries for budgets, fallback, checkpoints, and recovery;
- configuration, CLI, manifest, and regression tests for these behaviors.

It does not change the GAIA scorer, expose private reasoning, or introduce a new search provider.
Serper and SerpAPI remain optional providers activated only by their existing environment keys.

## Current Failure

`HostedOpenAIBackend.web_search` bounds the OpenAI Responses request with `asyncio.wait_for`.
That operation raises the built-in `TimeoutError`, while the provider classifier recognizes the
OpenAI SDK's `APITimeoutError` but not the built-in exception. The unclassified exception bypasses
`SearchProviderChain` retry and fallback behavior and becomes a generic tool error.

At the task layer, one 300-second attempt may consume the full deadline after producing a candidate.
`EvaluationRunner` then creates a fresh orchestrator and repeats Planner and Specialist. The second
attempt cannot reliably reuse notebook entries because generated search arguments may differ.

## Chosen Approach

Use checkpointed stage recovery with reserved budgets. This is preferred over a timeout-only patch,
which would leave expensive whole-task replay intact, and over a full event-sourced runtime, which
would add unnecessary complexity.

### Search error normalization

`OpenAISearchProvider` converts both built-in `TimeoutError` and supported SDK transient exceptions
into a retryable `SearchProviderError`. The public category is `timeout`; raw exception messages are
never persisted or traced.

`SearchProviderChain` receives a deadline callback. Before retrying or moving to a fallback provider,
it verifies that the remaining stage budget can accommodate the configured backoff and one bounded
request. If not, it raises a typed stage-deadline error without starting another network operation.
The request timeout is the minimum of the hosted-search cap and the remaining stage budget.

Provider ordering remains OpenAI hosted search, Serper, then SerpAPI. Missing keys omit the provider.
The supplied Serper key is stored only as `SERPER_API_KEY` in the ignored `.env` and passed through the
existing MCP environment allowlist.

### Stage budgets

The task retains one global hard deadline and gains explicit stage budgets:

- Planner has a bounded planning allowance and no tools.
- Specialist receives its own elapsed-time and tool-call allowance.
- Critic receives a reserved elapsed-time and tool-call allowance that Specialist cannot consume.
- The global tool budget remains a final hard cap.

Every tool call checks both its stage allowance and the global allowance before execution. Restored
notebook calls count as logical calls but not executed network calls, preserving current accounting.
The default stage values are derived conservatively from the global limits and are configurable via
environment and CLI. They are included in the run manifest so unsafe resume with different behavior
is rejected.

### Private stage checkpoints

A signed, task-scoped checkpoint is written after:

1. Planner produces a valid plan;
2. each Specialist step produces valid state;
3. evidence-link repair completes;
4. Critic produces a valid decision.

The checkpoint stores the stable graph cursor, plan, candidate, typed evidence, evidence lineage,
usage counters, completed step index, and tool budget counters. It lives under the existing private
run directory, uses the notebook run-generation trust anchor, bounded reads, atomic replacement, and
owner-only permissions. It never appears in `predictions.jsonl`, `report.md`, or Langfuse payloads.

Raw tool results remain in `ResearchNotebook`; the stage checkpoint references bounded evidence and
notebook entry hashes instead of duplicating full webpages. Invalid, stale, oversized, or improperly
signed checkpoints fail closed and are ignored.

### Recovery behavior

Recovery selects the latest authenticated stable checkpoint:

- a Planner checkpoint resumes at the next Specialist step;
- a Specialist checkpoint with a candidate resumes at evidence repair or Critic;
- a Critic checkpoint resumes at correction or submission;
- no checkpoint allows the existing bounded whole-task retry only for an early transient provider
  failure;
- a task that consumed most of its deadline or produced a candidate never restarts Planner.

Recovery has a separate bounded allowance inside the configured task envelope; it does not silently
turn a 300-second configuration into two 300-second attempts. Usage and attempt metadata remain
additive and auditable.

### Evidence-link repair

When Specialist returns a candidate but no evidence IDs while successful notebook cards exist, the
orchestrator performs one local semantic repair request with tools disabled. The request contains only
the candidate and bounded evidence cards and may return only known evidence IDs. It cannot introduce
new evidence or change the candidate.

If repair succeeds, Critic receives the linked evidence. If repair fails, Critic may use only its own
reserved tool budget. Benchmark policy may use its existing degraded-approval behavior only when
trusted usable evidence is present; strict policy remains unchanged.

### Observability

Langfuse records safe summaries only:

- stage name, configured budget, elapsed time, and remaining bucket;
- search provider, attempt count, timeout category, and fallback-used flag;
- checkpoint stage and whether recovery restored it;
- recovery source and terminal route;
- logical, restored, and executed tool-call counts.

Prompts, raw webpage text, candidate answers, evidence excerpts, URLs, keys, and exception messages
remain excluded from trace summaries.

## Failure Semantics

- Search timeout with time remaining: retry or fallback.
- Search timeout without time remaining: typed stage-budget exhaustion.
- Specialist budget exhaustion with a stable candidate: checkpoint and proceed to bounded evidence
  repair/Critic.
- Critic budget exhaustion: preserve the checkpoint and return a typed terminal outcome; do not submit
  an unverified answer.
- Corrupt checkpoint: ignore it and follow the safe early-retry rules.
- Missing fallback key: skip that provider without exposing configuration values.

## Testing Strategy

Implementation follows red-green-refactor. Required tests prove:

1. built-in `TimeoutError` triggers retry and Serper fallback;
2. retry does not begin when the stage deadline cannot accommodate it;
3. Specialist cannot spend Critic's reserved time or tool calls;
4. authenticated checkpoints round-trip and invalid checkpoints fail closed;
5. recovery after Specialist does not call Planner or Specialist again;
6. a near-deadline task with a candidate is not replayed from the beginning;
7. evidence repair can select only known notebook evidence IDs and uses no tools;
8. strict and benchmark quality policies retain their existing submission rules;
9. manifests include all behavior-defining budget settings;
10. Langfuse outputs contain recovery metadata without private content;
11. existing unit, integration, and CLI suites remain green.

## Success Criteria

- The reproduced Level 2 timeout path cannot bypass fallback through a built-in `TimeoutError`.
- A task with completed Specialist work never repeats Planner because Critic ran out of time.
- Total wall-clock duration remains within the configured task envelope plus a small cancellation
  cleanup tolerance, rather than doubling to two full task deadlines.
- Existing GAIA scoring behavior is unchanged.
- No secrets or private research content appear in Git, public results, or Langfuse.
