# LangGraph Orchestrator Migration Design

## Goal

Replace only the hand-written orchestration control flow with a LangGraph
`StateGraph`. Preserve the existing Planner, Specialists, Critic, MCP tools,
provider adapters, quality policy, scoring, persistence, terminal error
categories, audit events, and Langfuse spans.

## Scope

The migration is intentionally thin. It changes orchestration ownership, not
agent behavior or the GAIA answer protocol.

In scope:

- Add LangGraph as a runtime dependency.
- Compile one graph per `Orchestrator` instance.
- Express Plan, Act, Verify, Correct, Reject, and Submit as graph nodes.
- Express repeated plan-step execution and critic routing as conditional edges.
- Preserve the current public `Orchestrator.run()` contract and counters.
- Add tests that prove the graph exists and existing success, correction, and
  failure behavior remains unchanged.

Out of scope:

- Rewriting PlannerAgent, Specialist, or CriticAgent.
- Changing prompts, model configuration, tool schemas, or GAIA scoring.
- Adding a LangGraph checkpointer or replacing current persistence.
- Changing Langfuse payload policy or trace contents.
- Adding parallel specialist execution.

## Alternatives Considered

1. **Thin migration (selected):** retain existing business methods and wrap
   them in graph nodes. This has the smallest regression surface and gives the
   project explicit, inspectable control flow.
2. **Full agent rewrite:** reimplement Planner, Specialists, and Critic using
   LangGraph-native agents. This may reduce framework glue later, but it would
   change prompting, tool behavior, and error semantics at the same time.
3. **Graph plus checkpoint migration:** move orchestration and persistence to
   LangGraph together. This adds resumability but risks duplicate or conflicting
   persistence with the existing runner.

## Graph

```text
START -> plan -> act -- more planned steps --> act
                  |
                  +-- plan complete --> verify

verify -- approved --------------------------> submit -> END
       -- rejected and no correction used ---> correct -> verify
       -- rejected after correction ----------> reject -> error mapping
```

`act` processes exactly one `PlanStep` per graph invocation. A conditional edge
uses the next-step index to either loop back to `act` or continue to `verify`.
This makes partial progress visible and prevents the graph node from hiding the
entire plan loop.

## State

A private typed graph state carries:

- the sanitized `GaiaTask` used for execution;
- the existing mutable `_TaskState`, which remains the source of truth for
  plan, evidence, candidate, token usage, and critic verdict;
- the budgeted MCP tool client;
- the next planned-step index;
- the latest `CriticDecision`;
- the final `TaskPrediction`, once submitted.

The existing `_TaskState` is retained to avoid duplicating reducers or changing
how evidence and usage are accumulated. Graph nodes return only explicit state
updates, while existing business methods mutate `_TaskState` as they do today.

## Nodes and Routing

- `plan`: calls the current Planner, records usage, validates the step budget,
  writes audit events, and records the existing Langfuse planner span.
- `act`: dispatches one existing Specialist through `_act()` and advances the
  plan index.
- `verify`: calls `_verify()` and `_apply_quality_policy()` and stores the
  resulting decision.
- `correct`: calls `_correct_once()`; the existing one-correction limit remains
  authoritative.
- `reject`: raises `_CorrectionUnavailable`, preserving the current
  `critic_rejection` terminal category.
- `submit`: calls `_submit()` and stores the prediction.

Routers do not call models or tools. They inspect graph state only. An invalid
or missing state value raises `ValueError`, preserving the existing
`malformed_response` mapping.

## Error and Budget Compatibility

`Orchestrator.run()` continues to own the global timeout, tool context manager,
budgeted tool client, exception-to-terminal-category mapping, usage finalization,
and public counters. LangGraph exceptions are allowed to propagate back into
that existing boundary.

The graph recursion limit is derived from `max_steps` with room for Plan,
Verify, one Correct cycle, Submit, and framework boundary nodes. The existing
`max_steps`, `max_tool_calls`, and one-correction checks remain the authoritative
budgets.

## Observability

Existing Langfuse spans remain inside the reused planner, specialist, critic,
and submit logic. This migration does not upload additional raw prompts,
attachments, webpage bodies, secrets, or hidden reasoning. Audit event names
and safe summaries remain unchanged so existing reports continue to work.

## Testing

Tests will be added before implementation to require:

1. a compiled graph containing the expected nodes;
2. the same Plan -> Act -> Verify -> Submit call order on success;
3. the same one-correction routing behavior;
4. the same terminal error categories for rejection, malformed responses,
   attachment failures, tool failures, timeouts, and budget exhaustion.

The focused orchestrator suite will run first, followed by the full test suite
and Ruff checks.

## Success Criteria

- `Orchestrator` executes through a compiled LangGraph `StateGraph`.
- No existing public API or GAIA scoring behavior changes.
- Existing audit and Langfuse instrumentation remains active.
- All existing tests plus the new graph-structure tests pass.
