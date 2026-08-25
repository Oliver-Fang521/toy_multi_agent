# Final fix wave report

Base: `ae2c5b7`

## Root-cause hypotheses

1. `EvaluationRunner` owns the vote-wide deadline, but every recovered `Orchestrator` derives a new
   Planner/Specialist/Critic schedule from its own start time. Recovery therefore renews stage time.
2. `SearchProviderChain` gives each retry the complete remaining operation window. Four hosted
   primary attempts can consume the operation before any configured fallback receives a call.
   `SearchDeadlineExceeded` is not a `ToolCallError`, so the MCP server's generic exception boundary
   can serialize an empty `RuntimeError` instead of a typed timeout.
3. Stage telemetry subtracts the nominal stage boundary from `now` and discards the actual invocation
   start argument. Fast Specialist/Critic work therefore reports zero, and a recovered Orchestrator
   has no shared execution-time accumulator.
4. EvidenceLinker's unknown and duplicate selection guards work, but one combined test does not
   independently protect them. Empty cards are checked only after context construction, so a valid
   one-byte budget raises before the intended no-provider short circuit.

## Clean baseline

Command:

```text
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -p no:cacheprovider
```

Result: PASS, exit `0`, `[100%]`.

## RED probes (written before production changes)

### Absolute envelope and actual stage telemetry

Command:

```text
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -p no:cacheprovider \
  tests/test_cli_e2e.py::test_runner_injects_one_absolute_stage_runtime_into_recovery_attempts \
  tests/test_orchestrator.py::test_recovered_act_after_absolute_specialist_cutoff_skips_specialist_but_keeps_critic \
  tests/test_orchestrator.py::test_recovered_act_inside_cutoff_gets_only_remaining_specialist_time \
  tests/test_orchestrator.py::test_stage_summaries_use_actual_cumulative_execution_elapsed \
  tests/test_orchestrator.py::test_stage_remaining_bucket_uses_absolute_cutoff_not_active_elapsed
```

Result: expected RED, `5 failed`. Runner returned a terminal first attempt because it never called
the injected orchestrator's `set_stage_runtime`; all four direct Orchestrator probes failed because
`VoteStageRuntime` did not exist. These failures isolate the missing shared vote object rather than a
fixture or syntax error.

### Deadline-aware search fallback

Command:

```text
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -p no:cacheprovider \
  tests/test_search.py::test_default_retries_reserve_time_for_a_successful_serper_fallback \
  tests/test_search.py::test_retry_reservations_support_three_ordered_search_providers
```

Result: expected RED, `2 failed`. Both failed with `SearchDeadlineExceeded` from the primary path;
neither the two-provider Serper success nor the three-provider SerpAPI success was reached.

### Typed timeout across the real MCP client boundary

Command:

```text
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -p no:cacheprovider \
  tests/test_mcp.py::test_all_search_provider_expiry_is_typed_across_real_mcp_client_boundary
```

Result: expected RED, `1 failed`. Provider calls were `[1, 0, 0]` instead of `[4, 4, 4]`, proving
the primary consumed the whole operation and no fallback slice existed. The typed metadata
assertions remained unreachable after that first behavioral failure.

### EvidenceLinker empty-card order and independent hash guards

Command:

```text
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -p no:cacheprovider \
  tests/test_agents.py::test_evidence_linker_empty_cards_short_circuit_before_tiny_context_build \
  tests/test_agents.py::test_evidence_linker_rejects_the_whole_selection_for_an_unknown_hash \
  tests/test_agents.py::test_evidence_linker_rejects_the_whole_selection_for_a_duplicate_hash
```

Result: expected RED/GREEN characterization, `1 failed, 2 passed`. The one-byte empty-card probe
raised `ValueError: evidence linker context exceeds max_json_bytes`; the separately named unknown
and duplicate guard tests passed current production behavior. Each guard will receive an independent
mutation RED check after the split.

## GREEN and final verification

### Implementation

- `EvaluationRunner` now creates one `VoteStageRuntime` per vote and injects that same object into
  every compatible Orchestrator attempt. Its immutable `StageDeadlineEnvelope` retains the original
  Planner, Specialist, Critic, and vote-wide cutoffs; legacy factories without the optional setter
  remain supported.
- `Orchestrator` rejects an expired stage before constructing/awaiting its operation, uses the shared
  cutoffs for recovered work, and accumulates actual active execution time per stage in the shared
  runtime. Operational summaries use cumulative active time for `elapsed_seconds` and the absolute
  cutoff for `remaining_bucket`.
- `SearchProviderChain` dynamically divides the remaining operation time among the current
  provider's attempts and one reserved slot for every later provider. Calls and backoff share the
  original operation deadline. Exhaustion becomes a bounded `SearchDeadlineExceeded` `ToolCallError`
  whose timeout, retryability, provider, attempt, circuit, and fallback fields survive the real MCP
  server/client boundary.
- `EvidenceLinkerAgent` returns before context construction for empty cards. The unknown-hash and
  duplicate-hash behaviors now have separate tests.

### Independent EvidenceLinker mutation checks

- Removing only the unknown-hash guard made
  `test_evidence_linker_rejects_the_whole_selection_for_an_unknown_hash` fail with
  `('known', 'invented') != ()`; the duplicate guard remained present.
- After restoring that line, removing only the duplicate-hash guard made
  `test_evidence_linker_rejects_the_whole_selection_for_a_duplicate_hash` fail with
  `('known', 'known') != ()`; the unknown guard remained present.
- Both guards were restored before GREEN verification.

### Focused final-fix tests

The 11 exact RED probes were rerun together:

```text
...........                                                              [100%]
```

Result: PASS, exit `0`, `11 passed`.

### Impacted suites

Command covered all search, MCP, orchestrator, CLI, observability, OpenAI provider, DeepSeek
provider, and agent tests:

```text
........................................................................ [ 16%]
........................................................................ [ 32%]
........................................................................ [ 48%]
........................................................................ [ 64%]
........................................................................ [ 80%]
........................................................................ [ 97%]
.............                                                            [100%]
```

Result: PASS, exit `0`, `445 passed`.

### Full repository suite

Command:

```text
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -p no:cacheprovider
```

Result: PASS, exit `0`, `665 passed`, final output `[100%]`.

### Static and diff checks

```text
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check --no-cache src tests
All checks passed!
```

`git diff --check` passed with exit `0` and no output.
