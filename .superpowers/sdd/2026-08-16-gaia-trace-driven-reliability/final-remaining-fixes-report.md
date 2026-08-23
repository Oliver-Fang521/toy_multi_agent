# Final Remaining Reliability Fix Wave Report

Date: 2026-08-23

Branch: `codex/gaia-trace-reliability`

## Outcome

Implemented the final cross-module reliability wave without using the GAIA dataset, paid model calls, or successful network access. The production model and reasoning defaults remain `gpt-5.6-luna` and `max`; scorer/selection code was not changed.

The completed behavior includes:

- A safe typed provider-transient exhaustion cause after bounded in-place retry, preserved through Orchestrator output and recovered exactly once by `EvaluationRunner`.
- Persisted and traced task attempt, recovery, retry-reason/category, recovery-success, and provider-attempt counters.
- Visual attachment bytes excluded from the serialized text/evidence-card budget while existing attachment validation and text-budget enforcement remain active.
- Bounded retry metadata for OpenAI and DeepSeek provider generations.
- Root task terminal routes and allowlisted reasoning summaries for fresh, recovered, failed, and checkpoint-restored tasks.
- Exact non-boolean trust-anchor schema version validation.
- Per-tool MCP span key allowlists with unknown/content-like keys omitted and no values emitted.
- Deterministic Critic fallback limited to a concise existing candidate, trusted usable evidence, and benchmark policy, with allowlisted reason codes for every allowed or denied outcome.

## RED/GREEN Evidence

### Provider exhaustion and bounded task recovery

RED:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_openai_provider.py::test_openai_provider_raises_typed_safe_cause_after_transient_attempts_exhaust -q`
  - Failed because the typed exhaustion exception did not exist.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py::test_orchestrator_preserves_safe_provider_transient_exhaustion_cause -q`
  - Failed because `TaskPrediction` did not preserve a retry reason/category.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_e2e.py::test_runner_recovers_one_typed_provider_exhaustion_and_persists_safe_counters -q`
  - Failed because the runner made one call instead of one bounded recovery call.

GREEN:

- Focused provider, Orchestrator, runner success-after-recovery, final-exhaustion, timeout, and non-transient terminal tests passed.
- Successful recovery records attempt `2`, one recovery, a safe typed category/reason, and recovery success; final exhaustion records the same bounded attempt/recovery counts with recovery failure.
- A generic non-transient `retry_exhausted` prediction remains terminal and is not retried.

### Visual modality budget

RED:

- Focused Visual-agent tests failed because an approximately 20 KiB validated image was charged against a 1 KiB text budget and the estimator exceeded the actual text-only payload size.

GREEN:

- Focused agent and Orchestrator tests confirm initial and correction Visual routes receive the image while serialized text/evidence remains within budget.
- Existing excessive-text and unsafe/oversized attachment protections remain GREEN.

### Provider and root trace semantics

RED:

- OpenAI and DeepSeek focused trace tests failed because attempt number and retry decision/category metadata were missing.
- The fresh root-trace regression failed because attempt/recovery fields, terminal route, and bounded reasoning summary were missing.
- The trace reliability simulation failed its old exact terminal-output shape after the new required root fields were introduced.

GREEN:

- Both providers emit bounded structural retry metadata only.
- Fresh, recovered, final-failure, and resumed root-trace tests pass; resumed roots reuse the persisted real attempt number.
- Root reasoning summaries contain only allowlisted route, recovery, and stage codes.

### Deferred hardening

RED:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_research_notebook.py::test_notebook_rejects_non_integer_trust_anchor_schema_versions -q`
  - Both `true` and `1.0` were incorrectly accepted as schema version `1`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_mcp.py::test_mcp_span_omits_argument_and_result_keys_outside_the_tool_contract -q`
  - Span shapes included unknown argument/result keys.
- Three focused Critic fallback tests failed because the fallback was not policy-aware and had no safe reason-code surface.

GREEN:

- Trust anchors now require an exact non-boolean integer schema version.
- MCP span input/output shapes contain only keys registered for the named public tool and never include values.
- Benchmark fallback approves only a nonempty concise candidate with trusted usable evidence; strict policy and unusable provenance reject with bounded reason codes.

### Additional regression findings during final review

RED:

- The configured multi-vote test exposed an incorrectly summed task attempt number (`5` without any recovery).
- A private subclass of `ConnectionError` exposed its custom class name to the persistence validator and failed artifact writing.

GREEN:

- Multi-vote aggregation now uses the maximum real per-vote attempt number while summing recovery/provider counters.
- Task-boundary transient categories are collapsed to an allowlisted MRO category or `task_transient`; custom class names are not persisted or traced.

## Verification

- Baseline: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — GREEN before changes.
- Focused changed-module batches — GREEN:
  - `tests/test_agents.py tests/test_orchestrator.py tests/test_openai_provider.py tests/test_deepseek_provider.py`
  - `tests/test_cli_e2e.py tests/test_observability.py tests/test_persistence.py tests/test_research_notebook.py tests/test_mcp.py`
  - `tests/test_reliability.py tests/test_trace_reliability_e2e.py tests/test_reporting.py tests/test_scoring.py tests/test_config.py tests/test_dataset.py tests/test_search.py`
- Full suite: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` — GREEN at 100%.
- Lint: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — `All checks passed!`
- Whitespace: `git diff --check` — GREEN.
- Build: `uv build --offline` using the existing local uv cache — built both source distribution and wheel successfully.

The first isolated build attempt could not resolve `hatchling` because external DNS/network access was restricted, and the temporary cache did not contain the build dependency. No dependency was downloaded; the final offline build reused the existing local cache.

## Privacy and invariant inspection

- Changed production files contain no credential-like value matches in the diff.
- Only expected source, test, and this report file are part of the change; generated `dist/` artifacts remain ignored.
- No prompt, question, answer, URL, attachment bytes, raw tool body, raw exception message, credential, or hidden reasoning was added to public artifacts or trace payloads.
- MCP spans expose only allowlisted key names and structural counts, never values.
- Root/provider summaries use fixed allowlisted codes rather than model-authored prose.
- `src/gaia_agent/config.py` and `src/gaia_agent/scoring.py` are unchanged, preserving production Luna/max and GAIA scorer/selection semantics.

## Concerns

None remaining. The only environmental issue was isolated build dependency resolution under blocked network access; the required package artifacts were subsequently built successfully in fully offline mode from the existing local cache.
