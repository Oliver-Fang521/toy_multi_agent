# Task 7 report — safe recovery observability

## Delivered

- Added strict, bounded Langfuse summary allowlists for stage budget state, search fallback state,
  and root recovery source.
- Stage spans now publish configured budget, elapsed duration bucket, and logical/global/executed/
  restored call counts; no graph inputs, answers, evidence, or checkpoint fields are included.
- Search span summaries derive only provider/category/attempt-count/fallback-used fields from
  already-sanitized metadata.
- Root output now records `recovery_source` (`not_needed`, `checkpoint`, or `retry`) and detects
  `checkpoint_restored` from the checkpoint event rather than a restored tool-cache event.
- Tightened metadata enums for stage, specialist, tool, verdict, and existing summary values so
  arbitrary strings cannot pass through those fields.

## Tests and verification

- RED: `uv run pytest tests/test_observability.py tests/test_trace_reliability_e2e.py -k 'stage_budget or fallback or recovery' -q`
  failed first for missing `recovery_source`; a checkpoint-event regression test separately failed
  because the runner checked `tool_call_restored`.
- GREEN focused and impacted coverage:
  `uv run pytest tests/test_observability.py tests/test_trace_reliability_e2e.py tests/test_orchestrator.py tests/test_mcp.py -q`
- Full suite: `uv run pytest -q` — PASS.
- Static/privacy formatting checks: `uv run ruff check src tests` and `git diff --check` — PASS.

The recursive observability test uses only dummy sentinel strings and confirms they are absent from
all recorded trace payloads, including question/answer/candidate/prompt/URL/evidence/error/
checkpoint/notebook/secret categories.

## Scope notes

- No `.env`, real credentials, live GAIA/API calls, fake-backend scans, or public-result scans were
  performed; those remain controller-owned.
- The parked fully-positional custom-provider dispatcher edge was not touched.
