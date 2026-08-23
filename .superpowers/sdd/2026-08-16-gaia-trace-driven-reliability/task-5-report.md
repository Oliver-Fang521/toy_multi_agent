# Task 5: Structured Tool Errors and Search Provider Fallback

## Delivered

- Added `SafeToolFailure` and converted recoverable tool failures to a stable,
  non-sensitive JSON envelope for the model.
- Added injected-`httpx.AsyncClient` Serper and SerpAPI adapters with bounded,
  normalized `answer`/`sources` output, safe provider errors, retries, ordered
  OpenAI → Serper → SerpAPI fallback, and a transient-failure circuit breaker.
- Added MCP propagation and redaction for `SERPER_API_KEY` and `SERPAPI_API_KEY`.
- Added safe direct-webpage error categories for HTTP status, timeout, policy,
  parse, and empty-content failures.
- Added empty provider-key placeholders to `.env.example`.

No production model or reasoning-effort configuration was changed.

## TDD evidence

RED commands and observed failures:

1. `uv run pytest tests/test_openai_provider.py::test_openai_provider_recovers_from_tool_failure_without_leaking_message -q`
   failed because the response only contained `{"error":"tool_failed"}`.
2. `uv run pytest tests/test_mcp.py::test_mcp_server_env_propagates_only_optional_search_provider_keys -q`
   failed with `KeyError: 'SERPER_API_KEY'`.
3. `uv run pytest tests/test_mcp.py::test_mcp_client_categorizes_webpage_http_errors_without_retaining_raw_message -q`
   failed because `ToolExecutionError` had no structured `category`.

GREEN verification:

- `uv run pytest tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py -q`
  completed successfully: 114 tests passed.
- `uv run ruff check src/gaia_agent/search.py src/gaia_agent/providers/base.py src/gaia_agent/providers/openai.py src/gaia_agent/mcp_server.py src/gaia_agent/mcp_client.py tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py`
  completed successfully.
- `git diff --check` completed successfully.

No network, paid-model, or GAIA calls were made.

## Review follow-up: transient-only fallback and webpage resilience

- Search provider retries now use bounded exponential backoff. Only explicitly
  transient, normalized provider failures are retried, counted toward a circuit,
  or allowed to fall through. Permanent and unclassified failures return directly.
- Hosted OpenAI search is wrapped so real SDK timeout/transport/status failures
  are normalized before the provider chain decides whether fallback is allowed.
- Direct webpage reads retry transient failures with bounded backoff, emit safe
  HTTP/timeout/policy/empty-content categories, and return bounded search snippets
  after transient retry exhaustion.
- Server-owned search clients are retained in settings and closed alongside an
  owned webpage client when the MCP process exits.

RED evidence:

1. The new chain tests showed no retry delay, permanent parse failures falling
   through to Serper, and arbitrary `RuntimeError` values opening a circuit.
2. The direct webpage tests showed a transient reader failure returned an MCP
   error without retry, snippets were never returned, and empty content was sent
   as a successful page.
3. The OpenAI wiring regression raised `APITimeoutError` directly instead of
   classifying it and continuing to Serper.

GREEN verification:

- `uv run pytest tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py -q`
- `uv run ruff check src/gaia_agent/search.py src/gaia_agent/providers/base.py src/gaia_agent/providers/openai.py src/gaia_agent/mcp_server.py src/gaia_agent/mcp_client.py tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py`
- `git diff --check`

All commands completed successfully. No network, paid-model, or GAIA calls were made.

## Whole-branch review follow-up: availability, persistence, and safe trace decisions

- Typed provider-local permanent failures (including 401, malformed response,
  and empty content) are not retried or circuit-counted, but now fall through
  the configured OpenAI → Serper → SerpAPI chain. Untyped implementation errors
  still fail immediately. Exhaustion emits one content-free
  `search_unavailable` failure with bounded attempt/circuit/fallback decisions.
- Persistence now accepts exactly two public `read_webpage` success shapes: the
  direct page shape and the already bounded/redacted search-fallback shape
  (`url`, `answer`, `sources`, `fallback`). A real `EvaluationRunner` regression
  verifies that the fallback audit survives persistence and integrity checking.
- MCP failure spans now record only allowlisted provider/category/retryability,
  bounded attempt/circuit counts, safe HTTP status, and fallback decisions.
  Successful webpage fallback spans record `fallback_used`; raw result content
  remains absent. Malformed structured failure metadata fails closed, and the
  observability adapter independently rejects content-bearing or unbounded
  decision values.
- The exact public MCP success/failure payload contracts remain unchanged.
  Content-free routing decisions use private MCP `_meta`, are validated again
  by the client, and never enter structured tool results or persistence payloads.

Strict RED evidence:

1. `uv run pytest tests/test_search.py::test_search_chain_falls_through_nonretryable_provider_local_failures_without_circuit tests/test_search.py::test_search_chain_raises_one_safe_unavailable_error_after_all_provider_local_failures -q`
   failed in four cases because nonretryable `SearchProviderError` values were
   re-raised before the next provider.
2. `uv run pytest tests/test_persistence.py::test_read_webpage_search_fallback_audit_shape_is_public_and_allowlisted tests/test_cli_e2e.py::test_evaluation_runner_persists_read_webpage_search_fallback_audit -q`
   failed twice because the fallback result-key shape was rejected.
3. `uv run pytest tests/test_mcp.py::test_mcp_failure_span_records_only_bounded_structured_failure_metadata tests/test_mcp.py::test_mcp_success_span_records_fallback_decision_without_result_content -q`
   failed twice because spans contained only `error_type` or result-key count.
4. Focused regressions separately failed before implementation for unavailable
   circuit/fallback decisions, safe server serialization, malformed-envelope
   fail-closed behavior, and observability value bounds.

GREEN verification:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_search.py tests/test_mcp.py tests/test_observability.py tests/test_persistence.py tests/test_cli_e2e.py -q` — 202 tests passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` — passed.
- `git diff --check` — passed.

No network, paid-model, or GAIA calls were made. Production model, reasoning
effort, scoring, and stage configuration remain unchanged (`gpt-5.6-luna`,
`reasoning_effort=max`).

## Review follow-up round 2: bounded snippets and parser categories

- Search fallback results are now explicitly marked as search-shaped results, so
  `_safe_tool_call` redacts them and applies `_bounded_web_search_result` after
  construction. The fallback uses the normalized `answer`/`sources` schema and
  remains limited to 16 KiB and eight sources.
- Webpage `ValueError` handling now recognizes only the specific URL safety
  messages as `policy_rejection`; other parser-shaped values are `parse_failure`.
  HTTP decode and HTML parser errors also normalize to `parse_failure`.

RED evidence:

1. An oversized twenty-source fallback produced a page-shaped `text` result with
   unlimited sources instead of an `answer` bounded by the web-search contract.
2. A parser `ValueError` reached the MCP boundary as `policy_rejection`.

GREEN verification:

- `uv run pytest tests/test_mcp.py::test_http_reader_classifies_content_decode_errors_as_parse_failures -q`
- `uv run pytest tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py -q`
- `uv run ruff check src/gaia_agent/search.py src/gaia_agent/providers/base.py src/gaia_agent/providers/openai.py src/gaia_agent/mcp_server.py src/gaia_agent/mcp_client.py tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py`
- `git diff --check`

All commands completed successfully. No network, paid-model, or GAIA calls were made.

### Independent-review closure

An independent review identified three integration risks, all covered by new
regressions: decision fields had altered the exact public failure envelope,
successful provider fallback was not distinguishable in traces, and an
oversized fallback URL could cause `url`/`fallback` to be dropped. Decision
metadata now travels only through bounded private MCP `_meta`; the public error
envelope remains exact; and fallback URLs are capped before the 16 KiB search
bound. Oversized sources are trimmed before extra fallback fields are considered,
so the four-key persistence contract remains stable even for a single huge
source field.

## Fix round: OpenAI fail-fast, empty-result fallback, and consecutive circuits

- `OpenAISearchProvider` now normalizes only typed provider errors, recognized
  HTTP-status errors, and recognized transient SDK errors. Unknown backend or
  implementation exceptions re-raise immediately, so production composition
  cannot silently route around programmer defects.
- Hosted OpenAI normalized results must contain a string answer, a valid source
  list, and at least one non-empty answer/source. Empty or malformed output is a
  permanent provider-local failure and falls through to the next configured
  provider without retry or circuit accounting.
- A permanent provider-local failure resets the provider's transient streak and
  cooldown state. Only consecutive exhausted transient failures can open the
  circuit.

Strict RED evidence:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_mcp.py::test_settings_chain_propagates_untyped_openai_implementation_failure tests/test_mcp.py::test_hosted_openai_empty_result_falls_through_to_healthy_provider tests/test_search.py::test_permanent_provider_failure_resets_transient_circuit_streak -q`
  failed all three regressions: the implementation defect fell through to
  Serper, empty hosted output returned successfully, and the stale transient
  count opened the circuit after an intervening permanent failure.

GREEN verification:

- The same three-test command passed: 3 tests.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_search.py tests/test_mcp.py tests/test_openai_provider.py -q` passed: 139 tests.

No network, paid-model, or GAIA calls were made. Production model and reasoning
configuration remain `gpt-5.6-luna` with `reasoning_effort=max`; scoring and
public MCP contracts were unchanged. The requested structural-key trace
allowlisting minor remains deferred to the project ledger.
