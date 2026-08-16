# GAIA Trace-Driven Reliability Design

## Goal

Increase end-to-end GAIA answer completion and accuracy by eliminating the failure modes observed
in the ten recorded Langfuse traces: unretried provider connection failures, discarded successful
tool work, parsed-but-invalid specialist responses, impossible attachment routes, brittle search
providers, unnecessarily long static plans, unbounded context growth, and incomplete trace outputs.

All model stages continue to use `gpt-5.6-luna` with `reasoning_effort=max`. This work does not
change GAIA's final-answer scorer, reveal reference answers, or publish hidden chain-of-thought.

## Evidence and success criteria

The ten traces contained six `APIConnectionError` terminal failures, two semantic validation
failures after successful tools, and two visual plans for tasks with no attachment. None reached the
critic. Across those tasks, 38 tool calls produced 23 successes and 15 failures while consuming
548,775 tokens.

A subsequent ten-task diagnostic run should meet these engineering targets:

- ten root traces for ten selected tasks;
- at least eight tasks reach the critic;
- no more than two terminal `answer=null` predictions;
- no more than one terminal provider-transient failure;
- zero Visual/File dispatches for tasks without attachments;
- web-tool failure rate below 15 percent when at least one configured provider is healthy;
- median input usage below 30,000 tokens per task;
- every trace records terminal status, answer presence, evidence count, retry counts, tool counts,
  usage, scorer outcome, and a bounded reasoning summary.

These are reliability targets, not a guarantee that eight answers match the GAIA reference answer.

## Approaches considered

### 1. Incremental reliability layer (selected)

Keep the existing LangGraph and agent roles. Add focused recovery at component boundaries and a
private checkpoint store for reusable tool evidence. This directly addresses the trace evidence,
keeps existing public interfaces stable, and allows isolated tests for each failure mode.

### 2. Full orchestration rewrite

Replace the current Planner/Specialist/Critic workflow with a monolithic tool-using agent. This
could reduce protocol boundaries but would remove independent verification, invalidate existing
tests, and make the trace comparison inconclusive. It is rejected.

### 3. Whole-task retries and larger budgets

Retry the entire task whenever any stage fails and increase token/tool limits. This is simple but
repeats paid Planner and tool work, worsens the observed cost problem, and does not fix impossible
routes or semantic validation. It is rejected.

## Architecture

### Provider reliability

The OpenAI provider classifies `APIConnectionError`, timeout, rate-limit, transient 5xx, and
transport errors as retryable. Each individual Responses API call receives up to four attempts with
bounded exponential backoff and deterministic injectable jitter for tests. Retrying occurs at the
failed provider continuation, not at the whole-task boundary.

If a provider transient still escapes a stage, the orchestrator preserves its typed cause and the
runner performs one bounded stage/task recovery instead of accepting a generic
`retry_exhausted` prediction as final. Audit and Langfuse metadata record attempt number, retry
reason, and whether recovery succeeded.

### Private Research Notebook and stage checkpoint

Successful MCP tool results are captured immediately in a per-task Research Notebook before the
model synthesis continuation. Entries contain the tool name, normalized arguments hash, bounded
result, source identifiers, timestamp, and structural status. The notebook supports:

- replaying successful tool outputs when the synthesis continuation is retried;
- avoiding duplicate paid calls for identical normalized arguments;
- giving a repair/fallback synthesis call the evidence already collected;
- resuming a failed stage without rerunning completed stages.

Notebook content is private runtime data. It is stored atomically in a sibling hidden directory
derived from the output directory, created with user-only permissions, excluded from public result
verification, reports, git, and Langfuse. Public audit artifacts retain only hashes, counts, tool
names, and opaque evidence IDs.

The existing task checkpoint remains the authority for completed predictions. The private notebook
is advisory: corruption or absence causes a safe cache miss, never a fabricated answer.

### Semantic Repair

Stage repair expands from JSON parsing failures to repairable semantic validation failures. When a
parsed Planner, Specialist, or Critic payload fails validation, one tool-free repair request receives
the invalid payload, a bounded validation-code enum, and the original output schema. It must
preserve valid candidate/evidence fields and correct only protocol fields.

Repairable validation codes include specialist mismatch, invalid or duplicate evidence IDs,
missing source URL when one exists in tool citations, candidate formatting, checked-evidence ID
shape, and schema field constraints. Security, attachment policy, unknown evidence provenance, and
fact disagreement are not repairable and still fail closed.

If a later planned step fails semantically but the state already has a valid candidate and evidence,
the graph records a soft step failure and proceeds to the critic. The critic decides whether the
partial result is sufficient; the orchestrator no longer discards an earlier valid candidate solely
because an optional enrichment step malformed its result.

### Capability-aware plan normalization and adaptive execution

Planner input explicitly provides `attachment_present`, attachment media category, and the allowed
specialist names for that task. After planning, deterministic normalization enforces those
capabilities:

- no attachment: rewrite Visual/File to Lookup or Computation using the existing conservative
  question cues;
- non-image attachment: rewrite Visual to File;
- provider without vision: rewrite Visual to File;
- duplicate adjacent specialists: coalesce their instructions;
- initial plan: keep at most two executable steps.

After each step, a deterministic early-stop router sends a valid candidate with usable evidence to
the critic. The critic may request one correction step for a concrete gap. Plans remain bounded by
the existing global limits, and all stages continue using Luna/max.

### Tool reliability and provider fallback

Tool failures return a structured safe envelope instead of only `{"error":"tool_failed"}`. The
envelope exposes tool, provider, retryability, public error category, HTTP status when safe, and
attempt count; it excludes URLs with secrets, response bodies, credentials, and attachment content.

`web_search` uses a provider chain assembled from available credentials:

1. OpenAI hosted web search;
2. Serper when `SERPER_API_KEY` is configured;
3. SerpAPI when `SERPAPI_API_KEY` is configured.

Each provider gets bounded retry with backoff. A per-run circuit breaker temporarily skips a
provider after two consecutive transient failures and resets it after a successful probe or cooldown.
Equivalent results are normalized into the existing `answer` and `sources` contract. A provider's
absence is not an error; the chain uses only configured providers.

`read_webpage` distinguishes HTTP, timeout, access-policy, parse, and empty-content failures. It
retries only transient transport errors and lets the specialist fall back to search-result snippets
when direct reading is unavailable.

### Context and cost controls

Reasoning effort remains `max` everywhere. Cost is controlled through data volume and execution,
not weaker reasoning:

- bound and deduplicate tool results before they enter model context;
- reduce each webpage to question-relevant excerpts with source metadata;
- hash repeated URLs and arguments and reuse notebook results;
- set per-stage input-context limits and retain the highest-value evidence cards;
- stop planning after a candidate and usable evidence exist;
- retry only the failed continuation or stage;
- emit a budget-pressure audit event before forcing synthesis.

No raw webpage body or question is added to public artifacts or Langfuse.

### Langfuse observability

The root trace remains open through local scoring. Its final safe output records:

- `answer_present`, `evidence_count`, `critic_verdict`, `error_category`, and `correct`;
- provider/stage retry counts and checkpoint restoration;
- tool success/failure counts by safe category;
- per-stage usage and latency;
- terminal route and bounded reasoning summary.

Provider generations record attempt number and retry category. Tool spans record provider, status,
latency, safe HTTP status, and circuit-breaker/fallback decisions. The application supplies a
configurable model-price mapping for Langfuse cost calculation when the service does not recognize
the model name.

Tracing is still best-effort for task scoring, but startup performs a visible configuration check.
Flush failures are returned/logged as a bounded warning instead of being silently debug-only. No
secret, prompt, reference answer, candidate text, hidden reasoning, attachment content, full URL,
or webpage body is uploaded.

## Failure handling

The recovery order is:

1. retry a transient provider/tool operation in place;
2. restore successful tool work from the Research Notebook;
3. repair a protocol/semantic payload once without tools;
4. continue to Critic with an earlier valid candidate/evidence when an enrichment step failed;
5. execute one critic-directed correction;
6. emit a precise terminal category only when no scorer-eligible candidate can be safely submitted.

Attachment security violations, unregistered paths, symlink/path escapes, oversized files, and
unknown evidence provenance never use degraded submission.

## Testing strategy

Tests are written before implementation and cover:

- `APIConnectionError` retry at initial and continuation calls, backoff, and exhaustion;
- no duplicate tool execution after continuation retry;
- private notebook atomic persistence, permission mode, cache hit/miss, corruption, and public
  artifact exclusion;
- semantic repair success, repair exhaustion, and non-repairable security validation;
- partial-candidate continuation to Critic after optional step failure;
- no-attachment and non-image plan normalization;
- early stop after candidate plus evidence and critic-requested correction;
- Serper/SerpAPI normalization, fallback order, structured errors, circuit breaker, and missing keys;
- webpage retry/fallback categories;
- bounded context selection and duplicate evidence removal;
- root trace closure after scorer result with retry/tool/usage metadata;
- unchanged GAIA scoring, attachment security, public-artifact privacy, and Luna/max configuration.

Focused suites run after each component, followed by the full Pytest suite, Ruff, package build, an
offline deterministic trace simulation, and only then an explicitly authorized paid GAIA run.

## Non-goals

- Do not lower reasoning effort or switch any agent stage away from `gpt-5.6-luna`.
- Do not upload chain-of-thought or GAIA reference answers.
- Do not weaken attachment/path security or evidence provenance.
- Do not change the GAIA official scorer or claim leaderboard comparability.
- Do not automatically launch a paid benchmark run as part of code verification.
