# GAIA Agent Evaluation Design

## Purpose

Build a reproducible Python multi-agent system that answers a representative
subset of the GAIA 2023 validation benchmark and reports its measured
performance. Its architecture follows the four-plane design documented by
[`adorosario/customgpt-agent`](https://github.com/adorosario/customgpt-agent): a
provider-agnostic orchestrator, specialist agents, a shared tool layer, and a
quality/verification plane. The implementation must perform genuine
tool-assisted work, support benchmark attachments, and use GAIA's official
answer-normalization semantics.

## Scope

The first complete evaluation uses 15 validation questions: five from each of
GAIA levels 1, 2, and 3, selected deterministically with seed 42. The model is
configurable. Version 1 supplies an OpenAI provider through the Responses API,
but every agent depends on a provider protocol rather than the OpenAI client
directly. A later provider can therefore be added without changing the
orchestrator, specialists, tools, or evaluator.

The reference system advertises proprietary RAG, roughly 80 MCP tools, several
model vendors, and hosted E2B infrastructure. Recreating those services is not
required. Version 1 preserves the reference system's control-plane boundaries
and verification behavior while implementing the smallest GAIA-relevant shared
tool set: web search/page retrieval, Python computation, attachment inspection,
image understanding, and final-answer submission.

The project does not submit results to the private GAIA leaderboard, train or
fine-tune a model, provide a graphical interface, implement CustomGPT.ai RAG,
claim parity with the reference system's published score, or republish GAIA
questions, attachments, or reference answers.

## User Interface

The package exposes a `gaia-agent` command. The primary workflow is:

```console
gaia-agent evaluate --sample-size 15 --seed 42 --model MODEL
```

Configuration can come from command-line flags or environment variables. The
required secrets are `OPENAI_API_KEY` and `HF_TOKEN`. Command-line values take
precedence over non-secret environment defaults. Secrets must never be written
to logs or result artifacts.

The evaluator also supports `--resume` to continue an interrupted run and
`--output-dir` to choose where artifacts are written. A sample size must be at
least three and divisible by three because the evaluation is evenly stratified
across all GAIA difficulty levels.

## Architecture

The implementation mirrors the reference architecture with four planes.

### 1. Provider-Agnostic Orchestrator

The orchestrator owns a bounded `Plan -> Act -> Verify -> Submit` state machine.
It receives a task, requests a structured plan, dispatches each planned step to
one specialist at a time, accumulates observable evidence in task state, asks
the critic to verify the candidate answer, and either submits it or performs at
most one targeted correction pass. It never calls a vendor SDK directly.

`ModelProvider` is the boundary for model generation, tool schemas, attachment
inputs, token usage, and citations. `OpenAIProvider` is the initial adapter. The
configured reasoning effort is passed through only when the provider supports
it.

### 2. Specialist Agents

All specialists share the same provider interface and task state but have
separate system prompts and tool allowlists:

- `PlannerAgent` restates the goal and emits a short JSON step queue. It has no
  external tools and cannot answer the task.
- `LookupAgent` searches and reads web sources, prioritizing primary sources and
  returning evidence records with URLs.
- `ComputationAgent` executes Python for arithmetic, data transformation, and
  independent double-checking of numerical results.
- `FileAnalysisAgent` inspects GAIA attachments such as PDFs, spreadsheets,
  images, and audio through code/file tools.
- `VisualAgent` analyzes images or browser screenshots when a task requires
  visual interpretation. Version 1 does not automate interactive headed
  browsing; its interface leaves that addition isolated.
- `CriticAgent` receives the question, candidate answer, and evidence, then
  independently returns `approve`, `revise`, or `insufficient_evidence`. It can
  use lookup and computation tools to check claims rather than relying only on
  self-reflection.

Specialists return structured `AgentResult` values, never mutate orchestration
state directly, and cannot call one another.

### 3. Shared MCP Tool Layer

A local stdio MCP server exposes JSON-schema tool definitions and asynchronous
handlers. `MCPToolClient` discovers those schemas at startup, filters them by
the active specialist's allowlist, sends them through `ModelProvider`, and
executes requested calls through the server. The initial server contains:

- `web_search(query)` and `read_webpage(url)`;
- `execute_python(code, attachment_ids)` through a sandboxed code backend;
- `read_attachment(path)` for safe type-aware extraction;
- `submit_answer(answer, evidence_ids)` as the only terminal action.

`VisualAgent` sends task-contained image inputs through `ModelProvider` because
vision is a model capability rather than a deterministic tool. The web-search
MCP handler uses a `SearchBackend` protocol whose initial implementation uses
OpenAI web search. The Python handler uses a `CodeBackend` protocol whose
initial implementation uses an ephemeral OpenAI Code Interpreter container.
These provider-specific backends remain behind the MCP boundary and can later
be replaced by E2B or another sandbox.

Attachment access is restricted to files registered for the current task,
backend calls have hard timeouts, outputs are size-limited, and secrets are
redacted from audit events. No model-generated code runs on the host. The mocked
end-to-end suite starts the real stdio server so the MCP transport, schema
discovery, allowlists, and error mapping are tested rather than assumed.

### 4. Quality and Verification Plane

Cross-cutting hooks enforce per-agent and global tool budgets, detect repeated
identical tool calls, validate every structured result, and append an immutable
JSONL audit event. Numerical answers require either two independent
computations or computation plus source evidence. Research answers require at
least one source record. The critic must approve before submission; otherwise
the orchestrator may run one correction pass and then must submit its best
answer or a categorized failure.

Optional self-consistency repeats the complete task run an odd number of times
and selects the majority answer after GAIA normalization, with early stopping
once a majority is mathematically guaranteed. The default evaluation uses one
run per task to keep cost measurable; voting is an explicit CLI option.

### Package Boundaries

The Python package contains focused modules:

- `config` defines validated run configuration and limits;
- `models` defines task, plan, evidence, result, audit, and report value types;
- `providers` defines `ModelProvider` and the OpenAI adapter;
- `mcp_server` defines the MCP server, search backend, safety policy, and tool
  handlers;
- `mcp_client` owns server lifecycle, schema discovery, allowlists, and calls;
- `agents` defines planner, lookup, computation, file, visual, and critic roles;
- `orchestrator` implements the state machine and correction/voting loops;
- `dataset` loads GAIA and performs deterministic stratified sampling;
- `scoring` implements GAIA's quasi-exact-match scorer;
- `runner` handles tasks, retries, checkpoints, and resume;
- `reporting` calculates metrics and writes artifacts;
- `cli` connects configuration, execution, scoring, and reporting.

Dependencies point inward from CLI and runner to orchestration interfaces.
Scoring and sampling remain pure functions and do not depend on model providers.

## Agent Behavior

For each task, the planner receives the original GAIA question and attachment
metadata. The orchestrator dispatches only the minimum specialists required by
the plan, maintaining a durable `plan.json`, `notes.jsonl`, and `audit.jsonl` in
the run directory. Specialists see only the task context, evidence, and tools
needed for their role.

The final synthesis instructions incorporate GAIA's official answer contract:
return a number, as few words as possible, or an ordered comma-separated list,
without unnecessary units or prose. The concise answer is taken from a
validated structured result, with a strict `FINAL ANSWER:` parser only as a
compatibility fallback. Hidden chain-of-thought is neither requested nor
persisted. Stored traces contain plans, observable tool calls/results, source
URLs, timing, usage, verification decisions, and errors.

If an attachment must be uploaded to a model provider, it is scoped to that
task's request and its remote identifier is not included in the report.

## Evaluation Data Flow

1. Validate configuration and required credentials before incurring API cost.
2. Load `gaia-benchmark/GAIA`, configuration `2023_all`, validation split.
3. Partition records by integer `Level`, sort each partition by `task_id`, and
   select `random.Random(seed + level).sample(records, sample_size // 3)` from
   each partition (seed 42 for the first evaluation).
4. Order the selected tasks by level and task ID for stable execution.
5. Run each task through planner, routed specialists, critic, and optional
   correction; immediately append an atomic checkpoint.
6. If voting is enabled, normalize completed candidate answers and select the
   majority result before scoring.
7. Score every returned answer against `Final answer` with the official scorer.
8. Write predictions and aggregate/per-level evaluation results.

Selection is independent of model output and gold answers. The selected task
IDs are persisted so a resumed run evaluates exactly the same subset.

## Results and Privacy

Each run creates:

- `manifest.json`: model, limits, dataset revision/configuration, seed, selected
  task IDs, package versions, and start time.
- `predictions.jsonl`: task ID, level, final answer, correctness, duration,
  token usage, verification outcome, evidence IDs, and any error category.
- `summary.json`: counts, overall exact-match accuracy, accuracy by level,
  latency totals/percentiles, token totals, and failure counts.
- `report.md`: a human-readable interpretation of the same metrics and run
  configuration, including limitations.
- per-task `plan.json`, `notes.jsonl`, and `audit.jsonl` files containing only
  observable orchestration and tool events.

No artifact contains the question text, reference answer, attachment content,
credential, or hidden model reasoning. Result files are safe to commit with
respect to GAIA's redistribution restriction, though actual run artifacts are
ignored by default to avoid accidentally publishing model traces.

## Scoring

The scorer mirrors the official leaderboard implementation:

- numeric references compare after removing `$`, `%`, and comma characters and
  converting the model answer to a float;
- comma- or semicolon-separated references compare ordered lists with equal
  length, normalizing numeric elements as numbers and string elements by
  whitespace/case while retaining punctuation;
- other strings compare after removing whitespace, ASCII punctuation, and case.

`None` answers are treated as the string `"None"`. Scoring exceptions are
recorded as incorrect evaluation errors rather than terminating the run.

## Failure Handling and Resume

Missing credentials, inaccessible gated data, invalid stratification, or an
unwritable output directory fail before task execution. Rate limits, timeouts,
and server failures receive one retry with bounded exponential backoff. Invalid
requests and authentication failures do not retry.

A timeout, malformed model response, exhausted tool budget, loop detection,
critic rejection after correction, attachment error, or retry exhaustion
produces an incorrect result with a stable error category; the remaining tasks
continue. Checkpoint writes use a temporary file followed by an atomic rename.
In resume mode, configuration and selected task IDs must match the manifest,
and completed task IDs are skipped.

## Testing Strategy

Development follows test-driven development. Unit tests cover:

- official scorer behavior for numeric, list, string, punctuation, whitespace,
  and `None` cases;
- deterministic, balanced sampling and invalid sample sizes;
- provider contract behavior and structured answer extraction;
- planner schema validation and deterministic specialist routing;
- tool allowlists, path containment, timeouts, output limits, and loop detection;
- critic approval, one-pass correction, numerical double-checking, and grounded
  research requirements;
- normalized majority voting and early-stop conditions;
- retry classification, terminal task failures, and secret redaction;
- atomic checkpointing, manifest compatibility, and resume behavior;
- aggregate and per-level report calculations.

A mocked end-to-end test runs the CLI from dataset records through all output
files without network access or API spending. A credential-gated smoke test
checks one real API task. Completion requires a real 15-task run and a report
whose prediction counts and score calculations are independently rechecked from
the saved artifacts.

## Success Criteria

The project is complete when:

1. A clean environment can install the package and run its automated tests.
2. Unit and end-to-end tests prove the four architectural planes and the
   `Plan -> Act -> Verify -> Submit` state transitions.
3. The CLI can load an authenticated GAIA validation split with attachments.
4. A deterministic 15-task, evenly stratified evaluation completes or records a
   terminal result for every selected task.
5. Every submitted answer has a critic decision and an auditable evidence/tool
   trace, subject to the defined terminal-failure policy.
6. Predictions are scored with behavior matching GAIA's official scorer.
7. The summary and report accurately show overall and per-level performance,
   latency, usage, verification outcomes, and failures.
8. No committed file exposes GAIA questions, gold answers, attachments, or
   credentials.

## Assumptions and Constraints

- Python 3.11 or newer is available.
- The user has accepted the GAIA dataset terms and supplies a valid `HF_TOKEN`.
- The user supplies a valid `OPENAI_API_KEY` with access to the configured model
  and required Responses API tools.
- Network access and API usage may incur cost; limits are visible and
  configurable before the run.
- A 15-task subset provides a reproducible project evaluation, not a leaderboard
  comparable estimate for the complete 165-question validation split.
