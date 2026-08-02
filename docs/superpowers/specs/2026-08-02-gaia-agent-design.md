# GAIA Agent Evaluation Design

## Purpose

Build a reproducible Python agent that answers a representative subset of the
GAIA 2023 validation benchmark and reports its measured performance. The agent
must perform genuine tool-assisted work, support benchmark attachments, and use
GAIA's official answer-normalization semantics.

## Scope

The first complete evaluation uses 15 validation questions: five from each of
GAIA levels 1, 2, and 3, selected deterministically with seed 42. The model is
configurable and accessed through the OpenAI Responses API. The implementation
supports web search and hosted code/file analysis so a single API credential is
sufficient after the gated dataset has been downloaded with a Hugging Face
credential.

The project does not submit results to the private GAIA leaderboard, train or
fine-tune a model, provide a graphical interface, or republish GAIA questions,
attachments, or reference answers.

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

The implementation is a small Python package with focused modules:

- `config` defines validated run configuration and limits.
- `dataset` downloads/loads the gated GAIA 2023 validation split, resolves
  attachment paths, and performs deterministic stratified sampling.
- `agent` builds the benchmark prompt, attaches task files, invokes the OpenAI
  Responses API with web-search and code-analysis tools, and extracts one final
  answer.
- `scoring` implements the official GAIA quasi-exact-match scorer.
- `runner` executes tasks independently, retries transient failures, records
  checkpoints, and resumes completed work.
- `reporting` calculates aggregate and per-level metrics and writes durable
  artifacts.
- `cli` connects configuration, dataset loading, execution, and reporting.

Dependencies point inward: the CLI uses the runner; the runner coordinates
dataset records, the agent interface, scoring, and reporting. Scoring and
sampling remain pure functions and do not depend on API clients.

## Agent Behavior

For each task, the agent receives the original GAIA question and, when present,
its attachment. Its instructions incorporate GAIA's official answer contract:
return a number, as few words as possible, or an ordered comma-separated list,
without unnecessary units or prose.

The OpenAI Responses request enables web search and hosted code/file analysis.
The implementation owns the evaluation loop, limits, retries, answer
extraction, and persistence rather than delegating benchmark execution to an
agent framework. Each task has configurable maximum duration, tool calls, and
output tokens. The model's concise final answer is taken from structured output
when available, with a strict `FINAL ANSWER:` parser as a compatibility
fallback. Hidden chain-of-thought is neither requested nor persisted. The
stored trace contains only observable tool events, citations returned by the
API, timing, usage, and errors.

If an attachment must be uploaded, the uploaded file is scoped to that task's
request and its identifier is not included in the public report.

## Evaluation Data Flow

1. Validate configuration and required credentials before incurring API cost.
2. Load `gaia-benchmark/GAIA`, configuration `2023_all`, validation split.
3. Partition records by integer `Level`, sort each partition by `task_id`, and
   select `random.Random(seed + level).sample(records, sample_size // 3)` from
   each partition (seed 42 for the first evaluation).
4. Order the selected tasks by level and task ID for stable execution.
5. Run each task through the agent and immediately append an atomic checkpoint.
6. Score every returned answer against `Final answer` with the official scorer.
7. Write predictions and aggregate/per-level evaluation results.

Selection is independent of model output and gold answers. The selected task
IDs are persisted so a resumed run evaluates exactly the same subset.

## Results and Privacy

Each run creates:

- `manifest.json`: model, limits, dataset revision/configuration, seed, selected
  task IDs, package versions, and start time.
- `predictions.jsonl`: task ID, level, final answer, correctness, duration,
  token usage, observable tool trace, and any error category.
- `summary.json`: counts, overall exact-match accuracy, accuracy by level,
  latency totals/percentiles, token totals, and failure counts.
- `report.md`: a human-readable interpretation of the same metrics and run
  configuration, including limitations.

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

A timeout, malformed model response, exhausted tool budget, attachment error,
or retry exhaustion produces an incorrect result with a stable error category;
the remaining tasks continue. Checkpoint writes use a temporary file followed
by an atomic rename. In resume mode, configuration and selected task IDs must
match the manifest, and completed task IDs are skipped.

## Testing Strategy

Development follows test-driven development. Unit tests cover:

- official scorer behavior for numeric, list, string, punctuation, whitespace,
  and `None` cases;
- deterministic, balanced sampling and invalid sample sizes;
- structured answer extraction and strict fallback parsing;
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
2. The CLI can load an authenticated GAIA validation split with attachments.
3. A deterministic 15-task, evenly stratified evaluation completes or records a
   terminal result for every selected task.
4. Predictions are scored with behavior matching GAIA's official scorer.
5. The summary and report accurately show overall and per-level performance,
   latency, usage, and failures.
6. No committed file exposes GAIA questions, gold answers, attachments, or
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
