# GAIA Multi-Agent Evaluation

This project runs a reproducible, tool-assisted evaluation on a balanced subset of the
[GAIA 2023](https://huggingface.co/datasets/gaia-benchmark/GAIA) validation split. It is a
provider-neutral `Plan -> Act -> Verify -> Submit` system with an OpenAI production adapter.

## Install and authenticate

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```console
uv sync --extra dev
```

On the GAIA dataset page, sign in to Hugging Face, accept the gated dataset terms, and create a
Hugging Face token that can read gated repositories. Set that token and an OpenAI API key in the
shell that will run the evaluation:

```console
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
```

Do not commit these values or put them directly in command history. `.env` is ignored, but the CLI
does not load it automatically.

## Architecture

The architecture follows the four-plane boundaries of
[`adorosario/customgpt-agent`](https://github.com/adorosario/customgpt-agent), scoped to the tools
needed for GAIA:

1. A provider-agnostic orchestrator owns the bounded `Plan -> Act -> Verify -> Submit` state
   machine; only the provider adapter calls the OpenAI Responses API.
2. Six roles have separate prompts and permissions: the tool-free **planner**; **lookup**,
   **computation**, **file-analysis**, and **visual** specialists; and an independent **critic**.
3. A real local MCP v2 server runs as a separate stdio subprocess. Specialists receive only their
   allowlisted tools: hosted web search/page retrieval, hosted Python, registered attachment
   inspection, and terminal answer submission.
4. The quality plane enforces step and tool budgets, loop detection, evidence rules, critic review,
   audit events, scoring, and independently verifiable reports.

Web search uses OpenAI's hosted Web Search tool. Python and attachment computation use an ephemeral
OpenAI Code Interpreter container; model-generated code never runs on the host. These hosted tools,
model calls, file uploads, and optional repeated votes can incur API charges. Bound exposure before
a run with `--sample-size`, `--max-steps`, `--max-tool-calls`, `--timeout-seconds`, and `--votes`.
The default limits are 15 tasks, 8 steps, 24 tool calls, 300 seconds per attempt, and one vote.

## Configuration

`OPENAI_API_KEY` and `HF_TOKEN` are required and are accepted only from the environment. Non-secret
settings use the following precedence: an explicitly supplied CLI option overrides its `GAIA_*`
environment variable, which overrides the built-in default.

| CLI option | Environment variable | Default |
| --- | --- | --- |
| `--model` | `GAIA_MODEL` | `gpt-5.6-sol` |
| `--reasoning-effort` | `GAIA_REASONING_EFFORT` | `medium` |
| `--sample-size` | `GAIA_SAMPLE_SIZE` | `15` |
| `--seed` | `GAIA_SEED` | `42` |
| `--max-steps` | `GAIA_MAX_STEPS` | `8` |
| `--max-tool-calls` | `GAIA_MAX_TOOL_CALLS` | `24` |
| `--timeout-seconds` | `GAIA_TIMEOUT_SECONDS` | `300` |
| `--votes` | `GAIA_VOTES` | `1` |
| `--output-dir` | `GAIA_OUTPUT_DIR` | `results/gaia-15-seed42` |
| `--resume` / `--no-resume` | `GAIA_RESUME` | `false` |

The sample size must be at least three and divisible by three so the run contains equal counts from
GAIA levels 1, 2, and 3. Votes must be a positive odd number. Run `uv run gaia-agent evaluate
--help` for the current CLI surface.

## Run and verify

A bounded three-task smoke run selects one task per level and limits each attempt to one planned
step:

```console
GAIA_MAX_STEPS=1 uv run gaia-agent evaluate --sample-size 3 --seed 42 --model gpt-5.6-sol --output-dir results/smoke-3
uv run gaia-agent verify-results results/smoke-3
```

Run the deterministic 15-task evaluation and independently verify its public artifacts:

```console
uv run gaia-agent evaluate --sample-size 15 --seed 42 --model gpt-5.6-sol --output-dir results/gaia-15-seed42
uv run gaia-agent verify-results results/gaia-15-seed42
```

Each output directory contains `manifest.json`, `predictions.jsonl`, `summary.json`, `report.md`,
and per-task `plan.json` and `audit.jsonl` files when those stages ran. Artifacts retain only stable
task IDs, aggregate metrics, concise predictions, role/transition metadata, opaque evidence IDs,
and safe tool-shape metadata. They exclude GAIA question text, reference answers, attachment
content and paths, credentials, prompts, and hidden reasoning. `results/` is gitignored as an
additional safeguard.

To continue an interrupted run, repeat the same command and output directory with `--resume` (or
set `GAIA_RESUME=true`). The persisted selection and all run-defining configuration must match;
complete valid checkpoints are skipped and incomplete checkpoints are repaired. Re-run
`verify-results` after completion to check schemas, privacy allowlists, selection balance, audit
sequence integrity, prediction completeness, and recomputed summary/report metrics.

## Interpretation

The fixed seed makes this 15-task sample reproducible, but it is not statistically or procedurally
comparable to the full 165-question GAIA validation leaderboard. Report it as a measured project
evaluation, not as a leaderboard score or parity claim for the reference system.
