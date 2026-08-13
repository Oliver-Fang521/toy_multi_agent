# Langfuse Reasoning Summary Design

## Goal

Make the GAIA agent's decision path inspectable in Langfuse without sending
hidden chain-of-thought or task data. The trace should explain which route ran,
what bounded stages did, and why submission was or was not reached.

## Visible summary

Each Planner, Specialist, Critic, Submit, and provider-generation observation
may emit a compact `reasoning_summary` under Langfuse `output`. The summary is
restricted to stage-safe fields:

- route or specialist enum
- step and evidence counts/types
- bounded tool names
- candidate-present boolean
- critic verdict/decision enum
- repair/fallback booleans
- parsed/refused/tool-call counts for provider generations

The summary never includes the question, reference answer, candidate text,
feedback text, URLs, attachment content, raw tool arguments/results, prompts,
or model response text. Unknown keys and non-structural values are dropped by
the observability adapter.

## Lifecycle

`Span.set_summary()` stores a pending summary for the current observation and
the context manager publishes it when the stage exits. The orchestrator adds
summaries after each stage has a validated result, while providers add only
parse/tool-call status. Existing audit events and GAIA scoring remain unchanged.

## Verification

Tests cover allowlisted summary fields, rejection of sensitive/unknown fields,
stage wiring, and provider summaries. Full tests, Ruff, and package build must
remain green.
