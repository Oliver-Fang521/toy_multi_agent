# GAIA Response Tolerance Design

## Goal

Make the agent more tolerant of provider response formatting while preserving strict validation for planning, critic decisions, tool contracts, and the GAIA final-answer scorer.

## Scope

This change implements the approved items 1, 2, and 4:

1. Extract a final answer from strict JSON first, then from a `FINAL ANSWER:` marker, then from a concise plain-text fallback.
2. Allow Specialist result fields that are semantically optional (`evidence`, `candidate_answer`, and usage metadata) to be absent and fill safe defaults. Planner and Critic schemas remain strict.
3. Use a larger output budget only for `reasoning_effort="max"`; ordinary runs retain the existing 2,000-token default.

## Design

The provider returns both `parsed` and `output_text`. The agent will continue to prefer validated structured output. If `parsed` is unavailable, the final-answer boundary will attempt a conservative extraction from `output_text` and produce the smallest GAIA-compatible answer string. It will not infer answers from long prose or expose hidden reasoning.

Specialist validation will distinguish missing optional fields from malformed required fields. Pydantic defaults fill omitted fields at the internal boundary, while the outbound provider schema remains closed and fully required for OpenAI Structured Outputs; semantically optional values remain nullable/defaultable. Evidence IDs, evidence kinds, candidate normalization, and downstream critic checks remain enforced.

`ModelRequest.max_output_tokens` remains 2,000 by default. Agent request construction will select 8,000 for `reasoning_effort="max"`, giving the model room for reasoning plus the required structured response without increasing ordinary-run spend.

## Error handling

Unparseable planner or critic responses remain `malformed_response`. A Specialist response is malformed only when required identity/schema constraints fail after default filling. A final-answer extraction failure still produces a terminal malformed result rather than guessing.

## Verification

Add focused tests for marker extraction, plain-text extraction, Specialist defaults, and max-only output budgeting. Run the focused tests, full test suite, Ruff, and build.
