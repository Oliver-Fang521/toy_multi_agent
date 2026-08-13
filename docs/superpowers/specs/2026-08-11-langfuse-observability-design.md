# Langfuse Observability Design

## Goal

Add optional Langfuse tracing to the GAIA agent so each evaluation task and each agent/tool stage can be inspected without changing scoring behavior or leaking gated data.

## Scope

- Trace one evaluation task from planning through submission or terminal failure.
- Record Planner, Specialist, Critic, provider generation, and MCP tool-call timing and status.
- Record safe identifiers, model configuration, usage totals, error categories, and HTTP status codes when available.
- Keep tracing disabled as a no-op when Langfuse credentials are absent or `LANGFUSE_ENABLED=false`.
- Never send GAIA questions, reference answers, attachment bytes/text, complete prompts, model output text, or raw tool arguments/results to Langfuse.

## Design

Create a small provider-neutral observability adapter in `src/gaia_agent/observability.py`. The adapter exposes task traces and nested spans/generations through a narrow interface and owns flushing/shutdown. A real Langfuse implementation is selected only when both public and secret keys are configured; otherwise a no-op implementation is used so tests and local runs remain offline-safe.

The runner creates the task trace with a stable task ID, level, model, reasoning effort, run index, and evaluation mode. Orchestrator stages create child spans with stage names, specialist names, counts, durations, statuses, and sanitized exception types. Provider calls record model, reasoning effort, token usage, and whether the response was parsed/refused; they do not record request or response bodies. MCP calls record tool name, success/failure, call number, and safe status metadata; they do not record arguments or outputs.

All trace calls are best-effort. Langfuse transport errors are swallowed and surfaced only through local debug logging so observability cannot alter GAIA predictions. The adapter flushes once after each task and once at runner completion, with a bounded shutdown path.

## Configuration and security

Supported environment variables:

- `LANGFUSE_ENABLED` (default: enabled only when both keys are present)
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL` (default: `https://cloud.langfuse.com`)
- `LANGFUSE_PROJECT_NAME` (optional metadata)

Credentials are read only from the environment. They are never stored in reports, audit logs, trace metadata, or source files. Existing `.env` values are used locally but are not copied into `.env.example` or committed files.

## Verification

Tests will cover no-op behavior without credentials, sanitized metadata, span/generation lifecycle, provider usage recording, tool error recording, and flush failure isolation. Existing full tests, Ruff, and package build must remain green.
