# Public Agent Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver a publicly accessible, authenticated website that runs the existing research agent.

**Architecture:** Sites hosts a Vinext chat UI and authenticated proxy. A Python HTTP service owns
conversations, uploads, durable run state and bounded agent workers; each run constructs its own
agents and MCP subprocess. Prefer the available Modal CLI for Python hosting after confirming
account access, durable storage and supported deployment behavior.

**Tech Stack:** Existing Python/Pydantic/LangGraph/MCP, FastAPI with Uvicorn, SQLite, Modal,
Sites Vinext/TypeScript, native ChatGPT sign-in.

**Spec:** `docs/superpowers/specs/2026-10-14-public-agent-web-design.md`

## Global Constraints

- Agent and research/computation calls: `gpt-6.1-sol`, reasoning `low`; specialized transcription retained.
- One active run per user; three new runs per rolling hour; 500 seconds and 24 logical tool calls per run.
- Public website, authenticated execution, server-side ownership checks for every private resource.
- Keep API keys, research content and uploads out of client bundles and public telemetry.
- Preserve existing benchmark behavior and historical results; interactive tasks need no HF token.
- No simulated answers or progress when the real service is unreachable.

## Review Focus

- Reconnect/repeated submissions must not duplicate a paid run: Task 2 idempotency test.
- Another user's resource ID must not reveal data: Task 3 ownership tests.
- Cancellation/restart must release quotas and terminate active work: Tasks 2 and 3 tests.
- Empty/oversized/unsupported uploads must fail before model execution: Task 3 upload tests.
- HTML or unsafe links in model evidence must not execute in the browser: Task 4 rendering tests.

## File Structure

Create `src/gaia_agent/interactive/` with `models.py` (contracts), `composition.py` (agent factory),
`store.py` (durable ownership/state), `service.py` (lifecycle), and `api.py` (HTTP boundaries).
Create corresponding tests under `tests/test_interactive_*.py`.
Create `sites/agent-web/` using the official Sites Vinext starter; routes and components live there.
Create `deploy/modal_agent.py` and `docs/web-deployment.md` for backend deployment.
Modify `runner.py` only to extract reusable composition where necessary, and `pyproject.toml`/lock
for required web dependencies. Preserve the uncommitted response-repair and model-migration work.

### Task 1: Interactive execution composition

**Files:** `interactive/models.py`, `interactive/composition.py`, `runner.py`, `tests/test_interactive_composition.py`.

**Interfaces:** `InteractiveRequest(question: str, attachment_id: str | None)`;
`InteractiveResult(answer: str, evidence: list[Evidence], usage: Usage)`;
`execute_interactive(request, config, workspace, history, on_progress) -> InteractiveResult` (async).
History consists of bounded user/answer messages, never raw tool transcripts. Progress callbacks
receive stage enums and counts. Capture successful evidence through an explicit orchestrator result
accessor; do not fetch it by inspecting private runtime attributes.

- [ ] Write `test_interactive_without_hf_or_reference`, `test_each_run_has_distinct_agents`,
  `test_followup_context_is_bounded`, and `test_failed_run_preserves_safe_progress` using an injected provider.
- [ ] Run `uv run pytest tests/test_interactive_composition.py`; confirm missing behavior fails.
- [ ] Implement interactive-only configuration and composition; retain existing GAIA task adaptation
  internally without exposing reference answers or invoking scoring/downloads.
- [ ] Run new tests and the existing runner/agent tests; confirm pass.
- [ ] Commit only Task 1 files after preserving the earlier local work in a separate commit.

### Task 2: Durable conversation and run lifecycle

**Files:** `interactive/store.py`, `interactive/service.py`, `tests/test_interactive_service.py`.

**Interfaces:** `ConversationStore.create(owner_id)`, `get(owner_id, conversation_id)`,
`list(owner_id)`; `RunService.start(owner_id, conversation_id, request, idempotency_key)`,
`get(owner_id, run_id)`, `cancel(owner_id, run_id)`.
Run states: `queued`, `running`, `completed`, `failed`, `cancelled`, `interrupted`.

- [ ] Write tests for ownership, duplicate idempotency keys, rolling-hour quotas, simultaneous
  submissions, cancellation, restart recovery and bounded histories.
- [ ] Run service tests; confirm fail before implementation.
- [ ] Implement transactional SQLite state, atomic quota reservation, a maximum of two global workers,
  a maximum of ten queued runs, and explicit saturation errors. Mark unfinished runs interrupted
  on startup. Release active-run reservations in finally; enforce 500-second deadlines.
- [ ] Verify tests under concurrent requests and confirm no cross-user result reuse.
- [ ] Commit the service deliverable.

### Task 3: Authenticated Python HTTP API and file handling

**Files:** `interactive/api.py`, `tests/test_interactive_api.py`, `pyproject.toml`, `uv.lock`.

**Interfaces:** `/health`; `/v1/conversations` GET/POST; `/v1/conversations/{id}/uploads` POST;
`/v1/conversations/{id}/runs` POST; `/v1/runs/{id}` GET; `/v1/runs/{id}/cancel` POST.
The proxy supplies a backend service bearer credential and an authenticated owner ID; the Python
service trusts that identity only after validating the service credential. Never expose that
credential to browsers. Return typed errors with HTTP 401/404/409/413/415/429/503 as appropriate.

- [ ] Write API tests for invalid service credentials, missing identity, cross-user IDs, duplicate
  submission, quotas, uploads and cancellation.
- [ ] Run API tests and confirm fail.
- [ ] Implement FastAPI routes, streaming upload size enforcement (32 MiB total per run), server-created
  filenames, media-type validation and attachment registration. Initially support one attachment
  per run because the current task model has one attachment path. Reject extra files explicitly.
- [ ] Implement seven-day upload retention cleanup and cleanup of cancelled run temporary files.
- [ ] Verify API tests, privacy tests and full Python suite; commit.

### Task 4: Public chat website and authenticated proxy

**Files:** `sites/agent-web/` starter, `app/page.tsx`, `app/api/agent/[...path]/route.ts`,
`app/components/AgentChat.tsx`, relevant starter configuration and UI tests.

**Interfaces:** Use the Python Task 3 routes exclusively via same-origin proxy. Native Sites
`getChatGPTUser()` supplies ownership; the server adds the service credential. Unauthenticated
visitors see a public introduction and top-level sign-in link; API requests without identity return 401.

- [ ] Initialize the official starter using Sites helpers and register a private staging Site once.
- [ ] Write route/UI tests for missing identity, identity spoofing, upstream failure, reconnect polling,
  cancellation, keyboard submission, unsafe answer markup/URLs and one-file upload behavior.
- [ ] Implement a responsive chat layout, conversation navigation, composer, status timeline, answer
  and evidence cards. Poll active runs every two seconds, stop at terminal states, and avoid
  duplicate requests on rerender. Render text safely and permit only HTTP(S) source links.
- [ ] Build and run frontend tests; verify layout and flows in local preview using supported tools.
- [ ] Commit the web deliverable without credentials or deployment artifacts.

### Task 5: Backend deployment and real staging integration

**Files:** `deploy/modal_agent.py`, `docs/web-deployment.md`, `.env.example`, Site runtime configuration.

**Interfaces:** Runtime settings `OPENAI_API_KEY`, `AGENT_SERVICE_TOKEN`, `AGENT_BACKEND_URL`,
`AGENT_DATA_DIR`. The Site proxy requires only service token/backend URL; OpenAI key belongs to Python.

- [ ] Inspect supported Modal account/deployment capabilities without displaying credentials. Confirm
  persistence semantics before using SQLite: one backend container and exclusive DB writer are
  mandatory; do not share writable SQLite across multiple replicas. If unavailable, report the
  specific missing host access while completing local software and deployment documentation.
- [ ] Implement and validate a single-container async backend with durable volume, readiness and
  secrets. Configure native Sites secrets via supported tools; never put secrets into source or CLI args.
- [ ] Deploy Python, configure the staging proxy and run a controlled live question through the
  existing agent. Verify progress, result, ownership, cancellation and reconnect behavior.
- [ ] Run full Python suite, frontend checks/build and source/secret checks; document verified results.

### Task 6: Publication and handoff

**Files:** `README.md`, `docs/web-deployment.md`, Sites hosting manifest.

- [ ] Confirm the real staging backend is reachable and signed-in users can complete a task.
- [ ] Package and publish through Sites' source/version workflow, then change audience to public
  using the native supported access tool, retaining authenticated execution.
- [ ] Verify terminal deployment success, public landing page and protected API behavior.
- [ ] Deliver the deployed URL, instructions, verified limits and any operational dependencies.
  Update project docs to separate web behavior from the historical benchmark score.

## Execution Review

All spec sections map to tasks: composition/answers to Task 1; persistence, limits and follow-ups
to Task 2; uploads and ownership to Task 3; UI/login to Task 4; Python hosting/secrets to Task 5;
public delivery to Task 6. Native execution is recommended because tasks share tightly coupled
contracts; use one independent whole-change review before public publication.
