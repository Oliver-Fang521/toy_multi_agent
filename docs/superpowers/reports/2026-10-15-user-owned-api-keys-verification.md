# User-owned API keys — local implementation and verification

Date: 2026-10-15. Base: `ce03055`. Worktree: `/Users/HPOMEN/.codex/worktrees/public-agent-web/agent`.
Spec: `../specs/2026-10-15-user-owned-api-keys-design.md`.
Plan: `../plans/2026-10-15-user-owned-api-keys.md`.

## Implemented

- AES-256-GCM owner/provider/version-bound credential vault, fresh nonce, explicit consent,
  masked status, atomic replacement, deletion and version rotation.
- Owner-bound temporary in-memory sessions: 30 minutes, one per owner, 100 globally, generation
  invalidation and dispatch-time expiry checks. No browser key/handle persistence.
- Authenticated key status/save/delete/session routes with bounded pre-parser bodies, no input
  echoes, identity derived only from trusted proxy, mutation-time admission block and cancellation.
- Public startup without operator model/search keys. Explicit user key/endpoint in independent
  parent clients and authoritative minimal MCP environment. No process environment mutation.
- Idempotent lookup before credential resolution/quota charging; completed retry needs no key.
- Safe provider errors before orchestration/notebooks, actual injected-secret tool redaction,
  including nested dictionary keys. Unsupported providers/accounts do not use owner credentials.
- Settings UI: hostname shown before input, password fields, unchecked consent, temporary/save,
  optional search keys, masked metadata, replace/delete, cleared inputs and Chinese disclosures.
- Infrastructure-only deployment helper: service token, external encryption keyring/version,
  approved HTTPS model endpoint. Rejects local env paths and operator credential fields.

## Verification performed

| Check | Result |
| --- | --- |
| Baseline before implementation | 695 passed in 49.65s |
| Final full Python suite, `.venv/bin/pytest -o addopts='' -q -p no:cacheprovider` | 726 passed in 55.10s |
| MCP/provider/search/legacy focused suite | 172 passed in 17.31s, also included in full suite |
| Frontend `node --experimental-strip-types --test tests/*.test.ts` | 12 passed |
| TypeScript `tsc --noEmit --incremental false` | Passed |
| `npm run build`, after removing local mock environment | Passed |
| Ruff `check src tests deploy --no-cache` | Passed |
| `git diff --check` | Passed |
| DB/WAL full synthetic-key absence, captured-log privacy tests | Passed |
| Built client scan for synthetic credentials/local preview token | No matches |

New behavior was developed with observed failing tests, then passing focused runs. Final inline
review found and fixed literal-consent coercion and nested dictionary-key redaction; both were
reproduced RED before the correction and followed by a green full suite. Local browser testing also
found Worker fetch rejects `redirect: error`; the proxy now uses `manual`, rejects 3xx without
forwarding Location, and has a regression test. No temporary diagnostics remain in source.

Browser QA used only the native template's local mock sign-in, synthetic invalid keys, and a local
fake executor returning `MOCK UI TEST ONLY — no provider called.` It verified default unchecked,
hostname display, save/clear/mask, reopen empty, delete then reload unconfigured, old history
readable without a key, new task blocked without a key, mobile layout and Escape close. This is
NOT proof of live sign-in or real provider availability. Temporary mock configuration/helper and
the newly started mock servers were removed/stopped. Screenshot:
`/private/tmp/toy-agent-byok-settings-20261015.png`.

## Review Focus checked inline

1. Alice/Bob/owner credentials: parallel parent clients and explicit child environments remain
   user-scoped; model org/project and parent HTTP transport avoid inherited configuration.
2. Queued expiry/replacement/deletion: resolve after worker acquisition and before client creation;
   generations/mutation barrier stop new dispatch, deletion awaits user cancellation.
3. Lost start response: original same-payload run returned before requiring any credential again.
4. Nested upstream failures: safe wrapper precedes notebook; results/progress/errors omit plaintext.
5. Delete/reload/other device: persistent status becomes absent and temporary handles are invalidated;
   actual cloud snapshots remain an explicitly blocked release condition.

Review was by the implementer, as the plan explicitly forbids subagents here. No independent
security review was performed in this turn. Public release still requires one.

## Decisions / rulings and residual costs

- Follow the plan's no-subagents/local-only instruction instead of generic review dispatch.
  Cost: no fresh independent reviewer in this implementation session.
- Introduce `cancel_owner` during the API task and infrastructure cipher loading during runtime
  isolation because downstream contracts require them. Cost: file-map changes cross task commits;
  security scope and acceptance tests are unchanged.
- Reuse existing explicit provider/tools-factory injection in composition instead of changing
  unrelated orchestration. Cost: future public code must continue using those injection points.
- Synthetic literals intentionally exist in test fixtures; absence checks target DB/WAL/logs and
  browser artifacts. Cost: these checks are not a generic repository-wide real-secret scanner;
  only explicit source paths were staged, no real credential was read or transmitted.
- Leave cloud/provider/real-login/snapshot behavior unverified and block publishing accordingly.
  Cost: this is not yet an operational public service.

Deferred minor: Vite dev HMR emitted multiple-renderer context warnings. Mock flows, TypeScript and
production build passed; actual production warning behavior remains unverified. No framework
replacement or unrelated dependency upgrade was attempted.

## Commits / handoff

- `4ba2ee9`: encrypted vault.
- `fdc9390`: owner-bound temporary sessions.
- `c6071fc`: credential API/proxy boundary.
- `633457b`: per-user runtime isolation.
- `3f10b75`: consent/settings UI and Worker proxy compatibility.
- Final task: `chore: secure BYOK deployment and document verification` (commit containing this report).

No merge, push, cloud deployment, real-key transfer, paid smoke test or benchmark rerun performed.
Do not use the historical GAIA score as a result of this change. Preserve the managed worktree.
Before publishing, complete the release checklist in `../../web-deployment.md`: authorization,
infrastructure secrets, HTTPS/native sign-in, approved endpoint/model availability, real paid smoke
test, independent security review, real logging policy, and actual snapshot retention/deletion policy.
