# User-Owned API Keys Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This side conversation prohibits subagents; do not dispatch implementation or review agents here.

**Goal:** Public web users pay with their own API credentials and can explicitly opt into encrypted, account-scoped credential storage, replacement and deletion.

**Architecture:** Introduce an authenticated credential vault and short-lived in-memory credential sessions. Resolve a user's credentials at the execution boundary, construct isolated model/tool clients, and never consult owner credentials in the public runtime. Reuse existing Sites identity, server proxy, SQLite and run lifecycle.

**Tech Stack:** Python, FastAPI, SQLite, Pydantic SecretStr, cryptography AESGCM; existing React/TypeScript Sites app and Node test runner.

**Spec:** `docs/superpowers/specs/2026-10-15-user-owned-api-keys-design.md` (approved 2026-10-15).

## Global Constraints

- No owner model/search key fallback, no process-wide environment mutation, no real-key upload or live paid call during local implementation.
- Saved keys require explicit consent; AES-256-GCM, fresh nonce, owner/provider/version-bound associated data; master keys live outside SQLite.
- Temporary credential handles are owner-bound and expire after 30 minutes; expiry is checked when execution begins, while active runs retain their snapshot only through their 500-second execution deadline.
- Credential values never enter chat, InteractiveRequest, run records, notebook, logs, analytics, URL, browser persistence or response errors.
- Metadata exposes only provider, configured flag, last four characters and timestamps. No plaintext read/export endpoint.
- Preserve gpt-6.1-sol / low as configured preference, not an assertion of API availability. Unsupported account/model fails without fallback.
- Single administrator-configured HTTPS model endpoint; no browser-controlled base_url. Show its hostname before accepting a key.
- Separate search keys are optional, user-owned, and disabled when absent; provider-hosted tools still use the user's model credential.
- Preserve existing 1 active run/user, 3 runs/hour, 2 workers, 10 queued, 500 seconds/run and 24 tool-call budget.
- Keep CLI/benchmark behavior backward compatible; do not relabel or rerun historical GAIA results as part of this change.
- Deletion cancels user's queued/running tasks and removes live credentials. Do not promise cancellation of already sent provider requests or immediate erasure of infrastructure snapshots.

## Review Focus

1. Two users run concurrently while owner keys exist in the environment: every paid client and subprocess must use only its user's credentials (Task 4).
2. An accepted run waits in the queue while its key expires, is replaced or is deleted: invalidated credentials cannot start new paid work (Tasks 2–4).
3. Lost start response followed by an idempotent retry: return the original run without requiring credentials to be resubmitted or creating a second paid execution (Task 4).
4. Upstream authentication/transport failures echo keys in nested errors: public errors, progress, logs and notebook must never expose plaintext (Tasks 3–4).
5. A user deletes a key then reloads or logs in on another device: the UI reports unconfigured; browser persistence and backups must not silently restore usable credentials (Tasks 5–6).

## Workspace and File Map

Read the spec and inspect current Git state before execution. Existing web implementation was inspected at
`/Users/HPOMEN/.codex/worktrees/public-agent-web/agent`, commit `ce03055`; this is a context pointer, not permission to reset or overwrite it.
The spec and plan were written in `/Users/HPOMEN/Documents/agent` without modifying the parent task's Git state.
At execution time, use the git-worktrees skill to resolve an appropriate isolated checkout containing both the web implementation and these documents; preserve unrelated edits. Do not silently implement against the older main checkout which lacks the web modules.

New Python modules under `src/gaia_agent/interactive/`: `credential_crypto.py` (AEAD only), `credential_vault.py` (persistence), `credential_sessions.py` (memory handles/generations), `credential_api.py` (routes), `credential_runtime.py` (per-run clients).
Modify `interactive/api.py`, `interactive/service.py`, `mcp_client.py` and public composition wiring; leave unrelated orchestrator logic intact.
New frontend modules: `app/components/ApiKeySettings.tsx`, `app/lib/credentials.ts`; update `AgentChat.tsx` and `app/lib/proxy.ts`.
New tests: `tests/test_credential_crypto.py`, `test_credential_vault.py`, `test_credential_api.py`, `test_credential_runtime.py`, and `sites/agent-web/tests/credentials.test.ts`.
Update `pyproject.toml`, `uv.lock`, `deploy/configure_runtime.py`, `deploy/modal_agent.py`, `docs/web-deployment.md` and frontend privacy text with the task that needs each change.

## Shared Interfaces

Define these types in `credential_vault.py`; secret-bearing dataclasses use `repr=False`, secrets use SecretStr, and no JSON serialization helper is exposed.

- `ProviderName = Literal['model', 'serper', 'serpapi']`.
- `CredentialStatus(provider, configured, last_four, updated_at, consent_at)` is safe public metadata; absent values are null.
- `CredentialMaterial(model: SecretStr, serper: SecretStr | None, serpapi: SecretStr | None)` is private runtime input.
- `CredentialError(code: str, status_code: int)` exposes only an allowlisted code, never ciphertext or original exception text.
- `CredentialRef(owner: str, generation: int, handle: str | None)` is private runtime routing metadata, never stored in chat/run records or exposed as a user-selectable owner.

### Task 1: Authenticated Encryption and Persistent Vault

**Files:** Create `credential_crypto.py`, `credential_vault.py`, `tests/test_credential_crypto.py`, `tests/test_credential_vault.py`; modify `pyproject.toml`, `uv.lock`.

**Interfaces:**
- `KeyCipher(keyring: Mapping[str, bytes], active_version: str)` validates 32-byte keys and requires the active version.
- `encrypt(owner: str, provider: ProviderName, plaintext: SecretStr) -> EncryptedCredential` returns version, nonce, ciphertext; `decrypt(owner, provider, envelope) -> SecretStr` authenticates the same versioned associated data.
- `CredentialVault(connection: sqlite3.Connection, cipher: KeyCipher, lock: threading.RLock)` shares the store's connection and lock, but owns a new credential table.
- `put(owner, provider, secret: SecretStr, *, consent: bool) -> CredentialStatus`; `status(owner) -> list[CredentialStatus]`; `read(owner, provider) -> SecretStr`; `delete(owner, provider) -> None`; `rotate_to_active(owner, provider) -> None`.
- Schema primary key `(owner, provider)`; ciphertext/nonce/version/last_four/updated_at/consent_at only. No plaintext column.

- [ ] Write tests asserting round trip, different nonce/ciphertext on repeated save, invalid AES key rejection, changed owner/provider/version and tampered ciphertext rejection. Assert `put(..., consent=False)` writes no row; Bob cannot read Alice; metadata/repr contain no test secret; replacement is atomic; delete is idempotent; active-key rotation can decrypt old entries while old key remains, then rewrites with active version.
- [ ] Run `uv run pytest tests/test_credential_crypto.py tests/test_credential_vault.py -v`; expect failures for missing implementation, not broken fixtures.
- [ ] Implement the interfaces using `cryptography.hazmat.primitives.ciphers.aead.AESGCM`, fresh 12-byte random nonces and deterministic encoded associated data. Translate crypto/storage failures into safe codes with `raise ... from None`. Add the direct dependency and update lockfile normally.
- [ ] Rerun the same command; all tests pass. Inspect SQLite DB/WAL bytes for the complete test secret and assert absence.
- [ ] Commit only Task 1 files: `feat: add encrypted user credential vault`.

### Task 2: Temporary Sessions and Invalidation

**Files:** Create `credential_sessions.py`; extend `tests/test_credential_vault.py`.

**Interfaces:**
- `CredentialSessions(vault: CredentialVault, clock: Callable[[], float])` owns an in-memory map; maximum 1 temporary session/owner and 100 globally; expire/prune entries before accepting new ones, otherwise `credential_sessions_busy` (503).
- `issue(owner, material: CredentialMaterial) -> str` replaces owner's previous temporary session, returns cryptographically random 32-byte URL-safe handle, expires in 1800 seconds.
- `reference(owner, handle: str | None) -> CredentialRef` checks ownership, temporary expiry or presence of a saved model key.
- `resolve(ref: CredentialRef) -> CredentialMaterial` rechecks generation and expiry at execution start; `invalidate(owner) -> None` increments generation and removes temporary sessions; `close() -> None` releases memory references.
- Preserve a running task's material locally until completion; no shared mutable client, no hard memory-zeroing claim.

- [ ] Write tests using an injected clock: wrong owner handle rejected; 1799 seconds valid, 1800 expired; restart loses temporary sessions; cap enforced; expired entries reclaimed; deleting/replacing/invalidation makes a queued reference fail; representations do not contain secret text.
- [ ] Run `uv run pytest tests/test_credential_vault.py -v`; expect missing session implementation to fail.
- [ ] Implement handles, bounded memory, generation checks and persistent fallback only to the same user's vault. No process environment fallback.
- [ ] Rerun focused tests; all pass.
- [ ] Commit Task 2 files: `feat: add owner-bound temporary credential sessions`.

### Task 3: Credential API and Proxy Boundary

**Files:** Create `credential_api.py`, `tests/test_credential_api.py`; modify `interactive/api.py`, frontend proxy and proxy tests.

**Interfaces:**
- `mount_credential_routes(app, *, owner_dependency, vault, sessions, cancel_owner: Callable[[str], Awaitable[None]], provider_hostname: str) -> None`.
- GET `/v1/credentials`: safe statuses plus provider hostname; POST `/v1/credentials`: body `{provider, key, consent:true}` saves/replaces; POST `/v1/credentials/delete`: body `{provider}` deletes.
- POST `/v1/credential-sessions`: body `{model_key, serper_key?, serpapi_key?}` issues temporary handle; response `{handle, expires_in:1800}`.
- Extra fields including owner/base_url forbidden. Keys nonempty, max 4096 characters, reject ASCII control characters; all validation responses use only safe codes.
- Saved mutations invalidate all owner's temporary references and await `cancel_owner(owner)` before returning. Deletion serializes against queued-run dispatch; task creation never slips through with an invalidated generation.
- Proxy allowlists these exact paths, uses only native login identity, overrides auth/user headers, forwards `x-agent-credential-handle` only for run creation. This header is never logged or persisted; handle possession without matching owner is insufficient.

- [ ] Write TestClient/proxy tests: anonymous denied before reading body; spoofed owner ignored/rejected; absent consent denied; response has no key; hostile origin blocked; 64 KiB request boundary still enforced; invalid nested key input not echoed; cross-user handles rejected; deletion awaits cancellation; status returns only safe metadata.
- [ ] Run `uv run pytest tests/test_credential_api.py -v` and `node --experimental-strip-types --test tests/proxy.test.ts` in the frontend directory; expect route/allowlist failures before changes.
- [ ] Implement routes with SecretStr models, no request-body logging, reuse existing authenticated pre-parser boundary and server proxy. No network model call during save. Provider hostname comes from server configuration, not browser input.
- [ ] Rerun both commands; all pass. Capture logs using fake keys and assert no plaintext anywhere in captured output.
- [ ] Commit Task 3 files: `feat: expose authenticated user credential settings`.

### Task 4: Per-User Runtime and Paid-Call Isolation

**Files:** Create `credential_runtime.py`, `tests/test_credential_runtime.py`; modify `interactive/api.py`, `interactive/service.py`, `interactive/composition.py`, `mcp_client.py`, existing relevant tests.

**Interfaces:**
- `UserRuntime(settings: InteractiveRuntimeSettings, sessions: CredentialSessions)`; settings includes a validated admin HTTPS base URL, gpt-6.1-sol/low, existing budgets, and no model/search secret.
- `UserRuntime.execute(ref, request, workspace, history, on_progress, attachment_path) -> InteractiveResult` resolves material after obtaining a worker slot, creates independent clients, closes them in finally.
- `RunService.start(..., *, credential_ref: CredentialRef | None = None)` and `cancel_owner(owner: str) -> None`; refs remain only in memory, never SQLite. Existing fake-executor tests may inject a fake resolver.
- Keep idempotent lookup before credential resolution/quota charging: existing same-payload run returns unchanged, including terminal result. New run must have valid credentials before acceptance.
- Extend `MCPToolClient(..., runtime_env: Mapping[str,str] | None = None)`; `None` preserves existing CLI behavior. An explicitly supplied mapping is authoritative for base URL/search keys and excludes all inherited owner provider secrets. Update redaction to use actual injected secrets, not only os.environ.
- Pass explicit base_url to the parent AsyncOpenAI client and to MCP children. Construct InteractiveConfig directly from trusted settings and the resolved user's key; never call RunConfig.from_env for public work.

- [ ] Write async tests with distinct fake Alice/Bob/owner secrets: parent model, visual/audio clients and MCP env use only correct user values; os.environ unchanged; absent user search keys absent in child env; public startup succeeds without OPENAI_API_KEY; missing/invalid/unsupported user credentials never invoke owner clients.
- [ ] Add queue tests: expired/ref-revoked run never invokes executor; delete/replace cancels active and queued runs; idempotent completed retry returns existing answer without a new key or new paid call. Inject nested upstream errors containing fake keys; assert only safe error codes and no secrets in logs/results/notebooks/progress.
- [ ] Run `uv run pytest tests/test_credential_runtime.py tests/test_interactive_service.py tests/test_interactive_composition.py -v`; expect injection/isolation assertions to fail on existing global configuration.
- [ ] Implement interfaces, cancellation/invalidation ordering and explicit environment injection. Ensure all client/subprocess teardown completes even on timeout/cancellation; allowlist upstream failure classification rather than exposing str(error).
- [ ] Rerun focused tests plus existing MCP/provider/search tests selected using `rg --files tests`; all pass and existing CLI environment tests retain their previous behavior.
- [ ] Commit Task 4 files: `feat: isolate public agent execution by user credentials`.

### Task 5: Key Settings UI and User-Paid Execution

**Files:** Create `app/components/ApiKeySettings.tsx`, `app/lib/credentials.ts`, `tests/credentials.test.ts` under `sites/agent-web`; modify `AgentChat.tsx`, styles and proxy tests as needed.

**Interfaces:**
- `ApiKeySettings({onCredentialChange}:{onCredentialChange:(handle:string|null)=>void})` displays server-configured provider hostname, saved-key metadata, replacement/deletion and unchecked save-consent checkbox.
- `submitCredentials(input, api) -> Promise<{handle:string|null}>` selects persistent save only when consent checked; otherwise creates temporary session. Typed API interface must never return the original key.
- AgentChat holds temporary handle in React memory only, forwards header on start, clears it after expiry/sign-out/deletion. Saved configuration requires no browser-held secret. Never disable reading old answers because a key is missing.
- Required copy: “模型与工具费用计入你的 API 账号。Key 会发送至本服务后端及所显示的模型服务；勾选保存后将加密存储。未勾选时仅临时使用。” Also state API membership/subscription and API balance are separate concepts without promising a particular provider's billing terms.

- [ ] Write Node tests for helper behavior: unchecked consent uses temporary route; checked consent uses save route with true; failed save does not imply configured status; provider hostname mandatory before submit; deletion clears handle/state; no key is written to storage or included in query/path.
- [ ] Run `node --experimental-strip-types --test tests/*.test.ts`; expect missing helper failures.
- [ ] Implement accessible password fields, provider display, explicit consent, masked metadata, replace/delete controls and safe error messages. Clear plaintext input after successful submission and when closing settings. Optional search keys use the same explicit consent rule; do not auto-save them because a model key was previously saved.
- [ ] Rerun Node tests, `node node_modules/typescript/bin/tsc --noEmit --incremental false`, and `npm run build`; all must pass.
- [ ] Inspect desktop/mobile and keyboard behavior in local browser: checkbox initially unchecked, no plaintext after reopening settings, delete then reload reports unconfigured, missing key blocks new execution but not history. Use only fake keys with mock backend; clearly label mock UI testing and do not call it real provider verification.
- [ ] Commit Task 5 files: `feat: add explicit-consent API key settings`.

### Task 6: Deployment Safety, Regression and Handoff

**Files:** Modify `deploy/configure_runtime.py`, `deploy/modal_agent.py`, `docs/web-deployment.md`; create `tests/test_public_runtime_configuration.py`.

**Interfaces:**
- Replace deployment helper's local .env credential copying with explicit infrastructure-only configuration: service token, encryption keyring JSON and active version, allowed model endpoint. Never read or upload owner OpenAI/search keys.
- Backend accepts missing owner keys but requires valid infrastructure secrets before enabling credential storage. No auto-generated ephemeral encryption master key.
- Document rotation: add new version, migrate rows via vault rotation, verify all rows active, then retire old version; no plaintext export or command-line key literals.

- [ ] Write tests asserting the deployment helper never selects OPENAI_API_KEY/SERPER_API_KEY/SERPAPI_API_KEY; missing/malformed master key fails closed; public runtime ignores deliberately set owner credentials; old ciphertext survives normal restart with same keyring.
- [ ] Run `uv run pytest tests/test_public_runtime_configuration.py -v`; expect current .env-copy behavior to fail.
- [ ] Update deployment code/docs; explain cancellation/billing, session expiry, key deletion limits, operator trust, and that provider support is not verified merely by accepting key syntax. Add explicit release checklist for real hosting snapshot retention and its disclosed duration; keep public publishing blocked until verified.
- [ ] Run `uv run pytest -o addopts='' -q`, `uv run ruff check src tests deploy`, `git diff --check`, frontend Node tests, TypeScript and production build. Record exact counts/output; do not reuse previous 695-test report as verification for this change.
- [ ] Inspect repository, database/WAL, captured logs and built client artifacts for synthetic test secrets; all absent. Check git status before staging to exclude credentials, build artifacts and unrelated changes.
- [ ] Perform final inline spec-to-diff review in this side conversation; do not spawn review agents here. An independent security review may be arranged in the main task before public release, with appropriate authorization there.
- [ ] Commit Task 6 files: `chore: secure BYOK deployment and document verification`.
- [ ] Handoff with implemented/not implemented and verified/not verified clearly separated. No cloud deployment, real credential transfer or paid model smoke test without separate active authorization. Provide the approved spec, plan, test report and commit references for continuation in the main task.

## Plan Self-Review

Spec coverage: crypto/consent/rotation Task 1; temporary memory/expiry Task 2; identity/API/privacy boundaries Task 3; every paid route and no-fallback execution Task 4; explicit user consent and lifecycle UI Task 5; deployment secrets, snapshot disclosures and release gates Task 6.
All five Review Focus items have explicit test ownership above. Stored keys and temporary handles are distinct interfaces; neither is written into run payloads. Snapshot retention is an external release check, not a fabricated fixed retention promise. This plan does not itself authorize cloud changes.
