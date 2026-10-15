# Interactive website deployment

Status: BYOK implemented locally; no cloud deployment or real paid calls authorized here.
The website uses native Sites sign-in and a server-only proxy; Python runs separately on Modal.
GPT-6.1 Sol / low is a configured preference, not guaranteed API availability. Unsupported accounts
fail without model/provider/owner-key fallback. Historical GAIA scores and CLI remain unchanged.

## Runtime configuration

- Python: `AGENT_SERVICE_TOKEN`, `AGENT_CREDENTIAL_KEYRING`,
  `AGENT_CREDENTIAL_ACTIVE_VERSION`, `AGENT_MODEL_BASE_URL`.
- Sites: `AGENT_BACKEND_URL` and secret `AGENT_SERVICE_TOKEN`.
- Never put secrets in browser code, hosting.json, Git, command-line literals or tracked env files.

Keyring is JSON mapping version names to base64-encoded 32-byte random AES keys. The active version
must exist. Missing/malformed infrastructure configuration fails closed: no automatic ephemeral
master key. Master keys stay separate from SQLite and are never given to the frontend Site.
The model endpoint must be administrator-approved HTTPS, without credentials, query or fragment.
The hostname is displayed before key entry; users cannot choose their own endpoint.

Install with `uv sync --extra dev`. Configure a named Modal secret `toy-agent-web-byok-runtime`,
then deploy using `modal deploy deploy/modal_agent.py`.
The Modal CLI is separate deployment tooling. `deploy/configure_runtime.py` accepts exactly the four
Python infrastructure values as JSON on hidden stdin. Invoking it transmits secrets and requires
separate authorization. It never reads `.env` or copies owner model/search keys; extra fields and
env file paths are rejected. Do not invoke it or deploy this local mock configuration without
authorization. Public startup requires no owner OpenAI/search key and ignores deliberately set ones.
The API uses a single backend container and SQLite on the `toy-agent-web-data` Volume. Do not
increase container replicas: the current storage and queue implementation requires one writer.
Store updates checkpoint every 15 seconds and at clean shutdown; abrupt infrastructure failure
can lose recent updates, including a deletion. Revoke with the provider when urgent.
Unfinished runs become interrupted after restart. Deployment should occur
while no runs are active. This first release is suitable for low-volume use, not high-availability
multi-replica service.

## Limits and privacy

One active run per user, three runs per rolling hour, two executing runs globally, ten queued runs.
Each run has 500 seconds and 24 logical tool calls. One supported attachment per question,
up to 32 MiB; three successful uploads per user per hour. Upload records expire after seven days
and are removed during service startup. Questions, results and uploads are private to their owner.
Temporary run workspaces are removed after execution. Conversations persist on the backend.

These limits bound work, not a dollar spending ceiling. Configure provider and hosting budgets
separately. Model/tool charges go to the visitor's API account; the owner still pays backend,
storage and hosting. Membership subscriptions and API balances are separate concepts, whose terms
depend on the provider. Accepting key syntax never proves model access, balance or billing terms.

## Consent, private credentials and deletion

Save consent is unchecked by default. Keys go to this backend and the displayed model provider;
optional search keys go to their respective Serper/SerpAPI APIs. The operator/hosting infrastructure
can access runtime plaintext: this is not end-to-end encryption or a claim the operator cannot read
keys. Independently assess operator, hosting and provider trust before entering any real key.

Unchecked consent creates a private, owner-bound in-memory session for 30 minutes. One session per
owner, 100 globally. Refresh or restart can require re-entry. Queued runs recheck expiry/generation
at dispatch; active runs retain their private snapshot only through completion/cancellation or the
500-second deadline. There is no browser key/handle persistence or process environment mutation.
Each parent/visual/audio/hosted-tool client and MCP child uses only its user's credentials.
Missing user search keys disable independent search providers; hosted search uses the model key.

Saving uses AES-256-GCM, fresh 12-byte nonce and owner/provider/version-bound associated data.
SQLite holds ciphertext, nonce, version, last four characters and update/consent timestamps only.
There is no plaintext read/export endpoint or credential backup/export feature.
Replacing/deleting revokes queued references and awaits cancellation of that user's tasks before
returning. Online ciphertext and temporary references are removed; already submitted requests
cannot be recalled and may still cost money. For urgent revocation, revoke at the provider.
Deletion is not guaranteed physical erasure of Python memory, SQLite free pages, WAL or hosting
snapshots. A restart/restore must not silently restore usable deleted credentials; actual platform
snapshot behavior and retention are unverified release gates, not a fabricated fixed promise.

## Master-key rotation

Stop new admissions and wait for active runs to finish. Add a new random 32-byte version to the
platform Secret, retaining the old versions; switch active version and restart. In an administrator-
only maintenance session inside the trusted backend environment, construct the vault with that
keyring and call `rotate_to_active(owner, provider)` for each row. Inspect version counts only;
all rows must be active before removing old versions. Verify decryption/restart, then retire old
versions and restart. Never export plaintext or put key literals in shell arguments. Lost master
keys require affected users to re-enter keys. Historical snapshots may still need older versions:
decide recovery/retention policy explicitly rather than promising immediate total erasure.

## Verification

Run `uv run pytest -o addopts='' -q`, `uv run ruff check src tests deploy`, frontend Node tests,
TypeScript noEmit and production build. Local fake-key UI checks are not real-provider verification.
Before public release, ALL of the following must be independently verified:

- [ ] Independent BYOK security review (no paid client or child inherits owner credentials).
- [ ] Native real sign-in, HTTPS and proxy; anonymous/cross-user requests denied.
- [ ] Operator-approved provider endpoint/trust and model access with a consenting tester.
- [ ] Separate authorization for infrastructure Secret transfer and cloud deployment.
- [ ] Separately authorized paid smoke test; cancellation, expiry, retry and recovery checked.
- [ ] Real platform body/header/error/trace collection audited to exclude credentials.
- [ ] Actual Modal/Sites snapshot retention configured and verified; publish its actual duration
      in privacy copy. Duration is currently UNVERIFIED: public release remains blocked.
- [ ] Real deletion/restart/restore behavior and residual exposure accurately documented.

No cloud, real sign-in or provider checks above were performed during local BYOK implementation.
