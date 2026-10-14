# Interactive website deployment

The website uses Sites sign-in and a server-only proxy. Its Python service runs on Modal.
Agent execution uses GPT-6.1 Sol with low reasoning. Historical GAIA scores are unchanged.

## Runtime configuration

- Python: `OPENAI_API_KEY`, `AGENT_SERVICE_TOKEN`, optional search provider keys.
- Sites: `AGENT_BACKEND_URL` and secret `AGENT_SERVICE_TOKEN`.
- Never put these values in browser code, hosting.json or tracked env files.

Install with `uv sync --extra dev`. Configure a named Modal secret `toy-agent-web-runtime`,
then deploy using `modal deploy deploy/modal_agent.py`.
The API uses a single backend container and SQLite on the `toy-agent-web-data` Volume. Do not
increase container replicas: the current storage and queue implementation requires one writer.
Store updates checkpoint every 15 seconds and at clean shutdown; abrupt infrastructure failure
can lose recent updates. Unfinished runs become interrupted after restart. Deployment should occur
while no runs are active. This first release is suitable for low-volume use, not high-availability
multi-replica service.

## Limits and privacy

One active run per user, three runs per rolling hour, two executing runs globally, ten queued runs.
Each run has 500 seconds and 24 logical tool calls. One supported attachment per question,
up to 32 MiB; three successful uploads per user per hour. Upload records expire after seven days
and are removed during service startup. Questions, results and uploads are private to their owner.
Temporary run workspaces are removed after execution. Conversations persist on the backend.

These limits bound work, not a dollar spending ceiling. Configure provider and hosting budgets
separately. API and hosting charges are paid by the service owner.

## Verification

Run `uv run pytest`, `uv run ruff check src tests` and the Site TypeScript/build checks.
Test public landing, real sign-in, question-to-result, cancel, reconnect and an unauthorized request
before opening the service to other users. Report any unverified live integration explicitly.
