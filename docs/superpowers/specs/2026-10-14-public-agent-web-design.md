# Public interactive agent web application

## Goal and confirmed requirements

Expose the existing GAIA research agent as a working website for other people. Users can ask
questions, attach files, inspect execution progress, receive evidence-backed answers, and follow up.
Use gpt-6.1-sol with low reasoning for agent and hosted research/computation calls. Retain the
specialized audio transcription model. Existing GAIA evaluation and historical scores remain intact.

## Proposed first release

A responsive chat interface with conversation navigation, question composer, attachment upload,
execution timeline, cancel action, answer and source cards, and a follow-up composer. Show useful
stage events rather than hidden reasoning. Clearly distinguish failure, cancellation and completion.
Do not display invented progress or demo answers when the backend is unavailable.

The site is publicly accessible, but running the agent requires authentication. Prefer Sites'
supported ChatGPT sign-in. Initial proposed limits: one active run per user, three new runs per
hour per user, 500 seconds and 24 logical tool calls per run. Enforce limits on the server.
These are initial product settings, not an estimated spending ceiling; token usage and hosted
tools also affect cost. Provider credentials stay exclusively on the backend.

## Architecture

Sites hosts the web interface and a server-side proxy. A separately hosted Python service runs
the existing LangGraph orchestrator and stdio MCP processes. Sites Workers cannot execute this
Python/process runtime. The proxy authenticates visitors, derives their identity, and calls the
Python service with a server-held credential; do not accept a browser-supplied user ID as identity.

The Python service provides conversation/run creation, upload, progress polling, result retrieval,
and cancellation. Polling is sufficient for the first release and supports reconnecting. Every
operation checks conversation ownership. Use unique task and workspace IDs per run. Initial
worker concurrency is bounded and queue saturation returns a clear busy status.

Extract reusable agent composition from the benchmark runner to construct an interactive task
without dataset download, HF_TOKEN, reference answers, or GAIA scoring. The existing benchmark
path continues to use its own task/scoring contracts. Use per-run agent instances to avoid
cross-user state in usage and response-diagnostic fallbacks.

## Answers and follow-ups

Retain the candidate and checked evidence from successful execution for a private result view.
Present the concise answer with supporting evidence summaries and source links. Follow-ups become
new runs with bounded prior conversation context; do not replay raw tool history indefinitely.
If a richer natural-language answer requires another model call, it must use the same model and
reasoning, fit within the run budget, and not introduce unsupported claims.

## Uploads and data

Reuse attachment registration, path containment and parser resource limits. Limit total upload
size per run, validate media type, and isolate every user's workspace. Keep questions, uploads,
answers and research state private to the owner. Public audit/trace data continues to use
allowlisted metadata. Persist conversations and run status; establish cleanup for temporary files
and retained uploads. Do not publish dataset content or credentials.

## Deployment dependency

Publish a working private staging site for verification, then enable public access after the
authenticated API integration works. A Python-capable hosting destination and backend runtime
secrets are required; select an available supported host before deploying the Python service.
Do not present a frontend-only deployment as a working public agent.

## Verification and acceptance

Test interactive composition without GAIA credentials, per-user authorization, cross-user access
denial, quota and queue enforcement, upload validation, cancellation, progress terminal states,
follow-up context bounds, and secret/trace privacy. Run the existing Python suite. Validate web
build and actual login-to-question-to-result flow in staging with a controlled live request.
Delivery requires a deployed URL with a reachable real backend; document costs and operational
limits separately from historical benchmark accuracy.
