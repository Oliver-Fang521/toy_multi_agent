# GAIA Multi-Agent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a four-plane, tool-using multi-agent system on a reproducible 15-question subset of GAIA 2023 validation, then report its measured performance.

**Architecture:** A provider-neutral `Plan -> Act -> Verify -> Submit` orchestrator routes work to planner, lookup, computation, file, visual, and critic specialists. Specialists access a real local stdio MCP server with allowlisted tools; the initial OpenAI provider uses Responses, hosted Web Search, and hosted Code Interpreter. A separate evaluation layer loads GAIA, checkpoints every result, applies the official scorer, and generates privacy-safe reports.

**Tech Stack:** Python 3.11+, uv, Pydantic 2, OpenAI Python SDK, MCP Python SDK 2.x, Hugging Face `datasets`/`huggingface_hub`, HTTPX, Beautiful Soup, pypdf, openpyxl, Typer, pytest, pytest-asyncio, Ruff.

## Global Constraints

- The first measured run contains exactly 15 GAIA 2023 validation tasks: five from each level, selected with seed 42.
- The initial provider uses the OpenAI Responses API; all orchestration code depends on `ModelProvider`, never on the vendor client.
- Default model is `gpt-5.6-sol` with explicit `reasoning.effort="medium"`; both remain configurable.
- Model-generated Python runs only in an ephemeral OpenAI Code Interpreter container, never on the host.
- The shared tool plane is a real local stdio MCP server using `mcp>=2,<3`.
- Every terminal prediction is verified by `CriticAgent` or records a stable terminal failure category.
- Scoring matches the official GAIA quasi-exact-match function.
- Questions, reference answers, attachments, credentials, and hidden reasoning never appear in committed files or privacy-safe result artifacts.
- Automated development follows red-green-refactor; every production behavior begins with a test that fails for the intended reason.

---

## File Structure

```text
.
├── .env.example                         # credential names and safe defaults only
├── .gitignore                           # caches, credentials, datasets, run artifacts
├── README.md                            # installation, architecture, evaluation workflow
├── pyproject.toml                       # package metadata, dependencies, tools, CLI entry
├── uv.lock                              # reproducible dependency resolution
├── src/gaia_agent/
│   ├── __init__.py                      # package version
│   ├── agents.py                        # specialist roles and prompts
│   ├── cli.py                           # Typer commands: evaluate and verify-results
│   ├── config.py                        # validated environment/CLI configuration
│   ├── dataset.py                       # gated GAIA loading and stratified sampling
│   ├── mcp_client.py                    # stdio lifecycle, discovery, allowlists, calls
│   ├── mcp_server.py                    # MCP tools and hosted search/code backends
│   ├── models.py                        # immutable domain value objects
│   ├── orchestrator.py                  # Plan -> Act -> Verify -> Submit state machine
│   ├── persistence.py                   # manifest, atomic checkpoint, resume
│   ├── reporting.py                     # metrics, summary JSON, Markdown report
│   ├── runner.py                        # per-task execution and evaluation coordination
│   ├── scoring.py                       # official-compatible GAIA scoring
│   └── providers/
│       ├── __init__.py                  # provider exports
│       ├── base.py                      # ModelProvider and ToolExecutor protocols
│       └── openai.py                    # Responses adapter and tool-call loop
└── tests/
    ├── conftest.py                      # shared task/provider/tool fixtures
    ├── fixtures/gaia_records.json       # synthetic GAIA-shaped records only
    ├── test_agents.py                   # prompts, schemas, allowlists
    ├── test_cli_e2e.py                  # offline end-to-end CLI run
    ├── test_config.py                   # validation and secret handling
    ├── test_dataset.py                  # deterministic balanced sampling
    ├── test_mcp.py                      # real stdio discovery and calls
    ├── test_openai_provider.py          # Responses mapping and continuation loop
    ├── test_orchestrator.py             # routing, verification, correction, voting
    ├── test_persistence.py              # atomic checkpoints and resume guards
    ├── test_reporting.py                # aggregate/per-level metrics and privacy
    └── test_scoring.py                  # GAIA scorer parity cases
```

### Task 1: Package Foundation, Domain Models, and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/gaia_agent/__init__.py`
- Create: `src/gaia_agent/models.py`
- Create: `src/gaia_agent/config.py`
- Create: `tests/test_config.py`
- Create: `uv.lock`

**Interfaces:**
- Produces: `RunConfig.from_env(**overrides) -> RunConfig`
- Produces: immutable `GaiaTask`, `PlanStep`, `AgentPlan`, `Evidence`, `Usage`, `AgentResult`, `CriticDecision`, `TaskPrediction`
- Consumes: no earlier task interfaces

- [ ] **Step 1: Create packaging metadata and test configuration**

```toml
[project]
name = "gaia-multi-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "beautifulsoup4>=4.13,<5",
  "datasets>=4,<5",
  "huggingface-hub>=0.34,<2",
  "httpx>=0.28,<1",
  "mcp>=2,<3",
  "openai>=2,<3",
  "openpyxl>=3.1,<4",
  "pydantic>=2.11,<3",
  "pypdf>=6,<7",
  "typer>=0.16,<1",
]

[project.optional-dependencies]
dev = ["pytest>=8.4,<10", "pytest-asyncio>=1,<2", "ruff>=0.12,<1"]

[project.scripts]
gaia-agent = "gaia_agent.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-q"
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

Add `.gitignore` entries for `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.hf/`, `data/`, and `results/`. Add `.env.example` with names only: `OPENAI_API_KEY=`, `HF_TOKEN=`, `GAIA_MODEL=gpt-5.6-sol`, and `GAIA_REASONING_EFFORT=medium`.

- [ ] **Step 2: Write failing configuration and model tests**
      
```python
from pydantic import ValidationError
import pytest

from gaia_agent.config import RunConfig
from gaia_agent.models import GaiaTask


def test_run_config_requires_balanced_sample_size() -> None:
    with pytest.raises(ValidationError, match="divisible by 3"):
        RunConfig(openai_api_key="sk-test", hf_token="hf_test", sample_size=10)


def test_run_config_redacts_secrets() -> None:
    config = RunConfig(openai_api_key="sk-secret", hf_token="hf_secret")
    assert "sk-secret" not in repr(config)
    assert "hf_secret" not in repr(config)


def test_gaia_task_rejects_unknown_level() -> None:
    with pytest.raises(ValidationError):
        GaiaTask(task_id="x", question="q", level=4, final_answer="a")
```

- [ ] **Step 3: Run the tests and confirm the missing-module failure**

Run: `uv run pytest tests/test_config.py -q`

Expected: collection fails because `gaia_agent.config` and `gaia_agent.models` do not exist.

- [ ] **Step 4: Implement minimal typed configuration and domain models**

```python
# src/gaia_agent/config.py
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, SecretStr, model_validator


class RunConfig(BaseModel):
    openai_api_key: SecretStr
    hf_token: SecretStr
    model: str = "gpt-5.6-sol"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    sample_size: int = 15
    seed: int = 42
    max_steps: int = 8
    max_tool_calls: int = 24
    timeout_seconds: float = 300.0
    votes: int = 1
    output_dir: Path = Path("results/gaia-15-seed42")
    resume: bool = False

    @model_validator(mode="after")
    def validate_run_shape(self) -> "RunConfig":
        if self.sample_size < 3 or self.sample_size % 3:
            raise ValueError("sample_size must be at least 3 and divisible by 3")
        if self.votes < 1 or self.votes % 2 == 0:
            raise ValueError("votes must be a positive odd integer")
        return self

    @classmethod
    def from_env(cls, **overrides: object) -> "RunConfig":
        values: dict[str, object] = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY"),
            "hf_token": os.environ.get("HF_TOKEN"),
            "model": os.environ.get("GAIA_MODEL", "gpt-5.6-sol"),
            "reasoning_effort": os.environ.get("GAIA_REASONING_EFFORT", "medium"),
            "max_steps": os.environ.get("GAIA_MAX_STEPS", "8"),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)
```

Implement the domain models with these exact public fields:

```python
class SpecialistName(StrEnum):
    LOOKUP = "lookup"
    COMPUTATION = "computation"
    FILE = "file"
    VISUAL = "visual"


class GaiaTask(BaseModel):
    task_id: str
    question: str
    level: Literal[1, 2, 3]
    final_answer: str = Field(exclude=True)
    attachment_path: Path | None = None


class PlanStep(BaseModel):
    specialist: SpecialistName
    instruction: str


class AgentPlan(BaseModel):
    restatement: str
    steps: tuple[PlanStep, ...]


class Evidence(BaseModel):
    evidence_id: str
    kind: Literal["source", "computation", "file", "visual"]
    summary: str
    source_url: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class AgentResult(BaseModel):
    specialist: SpecialistName
    evidence: tuple[Evidence, ...] = ()
    candidate_answer: str | None = None
    usage: Usage = Field(default_factory=Usage)


class CriticDecision(BaseModel):
    verdict: Literal["approve", "revise", "insufficient_evidence"]
    feedback: str
    checked_evidence_ids: tuple[str, ...]
    corrected_answer: str | None = None


class TaskPrediction(BaseModel):
    task_id: str
    level: Literal[1, 2, 3]
    answer: str | None
    correct: bool = False
    duration_seconds: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    critic_verdict: str
    evidence_ids: tuple[str, ...] = ()
    error_category: str | None = None
    run_index: int = 0
    vote_count: int = 1
```

Add `AuditEvent` with `sequence`, `event_type`, `timestamp`, and redacted `payload`. Use enums for stable terminal error values while serializing their string values.

- [ ] **Step 5: Run tests and static checks**

Run: `uv lock && uv run pytest tests/test_config.py -q && uv run ruff check src tests`

Expected: all configuration tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit the foundation**

```bash
git add pyproject.toml uv.lock .gitignore .env.example src/gaia_agent tests/test_config.py
git commit -m "feat: add GAIA agent domain foundation"
```

### Task 2: Official Scoring and Deterministic GAIA Sampling

**Files:**
- Create: `src/gaia_agent/scoring.py`
- Create: `src/gaia_agent/dataset.py`
- Create: `tests/fixtures/gaia_records.json`
- Create: `tests/test_scoring.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `GaiaTask`
- Produces: `question_scorer(model_answer: str | None, ground_truth: str) -> bool`
- Produces: `sample_stratified(tasks: Sequence[GaiaTask], sample_size: int, seed: int) -> list[GaiaTask]`
- Produces: `load_gaia_validation(token: str, cache_dir: Path | None = None) -> tuple[list[GaiaTask], str]`, where the second item is the dataset revision

- [ ] **Step 1: Write failing scorer parity tests**

```python
import pytest
from gaia_agent.scoring import question_scorer


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("$1,234", "1234", True),
        ("12%", "12", True),
        ("Sea Gull", "seagull", True),
        ("A; B", "A;B", True),
        ("1, 2", "1,2", True),
        ("B, A", "A,B", False),
        (None, "None", True),
    ],
)
def test_question_scorer_matches_official_cases(answer, gold, expected) -> None:
    assert question_scorer(answer, gold) is expected
```

- [ ] **Step 2: Run scorer tests and confirm failure**

Run: `uv run pytest tests/test_scoring.py -q`

Expected: import fails because `gaia_agent.scoring` is absent.

- [ ] **Step 3: Port the official scorer without semantic changes**

```python
def question_scorer(model_answer: str | None, ground_truth: str) -> bool:
    answer = "None" if model_answer is None else model_answer
    if _is_float(ground_truth):
        return _normalize_number(answer) == float(ground_truth)
    if any(char in ground_truth for char in [",", ";"]):
        gold_parts = _split(ground_truth)
        answer_parts = _split(answer)
        if len(gold_parts) != len(answer_parts):
            return False
        return all(_compare_element(a, g) for a, g in zip(answer_parts, gold_parts, strict=True))
    return _normalize_string(answer) == _normalize_string(ground_truth)
```

Implement `_normalize_number`, `_normalize_string`, `_split`, `_is_float`, and `_compare_element` exactly as described in the design specification.

- [ ] **Step 4: Write failing balanced-sampling tests with synthetic records**

```python
from gaia_agent.dataset import sample_stratified


def test_sample_is_balanced_stable_and_gold_independent(synthetic_tasks) -> None:
    first = sample_stratified(synthetic_tasks, sample_size=6, seed=42)
    second = sample_stratified(list(reversed(synthetic_tasks)), sample_size=6, seed=42)
    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert [task.level for task in first].count(1) == 2
    assert [task.level for task in first].count(2) == 2
    assert [task.level for task in first].count(3) == 2
```

- [ ] **Step 5: Implement sampling and gated dataset conversion**

```python
def sample_stratified(tasks: Sequence[GaiaTask], sample_size: int, seed: int) -> list[GaiaTask]:
    if sample_size < 3 or sample_size % 3:
        raise ValueError("sample_size must be at least 3 and divisible by 3")
    per_level = sample_size // 3
    selected: list[GaiaTask] = []
    for level in (1, 2, 3):
        pool = sorted((task for task in tasks if task.level == level), key=lambda task: task.task_id)
        if len(pool) < per_level:
            raise ValueError(f"level {level} has {len(pool)} tasks; need {per_level}")
        selected.extend(random.Random(seed + level).sample(pool, per_level))
    return sorted(selected, key=lambda task: (task.level, task.task_id))
```

Use `snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset", token=token)` followed by `load_dataset(snapshot_path, "2023_all", split="validation")`. Resolve each non-empty `file_path` against the snapshot root and reject paths that escape it.

- [ ] **Step 6: Verify and commit scoring/sampling**

Run: `uv run pytest tests/test_scoring.py tests/test_dataset.py -q`

Expected: all cases pass without network access.

```bash
git add src/gaia_agent/scoring.py src/gaia_agent/dataset.py tests/fixtures tests/test_scoring.py tests/test_dataset.py
git commit -m "feat: add GAIA sampling and official scoring"
```

### Task 3: Provider Protocol and OpenAI Responses Adapter

**Files:**
- Create: `src/gaia_agent/providers/__init__.py`
- Create: `src/gaia_agent/providers/base.py`
- Create: `src/gaia_agent/providers/openai.py`
- Create: `tests/test_openai_provider.py`

**Interfaces:**
- Produces: `ToolSpec(name: str, description: str, parameters: dict[str, Any])`
- Produces: `ToolExecutor.call(name: str, arguments: dict[str, Any]) -> dict[str, Any]`
- Produces: `ModelProvider.complete(request: ModelRequest, executor: ToolExecutor | None) -> ModelResponse`
- `ModelRequest` fields: `instructions`, `input_items`, `output_schema`, `allowed_tools`, `max_output_tokens`, `max_tool_calls`, `reasoning_effort`
- `ModelResponse` fields: `parsed`, `output_text`, `usage`, `citations`, `tool_events`, `remote_file_ids`

- [ ] **Step 1: Write failing protocol mapping and continuation tests**

```python
@pytest.mark.asyncio
async def test_openai_provider_executes_function_call_and_continues(fake_responses_client) -> None:
    fake_responses_client.queue_function_call("execute_python", '{"code":"print(6*7)"}', "call-1")
    fake_responses_client.queue_text('{"answer":"42"}')
    executor = RecordingExecutor({"stdout": "42"})
    provider = OpenAIProvider(client=fake_responses_client, model="gpt-5.6-sol")

    response = await provider.complete(model_request(), executor)

    assert executor.calls == [("execute_python", {"code": "print(6*7)"})]
    assert fake_responses_client.second_input == [
        {"type": "function_call_output", "call_id": "call-1", "output": '{"stdout":"42"}'}
    ]
    assert response.parsed == {"answer": "42"}
```

Also test explicit `reasoning={"effort": "medium"}`, strict function schemas, `previous_response_id`, one retry for `RateLimitError`/timeouts/server errors, no retry for authentication/invalid requests, refusal mapping, usage aggregation, and source extraction.

- [ ] **Step 2: Run tests and confirm missing provider types**

Run: `uv run pytest tests/test_openai_provider.py -q`

Expected: collection fails because provider modules do not exist.

- [ ] **Step 3: Define provider-neutral request/response protocols**

```python
class ToolExecutor(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ModelProvider(Protocol):
    async def complete(
        self, request: ModelRequest, executor: ToolExecutor | None = None
    ) -> ModelResponse:
        raise NotImplementedError
```

Use frozen Pydantic models for request, response, tool, usage, citation, and observable tool-event values.

- [ ] **Step 4: Implement the bounded Responses function-call loop**

```python
response = await self._client.responses.create(
    model=self._model,
    instructions=request.instructions,
    input=request.input_items,
    tools=[tool.as_openai_dict() for tool in request.allowed_tools],
    text={"format": request.output_schema},
    reasoning={"effort": request.reasoning_effort},
    max_output_tokens=request.max_output_tokens,
    max_tool_calls=request.max_tool_calls,
)
while calls := [item for item in response.output if item.type == "function_call"]:
    outputs = []
    for call in calls:
        result = await executor.call(call.name, json.loads(call.arguments))
        outputs.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, separators=(",", ":")),
        })
    response = await self._client.responses.create(
        model=self._model,
        instructions=request.instructions,
        previous_response_id=response.id,
        input=outputs,
        tools=[tool.as_openai_dict() for tool in request.allowed_tools],
        text={"format": request.output_schema},
        reasoning={"effort": request.reasoning_effort},
        max_output_tokens=request.max_output_tokens,
        max_tool_calls=request.max_tool_calls,
    )
```

Guard the loop with the request's global tool-call budget and parse final JSON into `ModelResponse.parsed`. Never log request headers, secrets, or reasoning items.

- [ ] **Step 5: Verify and commit the provider adapter**

Run: `uv run pytest tests/test_openai_provider.py -q && uv run ruff check src/gaia_agent/providers tests/test_openai_provider.py`

```bash
git add src/gaia_agent/providers tests/test_openai_provider.py
git commit -m "feat: add provider-neutral Responses adapter"
```

### Task 4: Real Stdio MCP Tool Plane and Hosted Backends

**Files:**
- Create: `src/gaia_agent/mcp_server.py`
- Create: `src/gaia_agent/mcp_client.py`
- Create: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `ToolExecutor`, `ToolSpec`, `RunConfig`
- Produces: `build_server(settings: MCPServerSettings) -> MCPServer`
- Produces: `MCPToolClient(task_root: Path, config: RunConfig)` async context manager implementing `ToolExecutor`
- Exposes tools: `web_search`, `read_webpage`, `execute_python`, `read_attachment`, `submit_answer`

- [ ] **Step 1: Write failing stdio discovery, allowlist, and safety tests**

```python
@pytest.mark.asyncio
async def test_stdio_client_discovers_and_filters_tools(tmp_path, fake_backend_env) -> None:
    async with MCPToolClient(task_root=tmp_path, allowed={"execute_python"}) as client:
        assert [tool.name for tool in client.tool_specs] == ["execute_python"]


@pytest.mark.asyncio
async def test_read_attachment_rejects_path_escape(tmp_path, fake_backend_env) -> None:
    async with MCPToolClient(task_root=tmp_path, allowed={"read_attachment"}) as client:
        with pytest.raises(ToolExecutionError, match="outside task root"):
            await client.call("read_attachment", {"path": "../secret.txt"})
```

Add tests for MCP process startup/shutdown, backend timeout, 64 KiB output truncation, HTTP status errors, HTML text extraction, Code Interpreter tool configuration, attachment allowlisting, and redaction of `OPENAI_API_KEY`/`HF_TOKEN` from errors.

- [ ] **Step 2: Run tests and confirm missing MCP modules**

Run: `uv run pytest tests/test_mcp.py -q`

Expected: collection fails because the MCP client/server are absent.

- [ ] **Step 3: Implement the MCP v2 server and safe deterministic tools**

```python
from mcp.server import MCPServer


def build_server(settings: MCPServerSettings) -> MCPServer:
    server = MCPServer("gaia-tools")

    @server.tool()
    async def read_webpage(url: str) -> dict[str, object]:
        """Read public HTTP(S) page text with redirects, timeout, and size limits."""
        return await settings.web_reader.read(url)

    @server.tool()
    async def read_attachment(path: str) -> dict[str, object]:
        """Extract text and metadata from an attachment registered for this task."""
        return settings.attachment_reader.read(path)

    @server.tool()
    async def submit_answer(answer: str, evidence_ids: list[str]) -> dict[str, object]:
        """Return the concise terminal answer and evidence references."""
        return {"answer": answer.strip(), "evidence_ids": evidence_ids}

    return server
```

Implement type-aware attachment extraction for UTF-8 text/CSV/JSON, PDF via `pypdf`, and XLSX via `openpyxl`. Images/audio return metadata so `VisualAgent` or the provider can consume the original registered file. Resolve paths with `Path.resolve()` and require `resolved.is_relative_to(task_root.resolve())`.

- [ ] **Step 4: Implement hosted search and code backends behind MCP tools**

```python
search_response = await client.responses.create(
    model=settings.model,
    input=query,
    tools=[{"type": "web_search"}],
    tool_choice="required",
    include=["web_search_call.action.sources"],
    reasoning={"effort": settings.reasoning_effort},
)

code_response = await client.responses.create(
    model=settings.model,
    input=instruction,
    tools=[{
        "type": "code_interpreter",
        "container": {"type": "auto", "memory_limit": "1g", "file_ids": file_ids},
    }],
    tool_choice="required",
    include=["code_interpreter_call.outputs"],
    reasoning={"effort": settings.reasoning_effort},
)
```

Extract source URLs and code outputs into bounded structured results. Upload only files registered to the task and cache their remote file IDs for that task's lifetime.

- [ ] **Step 5: Implement the MCP v2 stdio client lifecycle**

```python
parameters = StdioServerParameters(
    command=sys.executable,
    args=["-m", "gaia_agent.mcp_server"],
    env=self._minimal_server_env(),
)
self._client_cm = Client(stdio_client(parameters))
self._client = await self._client_cm.__aenter__()
listed = await self._client.list_tools()
self.tool_specs = tuple(
    ToolSpec.from_mcp(tool) for tool in listed.tools if tool.name in self._allowed
)
```

The server module builds settings from its minimal environment and calls `server.run()` under `if __name__ == "__main__"`. The fake-backend environment used by tests returns deterministic search/code results without network access.

- [ ] **Step 6: Verify real transport behavior and commit**

Run: `uv run pytest tests/test_mcp.py -q`

Expected: tests start a subprocess, perform MCP `tools/list` and `tools/call`, enforce allowlists/path policy, and shut it down cleanly.

```bash
git add src/gaia_agent/mcp_server.py src/gaia_agent/mcp_client.py tests/test_mcp.py
git commit -m "feat: add GAIA MCP tool plane"
```

### Task 5: Specialist Agents and Structured Contracts

**Files:**
- Create: `src/gaia_agent/agents.py`
- Create: `tests/conftest.py`
- Create: `tests/test_agents.py`

**Interfaces:**
- Consumes: `ModelProvider`, `MCPToolClient`, all domain models
- Produces: `PlannerAgent.plan(task) -> AgentPlan`
- Produces: `SpecialistAgent.run(task, step, evidence) -> AgentResult`
- Produces: `CriticAgent.verify(task, candidate, evidence) -> CriticDecision`
- Tool allowlists: lookup=`web_search,read_webpage`; computation=`execute_python`; file=`read_attachment,execute_python`; visual=`read_attachment`; critic=`web_search,read_webpage,execute_python`

- [ ] **Step 1: Write failing prompt, schema, and allowlist tests**

```python
@pytest.mark.asyncio
async def test_planner_emits_bounded_specialist_steps(fake_provider, task) -> None:
    fake_provider.return_parsed({
        "restatement": "Find and verify the requested fact",
        "steps": [{"specialist": "lookup", "instruction": "Find a primary source"}],
    })
    plan = await PlannerAgent(fake_provider, max_steps=4).plan(task)
    assert plan.steps[0].specialist == SpecialistName.LOOKUP
    assert fake_provider.requests[0].allowed_tools == ()


def test_specialist_tool_allowlists_are_narrow() -> None:
    assert LookupAgent.allowed_tool_names == frozenset({"web_search", "read_webpage"})
    assert ComputationAgent.allowed_tool_names == frozenset({"execute_python"})
```

Also test planner rejection of empty/oversized plans, attachment routing, evidence IDs, source URL preservation, candidate-answer formatting, vision input detail, and critic verdict schema.

- [ ] **Step 2: Run tests and confirm missing agents**

Run: `uv run pytest tests/test_agents.py -q`

Expected: collection fails because `gaia_agent.agents` is absent.

- [ ] **Step 3: Implement one reusable specialist base with fixed role prompts**

```python
class SpecialistAgent:
    name: SpecialistName
    allowed_tool_names: frozenset[str]
    instructions: str

    async def run(
        self,
        task: GaiaTask,
        step: PlanStep,
        evidence: tuple[Evidence, ...],
        tools: MCPToolClient,
    ) -> AgentResult:
        request = ModelRequest(
            instructions=self.instructions,
            input_items=build_specialist_input(task, step, evidence),
            output_schema=agent_result_schema(),
            allowed_tools=tools.specs_for(self.allowed_tool_names),
            reasoning_effort=self.reasoning_effort,
        )
        response = await self.provider.complete(request, tools)
        return AgentResult.model_validate(response.parsed).with_usage(response.usage)
```

Prompts require observable evidence, prohibit hidden-reasoning requests, forbid inventing unavailable facts, and require concise candidates matching GAIA's format.

- [ ] **Step 4: Implement planner and grounded critic**

```python
decision = CriticDecision.model_validate(response.parsed)
if decision.verdict == "approve" and not decision.checked_evidence_ids:
    raise ValueError("approved critic decision must name checked evidence")
```

The critic prompt must independently check the candidate, use tools when evidence is insufficient, reject unsupported research answers, and never see the reference answer.

- [ ] **Step 5: Verify and commit specialists**

Run: `uv run pytest tests/test_agents.py -q`

```bash
git add src/gaia_agent/agents.py tests/conftest.py tests/test_agents.py
git commit -m "feat: add GAIA specialist agents"
```

### Task 6: Orchestration and Quality/Verification Plane

**Files:**
- Create: `src/gaia_agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: planner, specialist mapping, critic, MCP client factory, `GaiaTask`, `RunConfig`
- Produces: `Orchestrator.run(task: GaiaTask) -> TaskPrediction`
- Produces: `choose_majority(predictions: Sequence[TaskPrediction], votes: int) -> TaskPrediction`

- [ ] **Step 1: Write failing state-machine tests**

```python
@pytest.mark.asyncio
async def test_orchestrator_runs_plan_act_verify_submit(orchestrator, task, calls) -> None:
    result = await orchestrator.run(task)
    assert calls == ["planner", "lookup", "critic", "submit_answer"]
    assert result.answer == "Marie Curie"
    assert result.critic_verdict == "approve"


@pytest.mark.asyncio
async def test_orchestrator_allows_only_one_correction(orchestrator_with_rejecting_critic, task) -> None:
    result = await orchestrator_with_rejecting_critic.run(task)
    assert result.error_category == "critic_rejected"
    assert orchestrator_with_rejecting_critic.correction_count == 1
```

Add tests for routing every specialist name, unknown specialist rejection, max-step/global-tool budgets, repeated-call loop detection, numerical double-check requirement, source requirement, timeouts, audit ordering, and exception-to-terminal-category mapping.

- [ ] **Step 2: Run tests and confirm missing orchestrator**

Run: `uv run pytest tests/test_orchestrator.py -q`

Expected: collection fails because `gaia_agent.orchestrator` is absent.

- [ ] **Step 3: Implement the bounded state machine**

```python
async def run(self, task: GaiaTask) -> TaskPrediction:
    async with self._tool_client_factory(task) as tools:
        plan = await self._planner.plan(task)
        state = TaskState(task=task, plan=plan)
        for step in plan.steps[: self._config.max_steps]:
            agent = self._specialists[step.specialist]
            result = await asyncio.wait_for(
                agent.run(task, step, tuple(state.evidence), tools),
                timeout=self._config.timeout_seconds,
            )
            state = state.record(result)
        candidate = state.best_candidate()
        decision = await self._critic.verify(task, candidate, tuple(state.evidence), tools)
        if decision.verdict != "approve":
            state = await self._correct_once(state, decision, tools)
            decision = await self._critic.verify(task, state.best_candidate(), tuple(state.evidence), tools)
        return await self._terminal_prediction(state, decision, tools)
```

Audit every transition before executing the next. `submit_answer` is callable only from `_terminal_prediction`; tool-call counts include critic and correction calls.

- [ ] **Step 4: Implement normalized self-consistency voting**

```python
def choose_majority(predictions: Sequence[TaskPrediction], votes: int) -> TaskPrediction:
    groups: dict[str, list[TaskPrediction]] = defaultdict(list)
    for prediction in predictions:
        groups[normalize_for_voting(prediction.answer)].append(prediction)
    winner = max(groups.values(), key=lambda group: (len(group), -group[0].run_index))
    if len(winner) <= votes // 2:
        return predictions[0].model_copy(update={"error_category": "no_consensus"})
    return winner[0].model_copy(update={"vote_count": len(winner)})
```

Use scorer-compatible normalization without consulting ground truth. Stop early only when the current count is greater than `votes // 2`.

- [ ] **Step 5: Verify and commit orchestration**

Run: `uv run pytest tests/test_orchestrator.py -q`

```bash
git add src/gaia_agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add plan act verify orchestration"
```

### Task 7: Atomic Persistence, Resume, and Privacy-Safe Reporting

**Files:**
- Create: `src/gaia_agent/persistence.py`
- Create: `src/gaia_agent/reporting.py`
- Create: `tests/test_persistence.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Produces: `RunStore.create(root: Path, manifest: RunManifest) -> RunStore`
- Produces: `RunStore.append(prediction) -> None`, `completed_task_ids() -> frozenset[str]`
- Produces: `summarize(predictions) -> EvaluationSummary`
- Produces: `write_reports(store, summary) -> tuple[Path, Path]`

- [ ] **Step 1: Write failing persistence and privacy tests**

```python
def test_checkpoint_round_trip_is_atomic_and_resumable(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append(prediction)
    assert store.completed_task_ids() == frozenset({prediction.task_id})
    assert not list(tmp_path.glob("*.tmp"))


def test_public_artifacts_exclude_gated_content(tmp_path, prediction) -> None:
    store = RunStore.create(tmp_path, manifest_for([prediction.task_id]))
    store.append(prediction)
    combined = "".join(path.read_text() for path in tmp_path.rglob("*.json*"))
    assert '"question"' not in combined
    assert '"final_answer"' not in combined
```

Add tests for manifest mismatch on resume, duplicate task rejection, audit append behavior, secret redaction, empty levels, p50/p95 latency, usage totals, verification counts, and score recomputation.

- [ ] **Step 2: Run tests and confirm missing persistence/reporting modules**

Run: `uv run pytest tests/test_persistence.py tests/test_reporting.py -q`

- [ ] **Step 3: Implement manifest and atomic JSONL checkpoints**

```python
def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append(self, prediction: TaskPrediction) -> None:
    existing = self.load_predictions()
    if prediction.task_id in {item.task_id for item in existing}:
        raise ValueError(f"duplicate prediction for {prediction.task_id}")
    public = prediction.to_public_record()
    lines = [json.dumps(item.to_public_record(), sort_keys=True) for item in [*existing, prediction]]
    _atomic_write(self.predictions_path, "\n".join(lines) + "\n")
```

Store per-task `plan.json`, `notes.jsonl`, and `audit.jsonl` under a task-ID directory after recursively redacting known secret values.

- [ ] **Step 4: Implement metrics and Markdown report**

```python
def summarize(predictions: Sequence[TaskPrediction]) -> EvaluationSummary:
    by_level = {
        level: LevelMetrics.from_predictions([p for p in predictions if p.level == level])
        for level in (1, 2, 3)
    }
    return EvaluationSummary(
        total=len(predictions),
        correct=sum(p.correct for p in predictions),
        accuracy=sum(p.correct for p in predictions) / len(predictions),
        by_level=by_level,
        total_tokens=sum(p.usage.total_tokens for p in predictions),
        total_seconds=sum(p.duration_seconds for p in predictions),
        failures=Counter(p.error_category for p in predictions if p.error_category),
        critic_verdicts=Counter(p.critic_verdict for p in predictions),
    )
```

The Markdown report must state the sample size/seed, model/reasoning level, non-comparability with the full leaderboard, overall and per-level accuracy, usage, latency, verification outcomes, and failure categories.

- [ ] **Step 5: Verify and commit persistence/reporting**

Run: `uv run pytest tests/test_persistence.py tests/test_reporting.py -q`

```bash
git add src/gaia_agent/persistence.py src/gaia_agent/reporting.py tests/test_persistence.py tests/test_reporting.py
git commit -m "feat: add resumable GAIA result reporting"
```

### Task 8: Evaluation Runner and CLI

**Files:**
- Create: `src/gaia_agent/runner.py`
- Create: `src/gaia_agent/cli.py`
- Create: `tests/test_cli_e2e.py`

**Interfaces:**
- Consumes: all earlier components
- Produces: `EvaluationRunner.run() -> EvaluationSummary`
- Produces commands: `gaia-agent evaluate`, `gaia-agent verify-results`

- [ ] **Step 1: Write a failing fully offline CLI test**

```python
def test_evaluate_cli_runs_four_planes_and_writes_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    result = runner.invoke(app, [
        "evaluate",
        "--dataset-fixture", "tests/fixtures/gaia_records.json",
        "--sample-size", "3",
        "--seed", "42",
        "--output-dir", str(tmp_path),
        "--fake-provider",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "predictions.jsonl").read_text().count("\n") == 3
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
```

Also test missing-credential preflight, invalid model/limits, resume skip behavior, stable selected IDs, one terminal result per task, `verify-results` mismatch detection, and exit code 2 for incomplete runs.

- [ ] **Step 2: Run the end-to-end test and confirm failure**

Run: `uv run pytest tests/test_cli_e2e.py -q`

Expected: collection fails because runner and CLI modules do not exist.

- [ ] **Step 3: Implement task execution, retry boundary, scoring, and checkpoints**

```python
async def run(self) -> EvaluationSummary:
    tasks, revision = self._dataset_loader(self._config)
    selected = sample_stratified(tasks, self._config.sample_size, self._config.seed)
    store = RunStore.open_or_create(self._config, selected, revision)
    for task in selected:
        if task.task_id in store.completed_task_ids():
            continue
        started = time.monotonic()
        prediction = await self._run_task_with_retry(task)
        prediction = prediction.model_copy(update={
            "correct": question_scorer(prediction.answer, task.final_answer),
            "duration_seconds": time.monotonic() - started,
        })
        store.append(prediction)
    summary = summarize(store.load_predictions())
    write_reports(store, summary)
    return summary
```

Retry only transient provider/backend failures once with bounded exponential backoff. Convert every other per-task exception to a stable incorrect prediction and continue.

- [ ] **Step 4: Implement CLI preflight and independent result verification**

```python
@app.command()
def evaluate(
    sample_size: int = 15,
    seed: int = 42,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
    votes: int = 1,
    output_dir: Path = Path("results/gaia-15-seed42"),
    resume: bool = False,
) -> None:
    config = RunConfig.from_env(
        sample_size=sample_size,
        seed=seed,
        model=model,
        reasoning_effort=reasoning_effort,
        votes=votes,
        output_dir=output_dir,
        resume=resume,
    )
    summary = asyncio.run(build_runner(config).run())
    typer.echo(f"accuracy={summary.accuracy:.2%} ({summary.correct}/{summary.total})")
```

`verify-results OUTPUT_DIR` reloads the manifest/predictions/summary, verifies unique selected task IDs and level counts, recomputes all aggregate metrics from predictions, checks artifact privacy keys, and exits nonzero on any mismatch.

- [ ] **Step 5: Run the complete offline suite and commit**

Run: `uv run pytest -q && uv run ruff check src tests`

Expected: all tests pass; the end-to-end test exercises the actual stdio MCP subprocess with fake hosted backends.

```bash
git add src/gaia_agent/runner.py src/gaia_agent/cli.py tests/test_cli_e2e.py
git commit -m "feat: add GAIA evaluation CLI"
```

### Task 9: Documentation, Credential-Gated Smoke Test, and Real 15-Task Evaluation

**Files:**
- Create: `README.md`
- Create locally (gitignored): `results/gaia-15-seed42/*`
- Modify only if verification exposes defects: source/test files named by the failing check

**Interfaces:**
- Consumes: `gaia-agent evaluate`, `gaia-agent verify-results`
- Produces: installed project, passing test suite, and verified real evaluation report

- [ ] **Step 1: Write concise installation, architecture, privacy, and run documentation**

Document:

```console
uv sync --extra dev
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
uv run gaia-agent evaluate --sample-size 15 --seed 42 --model gpt-5.6-sol
uv run gaia-agent verify-results results/gaia-15-seed42
```

Explain how to accept GAIA's gated dataset terms, the six specialist roles, MCP process boundary, hosted sandbox cost, result artifacts, resume behavior, and why a 15-task subset is not directly comparable to the full leaderboard.

- [ ] **Step 2: Run all offline quality gates from a clean environment**

Run:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv build
```

Expected: all tests pass, Ruff is clean, and both wheel and source distribution build successfully.

- [ ] **Step 3: Check credential presence without printing values**

Run:

```bash
uv run python -c 'import os; missing=[k for k in ("OPENAI_API_KEY","HF_TOKEN") if not os.getenv(k)]; print("ready" if not missing else "missing:"+",".join(missing))'
```

Expected: `ready`. If credentials are missing, stop before network/API execution and request only the missing secret setup.

- [ ] **Step 4: Run one real smoke task and inspect its audit trail**

Run:

```bash
GAIA_MAX_STEPS=1 uv run gaia-agent evaluate --sample-size 3 --seed 42 --model gpt-5.6-sol --output-dir results/smoke-3
uv run gaia-agent verify-results results/smoke-3
```

Expected: three terminal predictions, one per level; `verify-results` passes; no artifact contains `Question`, `Final answer`, `OPENAI_API_KEY`, or `HF_TOKEN` keys. Although the command selects three tasks to preserve level balance, set `max_steps=1` for this smoke run through its config override so API exposure remains bounded.

- [ ] **Step 5: Run the real 15-task evaluation and verify it independently**

Run:

```bash
uv run gaia-agent evaluate --sample-size 15 --seed 42 --model gpt-5.6-sol --output-dir results/gaia-15-seed42
uv run gaia-agent verify-results results/gaia-15-seed42
```

Expected: 15 unique predictions, exactly five per GAIA level; summary totals recompute from predictions; the Markdown report contains measured overall/per-level accuracy, latency, token usage, critic outcomes, and failures.

- [ ] **Step 6: Inspect the measured report and document limitations**

Read `results/gaia-15-seed42/report.md` and cross-check every number against `summary.json`. If failures occurred, inspect their task-local audit logs and categorize them without copying gated question/gold content into the report.

- [ ] **Step 7: Commit source and documentation, excluding run artifacts**

```bash
git add README.md
git commit -m "docs: document GAIA multi-agent evaluation"
git status --short
```

Expected: the repository is clean; `results/` remains ignored and the verified local report remains available for handoff.

## Plan Self-Review Result

- Every design requirement maps to at least one implementation task and an observable verification command.
- Function names and types are consistent across producer/consumer tasks.
- The real GAIA evaluation remains a completion requirement rather than being replaced by mocked tests.
- The local MCP transport and hosted execution sandbox are tested explicitly.
- No implementation step requires committing gated benchmark content or credentials.
