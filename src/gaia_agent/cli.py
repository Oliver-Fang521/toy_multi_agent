"""Command-line entrypoint for GAIA evaluation and result verification."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from gaia_agent.config import RunConfig
from gaia_agent.runner import build_runner, validate_config, verify_run_directory

app = typer.Typer(no_args_is_help=True)


@app.command()
def evaluate(
    sample_size: Annotated[int, typer.Option("--sample-size")] = 15,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    model: Annotated[str, typer.Option("--model")] = "gpt-5.6-sol",
    reasoning_effort: Annotated[str, typer.Option("--reasoning-effort")] = "medium",
    votes: Annotated[int, typer.Option("--votes")] = 1,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 8,
    max_tool_calls: Annotated[int, typer.Option("--max-tool-calls")] = 24,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 300.0,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "results/gaia-15-seed42"
    ),
    resume: Annotated[bool, typer.Option("--resume")] = False,
    dataset_fixture: Annotated[
        Path | None, typer.Option("--dataset-fixture", hidden=True)
    ] = None,
    fake_provider: Annotated[bool, typer.Option("--fake-provider", hidden=True)] = False,
) -> None:
    """Evaluate a balanced GAIA validation sample."""

    missing = [name for name in ("OPENAI_API_KEY", "HF_TOKEN") if not os.environ.get(name)]
    if missing:
        typer.echo(
            "configuration error: missing required environment variables: " + ", ".join(missing),
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        config = RunConfig.from_env(
            sample_size=sample_size,
            seed=seed,
            model=model,
            reasoning_effort=reasoning_effort,
            votes=votes,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
            resume=resume,
        )
        validate_config(config)
    except (ValidationError, ValueError):
        typer.echo("configuration error: invalid evaluation configuration", err=True)
        raise typer.Exit(code=2) from None

    try:
        runner = build_runner(
            config,
            dataset_fixture=dataset_fixture,
            fake_provider=fake_provider,
        )
        summary = asyncio.run(runner.run())
    except Exception:  # noqa: BLE001 - CLI emits a value-free failure boundary
        typer.echo("evaluation failed; no credential or gated values were printed", err=True)
        raise typer.Exit(code=2) from None
    accuracy = 0.0 if summary.accuracy is None else summary.accuracy
    typer.echo(f"accuracy={accuracy:.2%} ({summary.correct}/{summary.total})")


@app.command("verify-results")
def verify_results(output_dir: Annotated[Path, typer.Argument()]) -> None:
    """Independently verify a completed public result directory."""

    try:
        summary = verify_run_directory(output_dir)
    except Exception:  # noqa: BLE001 - verification must fail closed without leaking artifacts
        typer.echo("verification failed: results are incomplete or inconsistent", err=True)
        raise typer.Exit(code=2) from None
    accuracy = 0.0 if summary.accuracy is None else summary.accuracy
    typer.echo(f"verified accuracy={accuracy:.2%} ({summary.correct}/{summary.total})")


if __name__ == "__main__":
    app()
