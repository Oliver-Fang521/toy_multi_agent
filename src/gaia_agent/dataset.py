import random
from collections.abc import Sequence
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from gaia_agent.models import GaiaTask

_MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024


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


def _resolve_attachment(snapshot_root: Path, file_path: str | None) -> Path | None:
    if not file_path:
        return None
    resolved_root = snapshot_root.resolve()
    candidate = resolved_root / file_path
    resolved_attachment = candidate.resolve()
    if not resolved_attachment.is_relative_to(resolved_root):
        raise ValueError("attachment path escapes dataset snapshot")
    if candidate.is_symlink() or not resolved_attachment.is_file():
        raise ValueError("attachment must be a regular materialized file")
    if resolved_attachment.stat().st_size > _MAX_ATTACHMENT_BYTES:
        raise ValueError("attachment exceeds the 32 MiB size limit")
    return resolved_attachment


def _materialized_snapshot_dir(cache_dir: Path | None, revision: str) -> Path:
    if not revision or revision in {".", ".."}:
        raise ValueError("dataset revision is invalid")
    cache_root = (
        cache_dir.expanduser().resolve()
        if cache_dir is not None
        else (Path.home() / ".cache" / "gaia-multi-agent").resolve()
    )
    materialized = (cache_root / "gaia-materialized" / revision).resolve()
    if not materialized.is_relative_to(cache_root):
        raise ValueError("materialized dataset path escapes its cache")
    return materialized


def load_gaia_validation(token: str, cache_dir: Path | None = None) -> tuple[list[GaiaTask], str]:
    download_args: dict[str, str | Path] = {
        "repo_id": "gaia-benchmark/GAIA",
        "repo_type": "dataset",
        "token": token,
    }
    if cache_dir is not None:
        download_args["cache_dir"] = cache_dir
    snapshot_root = Path(snapshot_download(**download_args))
    revision = snapshot_root.name
    materialized_dir = _materialized_snapshot_dir(cache_dir, revision)
    materialize_args = {
        **download_args,
        "revision": revision,
        "local_dir": materialized_dir,
    }
    materialized_root = Path(snapshot_download(**materialize_args)).resolve()
    if materialized_root != materialized_dir:
        raise ValueError("materialized dataset path does not match its controlled cache")
    records = load_dataset(str(materialized_root), "2023_all", split="validation")
    tasks = [
        GaiaTask(
            task_id=record["task_id"],
            question=record["Question"],
            level=int(record["Level"]),
            final_answer=record["Final answer"],
            attachment_path=_resolve_attachment(materialized_root, record.get("file_path")),
        )
        for record in records
    ]
    return tasks, revision
