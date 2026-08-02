import random
from collections.abc import Sequence
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from gaia_agent.models import GaiaTask


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
    resolved_attachment = (resolved_root / file_path).resolve()
    if not resolved_attachment.is_relative_to(resolved_root):
        raise ValueError("attachment path escapes dataset snapshot")
    return resolved_attachment


def load_gaia_validation(token: str, cache_dir: Path | None = None) -> tuple[list[GaiaTask], str]:
    download_args: dict[str, str | Path] = {
        "repo_id": "gaia-benchmark/GAIA",
        "repo_type": "dataset",
        "token": token,
    }
    if cache_dir is not None:
        download_args["cache_dir"] = cache_dir
    snapshot_root = Path(snapshot_download(**download_args))
    records = load_dataset(str(snapshot_root), "2023_all", split="validation")
    tasks = [
        GaiaTask(
            task_id=record["task_id"],
            question=record["Question"],
            level=int(record["Level"]),
            final_answer=record["Final answer"],
            attachment_path=_resolve_attachment(snapshot_root, record.get("file_path")),
        )
        for record in records
    ]
    return tasks, snapshot_root.name
