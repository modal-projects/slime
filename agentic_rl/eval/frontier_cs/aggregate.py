"""Aggregate a Frontier-CS eval dump into strict per-task avg@k metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _instance_id(sample: Any, index: int) -> str:
    metadata = _field(sample, "metadata", {}) or {}
    value = metadata.get("instance_id") or _field(sample, "label")
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample {index} has no metadata.instance_id or label")
    return value


def _reward(sample: Any, index: int) -> float:
    value = _field(sample, "reward")
    if isinstance(value, dict):
        for key in ("reward", "final", "score"):
            if isinstance(value.get(key), (int, float)):
                value = value[key]
                break
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"sample {index} has invalid reward {value!r}")
    return float(value)


def aggregate_samples(
    samples: Iterable[Any],
    *,
    samples_per_task: int = 3,
    expected_tasks: int = 38,
    strict: bool = True,
) -> dict[str, Any]:
    """Compute macro avg@k, preserving per-task rewards for auditability."""

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for ordinal, sample in enumerate(samples):
        instance_id = _instance_id(sample, ordinal)
        sample_index = _field(sample, "index")
        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            sample_index = ordinal
        grouped[instance_id].append((sample_index, _reward(sample, ordinal)))

    if strict and len(grouped) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, found {len(grouped)}")

    per_task: dict[str, dict[str, Any]] = {}
    incomplete: dict[str, int] = {}
    for instance_id, indexed_rewards in sorted(grouped.items()):
        indexed_rewards.sort()
        indices = [index for index, _ in indexed_rewards]
        if len(indices) != len(set(indices)):
            raise ValueError(f"{instance_id}: duplicate sample indices {indices}")
        rewards = [reward for _, reward in indexed_rewards]
        if len(rewards) != samples_per_task:
            incomplete[instance_id] = len(rewards)
            if strict:
                continue
        per_task[instance_id] = {
            "sample_indices": indices,
            "rewards": rewards,
            "mean_reward": statistics.mean(rewards),
            "max_reward": max(rewards),
            "solved_samples": sum(reward >= 1.0 - 1e-9 for reward in rewards),
        }

    if strict and incomplete:
        preview = dict(list(incomplete.items())[:10])
        raise ValueError(
            f"expected {samples_per_task} samples for every task; "
            f"{len(incomplete)} incomplete task(s), e.g. {preview}"
        )
    if not per_task:
        raise ValueError("no complete evaluation tasks found")

    task_means = [task["mean_reward"] for task in per_task.values()]
    all_rewards = [reward for task in per_task.values() for reward in task["rewards"]]
    solved_tasks = sum(task["max_reward"] >= 1.0 - 1e-9 for task in per_task.values())
    standard_error = statistics.stdev(task_means) / math.sqrt(len(task_means)) if len(task_means) > 1 else 0.0
    return {
        "metric": f"avg@{samples_per_task}",
        "avg_at_k": statistics.mean(task_means),
        "task_standard_error": standard_error,
        "pass_at_k": solved_tasks / len(per_task),
        "sample_solve_rate": sum(reward >= 1.0 - 1e-9 for reward in all_rewards) / len(all_rewards),
        "num_tasks": len(per_task),
        "samples_per_task": samples_per_task,
        "num_samples": len(all_rewards),
        "incomplete_tasks": incomplete,
        "per_task": per_task,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_dump(
    dump_path: Path,
    *,
    samples_per_task: int = 3,
    expected_tasks: int = 38,
    strict: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load Slime's ``rollout_eval_0.pt`` and aggregate its serialized samples."""

    import torch

    payload = torch.load(dump_path, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"{dump_path}: expected a mapping with a samples list")
    summary = aggregate_samples(
        payload["samples"],
        samples_per_task=samples_per_task,
        expected_tasks=expected_tasks,
        strict=strict,
    )
    return {
        **summary,
        "dump_path": str(dump_path),
        "dump_sha256": _sha256(dump_path),
        "rollout_id": payload.get("rollout_id"),
        "metadata": metadata or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="Path to rollout_eval_0.pt")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--expected-tasks", type=int, default=38)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = aggregate_dump(
        args.dump,
        samples_per_task=args.samples_per_task,
        expected_tasks=args.expected_tasks,
        strict=not args.allow_incomplete,
    )
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
