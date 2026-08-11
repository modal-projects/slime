"""Phase-1 branchability metrics for eight sibling continuations."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BranchOutcome:
    reward: float
    best_reward: float
    output_tokens: int
    sandbox_seconds: float
    judge_calls: int
    weight_versions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.reward <= 1.0:
            raise ValueError(f"branch reward outside [0, 1]: {self.reward}")
        if not 0.0 <= self.best_reward <= 1.0:
            raise ValueError(f"branch best_reward outside [0, 1]: {self.best_reward}")


@dataclass(frozen=True)
class BranchSurveyGroup:
    snapshot_id: str
    selector: str
    inherited_score: float
    inherited_best: float
    outcomes: tuple[BranchOutcome, ...]
    snapshot_latency_seconds: float = 0.0
    restore_latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if len(self.outcomes) != 8:
            raise ValueError(f"branchability survey requires 8 siblings, got {len(self.outcomes)}")

    def metrics(self, *, reward_epsilon: float = 1e-6) -> dict[str, Any]:
        rewards = [item.reward for item in self.outcomes]
        improvements = [reward - self.inherited_score for reward in rewards]
        versions = {version for item in self.outcomes for version in item.weight_versions}
        reward_std = statistics.stdev(rewards)
        useful = max(rewards) - min(rewards) > reward_epsilon
        sandbox_seconds = sum(item.sandbox_seconds for item in self.outcomes)
        return {
            "snapshot_id": self.snapshot_id,
            "selector": self.selector,
            "reward_mean": statistics.fmean(rewards),
            "reward_std": reward_std,
            "reward_iqr": _quantile(rewards, 0.75) - _quantile(rewards, 0.25),
            "reward_range": max(rewards) - min(rewards),
            "nondegenerate": useful,
            "improvement_mean": statistics.fmean(improvements),
            "improvement_probability": sum(delta > reward_epsilon for delta in improvements) / len(improvements),
            "best_of_8_uplift": max(rewards) - self.inherited_score,
            "recover_parent_best_probability": (
                sum(reward >= self.inherited_best - reward_epsilon for reward in rewards) / len(rewards)
            ),
            "solve_at_8": any(reward >= 1.0 - reward_epsilon for reward in rewards),
            "output_tokens": sum(item.output_tokens for item in self.outcomes),
            "sandbox_seconds": sandbox_seconds,
            "judge_calls": sum(item.judge_calls for item in self.outcomes),
            "snapshot_latency_seconds": self.snapshot_latency_seconds,
            "restore_latency_seconds": self.restore_latency_seconds,
            "behavior_versions": sorted(versions),
            "single_behavior_version": len(versions) <= 1,
            "sandbox_seconds_per_useful_group": sandbox_seconds if useful else None,
        }


class SurveyWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, group: BranchSurveyGroup) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"group": _group_to_dict(group), "metrics": group.metrics()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def summarize_groups(groups: list[BranchSurveyGroup]) -> dict[str, Any]:
    rows = [group.metrics() for group in groups]
    if not rows:
        raise ValueError("cannot summarize an empty branchability survey")
    selectors = sorted({str(row["selector"]) for row in rows})
    return {
        "groups": len(rows),
        "nondegenerate_rate": statistics.fmean(float(row["nondegenerate"]) for row in rows),
        "single_behavior_version_rate": statistics.fmean(float(row["single_behavior_version"]) for row in rows),
        "reward_std_mean": statistics.fmean(float(row["reward_std"]) for row in rows),
        "improvement_probability_mean": statistics.fmean(float(row["improvement_probability"]) for row in rows),
        "best_of_8_uplift_mean": statistics.fmean(float(row["best_of_8_uplift"]) for row in rows),
        "solve_at_8_rate": statistics.fmean(float(row["solve_at_8"]) for row in rows),
        "sandbox_seconds": sum(float(row["sandbox_seconds"]) for row in rows),
        "judge_calls": sum(int(row["judge_calls"]) for row in rows),
        "by_selector": {
            selector: summarize_groups([group for group in groups if group.selector == selector])
            for selector in selectors
        }
        if len(selectors) > 1
        else {},
    }


def _group_to_dict(group: BranchSurveyGroup) -> dict[str, Any]:
    value = asdict(group)
    value["outcomes"] = [asdict(item) for item in group.outcomes]
    return value


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
