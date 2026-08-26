"""The held-out results roll-up: schema, determinism, and paired-bootstrap sanity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentic_rl.eval.frontier_cs.rollup import bootstrap_se, paired_comparison, rollup

pytest.importorskip("numpy")


def _summary(rewards_by_task: dict[str, list[float]], *, eval_id: str, step: int | None):
    per_task = {
        task: {
            "rewards": rewards,
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
        }
        for task, rewards in rewards_by_task.items()
    }
    all_rewards = [r for rewards in rewards_by_task.values() for r in rewards]
    task_means = [t["mean_reward"] for t in per_task.values()]
    return {
        "avg_at_k": sum(task_means) / len(task_means),
        "pass_at_k": sum(t["max_reward"] >= 1.0 for t in per_task.values()) / len(per_task),
        "sample_solve_rate": sum(r >= 1.0 for r in all_rewards) / len(all_rewards),
        "per_task": per_task,
        "metadata": {"eval_id": eval_id, "checkpoint_step": step},
    }


TASKS_A = {f"task-{i}": [0.1 * (i % 4), 0.2, 1.0 if i == 0 else 0.0] for i in range(6)}
TASKS_B = {f"task-{i}": [0.05 * (i % 3), 0.1, 0.0] for i in range(6)}


def test_rollup_schema_matches_results_documents():
    doc = rollup(
        {
            "left": _summary(TASKS_A, eval_id="eval-left", step=79),
            "right": _summary(TASKS_B, eval_id="eval-right", step=None),
        },
        [("left", "right")],
        replicates=2000,
        seed=7,
        notes="unit test.",
        date="2026-08-26",
    )

    assert doc["evaluation"] == "frontier-cs-heldout-avg3"
    assert doc["date"] == "2026-08-26"
    assert "seed 7" in doc["notes"]
    left = doc["arms"]["left"]
    assert set(left) == {"avg_at_3", "se", "pass_at_3", "solve", "eval_id", "checkpoint_step"}
    assert left["eval_id"] == "eval-left"
    assert left["checkpoint_step"] == 79
    assert doc["arms"]["right"]["checkpoint_step"] is None
    (pair,) = doc["paired"]
    assert pair["pair"] == "left - right"
    assert set(pair) == {
        "pair",
        "mean_difference",
        "paired_standard_error",
        "bootstrap_95_interval",
        "wins_ties_losses",
    }
    assert sum(pair["wins_ties_losses"]) == len(TASKS_A)
    low, high = pair["bootstrap_95_interval"]
    assert low <= pair["mean_difference"] <= high


def test_rollup_is_deterministic_and_pair_order_flips_sign():
    kwargs = dict(replicates=1500, seed=11, date="2026-08-26")
    summaries = {
        "a": _summary(TASKS_A, eval_id="ea", step=1),
        "b": _summary(TASKS_B, eval_id="eb", step=2),
    }
    one = rollup(summaries, [("a", "b")], **kwargs)
    two = rollup(summaries, [("a", "b")], **kwargs)
    assert one == two

    flipped = rollup(summaries, [("b", "a")], **kwargs)
    assert flipped["paired"][0]["mean_difference"] == pytest.approx(
        -one["paired"][0]["mean_difference"]
    )


def test_bootstrap_se_tracks_analytic_se():
    import statistics

    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
    analytic = statistics.stdev(values) / len(values) ** 0.5
    boot = bootstrap_se(values, replicates=20000, seed=3)
    assert boot == pytest.approx(analytic, rel=0.15)


def test_paired_comparison_rejects_mismatched_task_sets():
    with pytest.raises(ValueError, match="different tasks"):
        paired_comparison({"x": 1.0}, {"y": 1.0}, replicates=10, seed=1)


def test_rollup_rejects_pairs_over_unknown_arms():
    with pytest.raises(ValueError, match="unknown arm"):
        rollup({"a": _summary(TASKS_A, eval_id="e", step=1)}, [("a", "ghost")], replicates=10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
