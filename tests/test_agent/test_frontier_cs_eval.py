from __future__ import annotations

import pytest

from agentic_rl.eval.frontier_cs.aggregate import aggregate_samples
from agentic_rl.eval.frontier_cs.protocol import load_registry
from agentic_rl.eval.frontier_cs.split import validate_split_rows


def _row(instance_id: str, problem_id: int) -> dict:
    return {
        "label": instance_id,
        "metadata": {
            "instance_id": instance_id,
            "verifier": {"env": {"PROBLEM_ID": str(problem_id)}},
        },
    }


def _sample(instance_id: str, index: int, reward: float) -> dict:
    return {
        "index": index,
        "label": instance_id,
        "reward": reward,
        "metadata": {"instance_id": instance_id},
    }


def test_frontier_cs_registry_pins_common_avg3_protocol():
    protocol, arms = load_registry()

    assert list(arms) == ["baseline", "p25", "p50", "p75"]
    assert protocol.expected_tasks == 38
    assert protocol.samples_per_task == 3
    assert protocol.max_response_len == 24576
    assert protocol.max_context_len == 65536
    assert protocol.max_steps == 75
    assert protocol.episode_timeout_seconds == 1800
    assert protocol.think_closure is False
    assert len(protocol.train_sha256) == 64
    assert len(protocol.eval_sha256) == 64
    assert arms["baseline"].checkpoint_path.endswith(
        "qwen3.6-27b-frontier-cs-noncolocate-5n-baseline-20260710-125341"
    )
    assert arms["p75"].checkpoint_path.endswith(
        "qwen3.6-27b-frontier-cs-retro-a-final-p75-20260810-000459"
    )
    assert arms["baseline"].checkpoint_step == 79
    assert arms["p25"].checkpoint_step == 79
    assert arms["p50"].checkpoint_step == 79
    assert arms["p75"].checkpoint_step == 79
    assert arms["p50"].environment("eval-id")["FRONTIER_CS_EVAL_CKPT_STEP"] == "79"


def test_split_validation_proves_disjoint_instance_and_problem_ids():
    summary = validate_split_rows(
        [_row("train-1", 1), _row("train-2", 2)],
        [_row("eval-3", 3), _row("eval-4", 4)],
        expected_eval_tasks=2,
    )

    assert summary["train_tasks"] == 2
    assert summary["eval_tasks"] == 2
    assert summary["instance_overlap"] == 0
    assert summary["problem_id_overlap"] == 0


@pytest.mark.parametrize(
    ("train", "evaluation", "message"),
    [
        ([_row("same", 1)], [_row("same", 2)], "instance_id overlap"),
        ([_row("train", 1)], [_row("eval", 1)], "PROBLEM_ID overlap"),
    ],
)
def test_split_validation_rejects_overlap(train, evaluation, message):
    with pytest.raises(ValueError, match=message):
        validate_split_rows(train, evaluation, expected_eval_tasks=1)


def test_avg3_is_macro_average_over_exactly_three_samples_per_task():
    summary = aggregate_samples(
        [
            _sample("a", 0, 0.0),
            _sample("a", 1, 0.5),
            _sample("a", 2, 1.0),
            _sample("b", 3, 0.25),
            _sample("b", 4, 0.25),
            _sample("b", 5, 0.25),
        ],
        samples_per_task=3,
        expected_tasks=2,
    )

    assert summary["metric"] == "avg@3"
    assert summary["avg_at_k"] == pytest.approx(0.375)
    assert summary["pass_at_k"] == pytest.approx(0.5)
    assert summary["sample_solve_rate"] == pytest.approx(1 / 6)
    assert summary["num_tasks"] == 2
    assert summary["num_samples"] == 6
    assert summary["per_task"]["a"]["rewards"] == [0.0, 0.5, 1.0]


def test_avg3_rejects_incomplete_or_duplicate_attempts():
    with pytest.raises(ValueError, match="incomplete task"):
        aggregate_samples(
            [_sample("a", 0, 0.0), _sample("a", 1, 1.0)],
            samples_per_task=3,
            expected_tasks=1,
        )

    with pytest.raises(ValueError, match="duplicate sample indices"):
        aggregate_samples(
            [_sample("a", 0, 0.0), _sample("a", 0, 0.5), _sample("a", 1, 1.0)],
            samples_per_task=3,
            expected_tasks=1,
        )
