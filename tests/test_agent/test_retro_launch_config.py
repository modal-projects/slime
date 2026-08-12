from __future__ import annotations

import pytest

from agentic_rl.retro.launch_config import build_launch_configs


def _env(**overrides: str) -> dict[str, str]:
    return {
        "LAUNCH_STAMP": "20260812-130000",
        "WANDB_PROJECT": "Modal",
        "MODAL_ENVIRONMENT": "junlin-dev",
        **overrides,
    }


def test_full_p75_arm_a_launch_is_self_contained():
    modal, slime = build_launch_configs(
        _env(
            RETRO_REWARD_ARM="final",
            RETRO_TARGET_TRAJECTORY_FRACTION="0.75",
            RETRO_MAX_FRACTION_ERROR="0.40",
            RETRO_CAPTURE_PROMISING_RATIO="0.50",
            RETRO_POOL_PROMISING_RATIO="0.50",
            RETRO_POOL_ORDER="newest",
            RETRO_PHASE2_GROUPS="32",
            RETRO_PHASE2_ROLLOUTS="20",
        )
    )

    assert slime.total_nodes() == 6
    assert slime.actor_num_nodes == 2
    assert slime.rollout_num_gpus == 32
    assert slime.rollout_batch_size == 32
    assert slime.num_rollout == 20
    assert slime.reward_arm == "final"
    assert slime.run_tag == "qwen3.6-27b-frontier-cs-retro-final-p75-20260812-130000"
    assert slime.environment["ASYNC_RL_OUTCOME_REWARD"] == "final"
    assert slime.environment["ASYNC_RL_RETRO_TARGET_TRAJECTORY_FRACTION"] == "0.75"
    assert slime.environment["ASYNC_RL_RETRO_MAX_FRACTION_ERROR"] == "0.4"
    assert slime.environment["ASYNC_RL_RETRO_POOL_PROMISING_RATIO"] == "0.5"
    assert slime.environment["ASYNC_RL_RETRO_POOL_ORDER"] == "newest"
    assert modal.image_env["LAUNCH_STAMP"] == "20260812-130000"

    cli = slime.cli_args()
    assert "--rollout-function-path" in cli
    assert "agentic_rl.retro.rollout.generate_retro_mixed" in cli
    assert "--run-tag" not in cli
    assert "--reward-arm" not in cli


def test_smoke_and_best_arm_defaults():
    _, slime = build_launch_configs(
        _env(
            RETRO_REWARD_ARM="best",
            RETRO_TARGET_TRAJECTORY_FRACTION="0.25",
        )
    )

    assert slime.total_nodes() == 2
    assert slime.actor_num_nodes == 1
    assert slime.rollout_num_gpus == 8
    assert slime.rollout_batch_size == 4
    assert slime.num_rollout == 1
    assert slime.environment["ASYNC_RL_OUTCOME_REWARD"] == "best"
    assert slime.dynamic_sampling_filter_path is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RETRO_REWARD_ARM", "bonus"),
        ("RETRO_TARGET_TRAJECTORY_FRACTION", "1.5"),
        ("RETRO_MAX_FRACTION_ERROR", "-0.1"),
        ("RETRO_SELECTOR_ASSIGNMENT", "problem_hash"),
        ("RETRO_POOL_ORDER", "random"),
    ],
)
def test_invalid_launch_ablation_values_fail_closed(name: str, value: str):
    with pytest.raises(ValueError):
        build_launch_configs(_env(**{name: value}))
