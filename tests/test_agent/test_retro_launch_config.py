from __future__ import annotations

import pytest

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
            RETRO_PROMISING_CONSECUTIVE="0",
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
    assert slime.environment["ASYNC_RL_RETRO_PROMISING_CONSECUTIVE"] == "0"
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
            RETRO_PROMISING_CONSECUTIVE="2",
        )
    )

    assert slime.total_nodes() == 2
    assert slime.actor_num_nodes == 1
    assert slime.rollout_num_gpus == 8
    assert slime.rollout_batch_size == 4
    assert slime.num_rollout == 1
    assert slime.environment["ASYNC_RL_OUTCOME_REWARD"] == "best"
    assert slime.environment["ASYNC_RL_RETRO_PROMISING_CONSECUTIVE"] == "2"
    assert slime.dynamic_sampling_filter_path is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RETRO_REWARD_ARM", "bonus"),
        ("RETRO_TARGET_TRAJECTORY_FRACTION", "1.5"),
        ("RETRO_MAX_FRACTION_ERROR", "-0.1"),
        ("RETRO_STAGNANT_SUBMISSIONS", "0"),
        ("RETRO_PROMISING_CONSECUTIVE", "-1"),
        ("RETRO_SELECTOR_ASSIGNMENT", "problem_hash"),
        ("RETRO_POOL_ORDER", "random"),
    ],
)
def test_invalid_launch_ablation_values_fail_closed(name: str, value: str):
    with pytest.raises(ValueError):
        build_launch_configs(_env(**{name: value}))


def test_vanilla_mode_runs_stock_fully_async_without_retro_envs():
    modal, slime = build_launch_configs(
        _env(
            ROLLOUT_MODE="vanilla",
            RETRO_PHASE2_GROUPS="32",
            RETRO_PHASE2_ROLLOUTS="100",
            ROLLOUT_PREFETCH_BATCHES="4",
            FRESH_MAX_BEHAVIOR_LAG="4",
        )
    )

    assert slime.rollout_function_path == (
        "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
    )
    assert slime.custom_generate_function_path == "agentic_rl.generate.generate"
    # No retro env leaks into the job: without the manifest path and the
    # RETRO_ENV_SPEC task-type stamp, episodes never attempt snapshot capture.
    assert not any(key.startswith("ASYNC_RL_RETRO") for key in slime.environment)
    # Shared recipe is untouched: reward shaping, DAPO filter, sizing.
    assert slime.environment["ASYNC_RL_OUTCOME_REWARD"] == "final"
    assert slime.dynamic_sampling_filter_path is not None
    assert slime.num_rollout == 100
    assert slime.save_interval == 10
    assert slime.rollout_prefetch_batches == 4
    assert slime.rollout_max_behavior_lag == 4
    assert slime.run_tag == "qwen3.6-27b-frontier-cs-vanilla-final-20260812-130000"
    assert modal.image_env["ROLLOUT_MODE"] == "vanilla"
    # The canonical 2026-08-24 inference stack rides the base image (sglang
    # 0.5.15.post1); no in-place sglang upgrade, deterministic off by default.
    assert modal.docker_image == "slimerl/slime:nightly-dev-20260810a-cu129"
    assert not any("sglang[all]" in cmd for cmd in modal.image_run_commands)
    assert slime.sglang_enable_deterministic_inference is False


def test_default_image_has_no_sglang_upgrade():
    modal, slime = build_launch_configs(_env(RETRO_PHASE2_GROUPS="32"))
    assert modal.docker_image == "slimerl/slime:nightly-dev-20260810a-cu129"
    assert not any("sglang[all]" in cmd for cmd in modal.image_run_commands)
    assert slime.rollout_function_path == "agentic_rl.retro.rollout.generate_retro_mixed"


def test_sglang_version_knob_installs_with_torch_pinned():
    # Escape hatch for future engine probes: in-place sglang install with
    # torch pinned to the base image's version, so an incompatible closure
    # (e.g. 0.5.18 -> torch 2.13/cu13) fails the image build loudly.
    modal, _ = build_launch_configs(_env(RETRO_PHASE2_GROUPS="32", SGLANG_VERSION="0.5.16"))
    upgrade = [cmd for cmd in modal.image_run_commands if "sglang[all]==0.5.16" in cmd]
    assert len(upgrade) == 1 and 'torch==$TORCH_PIN' in upgrade[0]
    assert modal.image_env["SGLANG_VERSION"] == "0.5.16"


def test_invalid_rollout_mode_fails_closed():
    with pytest.raises(ValueError, match="ROLLOUT_MODE"):
        build_launch_configs(_env(ROLLOUT_MODE="hybrid"))


def test_behavior_lag_gate_zero_disables_enforcement():
    _, slime = build_launch_configs(
        _env(
            RETRO_PHASE2_GROUPS="32",
            FRESH_MAX_BEHAVIOR_LAG="0",
            RETRO_MAX_BEHAVIOR_LAG="0",
        )
    )
    assert slime.rollout_max_behavior_lag is None
    assert "--rollout-max-behavior-lag" not in slime.cli_args()
    assert "ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG" not in slime.environment


def test_oldp50_recreation_env_combo():
    """The exact frontiercs_oldp50_recreate.sh environment, end to end.

    Rollout semantics are faithful to old P50 (sequential legs, prefetch 1,
    ungated fresh, crossed-version drop, 600s query cap); the inference stack
    is deliberately the uniform 2026-08-24 canon (0.5.15.post1 image, det
    OFF) shared with the vanilla/retro0 arms — NOT old P50's 0.5.12 + det on.
    """
    modal, slime = build_launch_configs(
        _env(
            RETRO_PHASE2_GROUPS="32",
            RETRO_PHASE2_ROLLOUTS="100",
            RETRO_GROUP_RATIO="0.25",
            RETRO_SEQUENTIAL_LEGS="1",
            ROLLOUT_PREFETCH_BATCHES="1",
            FRESH_MAX_BEHAVIOR_LAG="0",
            RETRO_MAX_BEHAVIOR_LAG="1",
            RETRO_PREFETCH_BATCHES="0",
            AGENTIC_QUERY_TIMEOUT="600",
        )
    )
    assert slime.environment["ASYNC_RL_RETRO_SEQUENTIAL_LEGS"] == "1"
    assert slime.environment["ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG"] == "1"
    assert slime.environment["ASYNC_RL_RETRO_PREFETCH_BATCHES"] == "0"
    assert slime.environment["ASYNC_RL_RETRO_GROUP_RATIO"] == "0.25"
    assert slime.rollout_prefetch_batches == 1
    assert slime.rollout_max_behavior_lag is None
    assert slime.sglang_enable_deterministic_inference is False
    assert slime.custom_config_path["agentic_query_timeout"] == 600
    assert modal.docker_image == "slimerl/slime:nightly-dev-20260810a-cu129"
    assert not any("sglang[all]" in cmd for cmd in modal.image_run_commands)


def test_dapo_filter_zero_disables_dynamic_sampling():
    _, slime = build_launch_configs(
        _env(ROLLOUT_MODE="vanilla", RETRO_PHASE2_GROUPS="32", DAPO_FILTER="0")
    )
    assert slime.dynamic_sampling_filter_path is None
    assert "--dynamic-sampling-filter-path" not in slime.cli_args()
    modal, slime_on = build_launch_configs(_env(RETRO_PHASE2_GROUPS="32"))
    assert slime_on.dynamic_sampling_filter_path is not None
    assert modal.image_env.get("DAPO_FILTER") is None  # only passed through when set


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
