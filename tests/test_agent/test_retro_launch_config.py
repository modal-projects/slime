from __future__ import annotations

import pytest

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentic_rl.launch.launch_config import build_launch_configs


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

    assert slime.rollout_function_path == "agentic_rl.core.fully_async.generate_rollout_fully_async"
    assert slime.custom_generate_function_path == "agentic_rl.core.generate.generate"
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
    # Since the de-fork the knobs travel as env vars, never as slime CLI flags.
    assert slime.environment["ASYNC_RL_ROLLOUT_PREFETCH_BATCHES"] == "4"
    assert slime.environment["ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG"] == "4"
    cli = slime.cli_args()
    assert "--rollout-prefetch-batches" not in cli
    assert "--rollout-max-behavior-lag" not in cli
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
    assert "ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG" not in slime.environment
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


def test_train_dataset_key_switches_the_family():
    _, slime = build_launch_configs(
        _env(ROLLOUT_MODE="vanilla", TRAIN_DATASET="terminal_bench_2_1")
    )

    assert slime.train_dataset == "terminal_bench_2_1"
    assert slime.prompt_data == "/data/terminal_bench_2_1/train.split-20260826.jsonl"
    # Family wiring: no judge for harbor-native families; naming carries the key.
    assert "FRONTIER_CS_JUDGE_URL" not in slime.environment
    assert "terminal-bench-2-1" in slime.run_tag
    assert slime.custom_generate_function_path == "agentic_rl.core.generate.generate"

    _, swe = build_launch_configs(_env(ROLLOUT_MODE="vanilla", TRAIN_DATASET="swebenchpro"))
    assert swe.prompt_data == "/data/swebenchpro/train.split-20260826.jsonl"

    # frontier_cs stays the default and keeps its judge wiring.
    _, default = build_launch_configs(_env(ROLLOUT_MODE="vanilla"))
    assert default.train_dataset == "frontier_cs"
    assert default.prompt_data == "/data/frontier_cs/train.jsonl"
    assert "FRONTIER_CS_JUDGE_URL" in default.environment


def test_train_dataset_travels_to_the_container():
    modal, _ = build_launch_configs(
        _env(ROLLOUT_MODE="vanilla", TRAIN_DATASET="swebenchpro", AGENTIC_MAX_STEPS="50")
    )
    assert modal.image_env["TRAIN_DATASET"] == "swebenchpro"
    assert modal.image_env["AGENTIC_MAX_STEPS"] == "50"


def test_retro_mode_rejects_families_without_capture_support():
    with pytest.raises(ValueError, match="capture env"):
        build_launch_configs(_env(TRAIN_DATASET="terminal_bench_2_1"))
    with pytest.raises(ValueError, match="TRAIN_DATASET must be one of"):
        build_launch_configs(_env(ROLLOUT_MODE="vanilla", TRAIN_DATASET="mystery_bench"))



def test_eval_mode_builds_the_heldout_avg3_protocol():
    modal, slime = build_launch_configs(_env(ROLLOUT_MODE="eval", FRONTIER_CS_EVAL_ARM="p50"))

    # Protocol pin: eval must run the wave-1 image, never the training image.
    assert modal.docker_image == "slimerl/slime:nightly-dev-20260529a"
    assert slime.rollout_mode == "eval"
    # The registry key alone fills checkpoint identity.
    assert slime.source_run_tag.endswith("retro-a-final-p50-20260810-000459")
    assert slime.ckpt_step == 79
    assert slime.load == f"/checkpoints/swe_ckpts/{slime.source_run_tag}"
    assert slime.eval_id == "frontier-cs-heldout-avg3-p50-20260812-130000"
    assert slime.total_nodes() == 2
    assert slime.async_mode is False

    cli = slime.cli_args()
    assert cli[cli.index("--num-rollout") + 1] == "0"
    assert cli[cli.index("--eval-interval") + 1] == "1"
    assert cli[cli.index("--n-samples-per-eval-prompt") + 1] == "3"
    assert "--sglang-enable-deterministic-inference" in cli
    # The pinned 0.5.12 image needs the OLD ServerArgs names.
    assert "--sglang-cuda-graph-bs" in cli
    assert not any("cuda-graph-bs-decode" in flag for flag in cli)
    assert cli[cli.index("--sglang-mamba-scheduler-strategy") + 1] == "extra_buffer"

    # Sampling pins come from the registry protocol.
    assert slime.eval_max_response_len == 24576
    assert slime.rollout_max_context_len == 65536
    assert slime.rollout_seed == 20260802
    assert slime.custom_config_path["agentic_max_steps"] == 75
    assert slime.custom_config_path["agentic_close_think_on_length"] is False
    assert slime.eval_config["datasets"][0]["path"] == "/data/frontier_cs/eval.jsonl"

    # Reward pins are not env-overridable; no retro machinery leaks in.
    assert slime.environment["ASYNC_RL_OUTCOME_REWARD"] == "final"
    assert slime.environment["ASYNC_RL_REWARD_SHAPE"] == "fractional"
    assert not any(key.startswith("ASYNC_RL_RETRO") for key in slime.environment)


def test_eval_mode_base_arm_uses_registry_load_override():
    _, slime = build_launch_configs(_env(ROLLOUT_MODE="eval", FRONTIER_CS_EVAL_ARM="base"))

    assert slime.ckpt_step is None
    assert slime.load == "/checkpoints/Qwen3.6-27B_torch_dist"
    assert slime.environment["FRONTIER_CS_EVAL_LOAD"] == "/checkpoints/Qwen3.6-27B_torch_dist"


def test_eval_mode_unregistered_arm_requires_explicit_run_tag():
    with pytest.raises(ValueError, match="not in the registry"):
        build_launch_configs(_env(ROLLOUT_MODE="eval", FRONTIER_CS_EVAL_ARM="mystery"))

    _, slime = build_launch_configs(
        _env(
            ROLLOUT_MODE="eval",
            FRONTIER_CS_EVAL_ARM="mystery",
            FRONTIER_CS_EVAL_RUN_TAG="adhoc-run-tag",
        )
    )
    assert slime.load == "/checkpoints/swe_ckpts/adhoc-run-tag"
    assert slime.ckpt_step is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
