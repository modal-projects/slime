"""Self-contained launch configuration for Frontier-CS retro replay.

This module intentionally depends only on the Slime repository and stdlib at
import time. Optional Hugging Face dependencies are imported only by the Modal
download hooks.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")
SLIME_ROOT = "/root/slime"

_YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")
_JSON_CONFIG_FIELDS = ("train_env_vars", "apply_chat_template_kwargs", "multimodal_keys")
_SLIME_SKIP = {
    "environment",
    "async_mode",
    "slime_model_script",
    "source_hf_checkpoint",
    "megatron_conversion_hf_checkpoint",
}


@dataclass(frozen=True)
class ModalLaunchConfig:
    docker_image: str
    gpu: str
    memory: tuple[int, int]
    ephemeral_disk: int
    image_run_commands: tuple[str, ...]
    image_env: dict[str, str]
    cloud: str | None = None
    region: str | None = None


class RetroSlimeConfig:
    """Concrete Slime arguments for the paired 27B retro experiment."""

    def __init__(self, env: Mapping[str, str]):
        arm = env.get("RETRO_REWARD_ARM", "final").strip().lower()
        if arm not in ("final", "best"):
            raise ValueError("RETRO_REWARD_ARM must be final or best")
        groups = max(2, _env_int(env, "RETRO_PHASE2_GROUPS", 4))
        rollouts = max(1, _env_int(env, "RETRO_PHASE2_ROLLOUTS", 1))
        retro_ratio = _bounded_fraction(env, "RETRO_GROUP_RATIO", 0.25)
        target_fraction = _bounded_fraction(env, "RETRO_TARGET_TRAJECTORY_FRACTION", 0.5)
        max_fraction_error = _bounded_fraction(env, "RETRO_MAX_FRACTION_ERROR", 0.4)
        capture_promising_ratio = _bounded_fraction(env, "RETRO_CAPTURE_PROMISING_RATIO", 0.5)
        pool_promising_ratio = _bounded_fraction(env, "RETRO_POOL_PROMISING_RATIO", 0.5)
        selector_assignment = env.get("RETRO_SELECTOR_ASSIGNMENT", "hashed").strip().lower()
        if selector_assignment not in ("hashed", "alternating", "promising", "recovery", "any"):
            raise ValueError(
                "RETRO_SELECTOR_ASSIGNMENT must be hashed, alternating, promising, recovery, or any"
            )
        pool_order = env.get("RETRO_POOL_ORDER", "newest").strip().lower()
        if pool_order not in ("newest", "fifo"):
            raise ValueError("RETRO_POOL_ORDER must be newest or fifo")

        launch_stamp = env.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"
        fraction_tag = f"p{round(target_fraction * 100):02d}"
        default_name = f"qwen3.6-27b-frontier-cs-retro-{arm}-{fraction_tag}"
        run_tag = f"{env.get('WANDB_GROUP') or default_name}-{launch_stamp}"
        resume = env.get("RESUME")
        state_tag = resume or run_tag
        resume_ckpt_step = int(env["RESUME_CKPT_STEP"]) if env.get("RESUME_CKPT_STEP") else None
        manifest_path = f"{CHECKPOINTS_PATH}/frontier_retro/{state_tag}/manifests.jsonl"
        full_topology = groups >= 32

        # Model and topology.
        self.slime_model_script = "scripts/models/qwen3.5-27B.sh"
        self.make_vocab_size_divisible_by = 32
        self.hf_checkpoint = "Qwen/Qwen3.6-27B"
        self.ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-27B_torch_dist"
        self.colocate = False
        self.actor_num_nodes = 2 if full_topology else 1
        self.actor_num_gpus_per_node = 8
        self.rollout_num_gpus = 32 if full_topology else 8
        self.update_weights_interval = 1
        self.update_weight_buffer_size = 2147483648
        self.async_mode = True
        self.rollout_max_staleness = 4 if full_topology else 1
        self.rollout_function_path = "agentic_rl.retro.rollout.generate_retro_mixed"
        self.use_fault_tolerance = True
        self.sglang_server_concurrency = 64 if full_topology else 16
        self.no_check_for_nan_in_loss_and_grad = True

        # Agentic rollout.
        self.custom_generate_function_path = "agentic_rl.retro.generate.generate"
        self.custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
        self.custom_config_path = {
            "agentic_max_steps": 75,
            "agentic_episode_timeout": 1800,
            "agentic_eval_timeout": 600,
            "agentic_exec_timeout": 60,
            "agentic_close_think_on_length": env.get("THINK_CLOSURE", "0") == "1",
            "agentic_max_think_closures": 2,
            "agentic_think_closure_budget": 4096,
            "router_policy": "consistent_hashing",
            "agentic_max_boot_retries": 3,
        }
        self.metadata_key = "metadata"
        self.prompt_data = f"{DATA_PATH}/frontier_cs/train.jsonl"
        self.input_key = "prompt"
        self.label_key = "label"
        self.apply_chat_template = False
        self.rollout_shuffle = False
        self.rm_type = None
        self.balance_data = True

        # Rollout sizing and deterministic sibling sampling.
        self.num_rollout = rollouts
        self.rollout_batch_size = groups
        self.rollout_max_response_len = 24576
        self.rollout_temperature = 1.0
        self.n_samples_per_prompt = 8
        self.num_steps_per_rollout = 1
        self.global_batch_size = groups * 8
        self.micro_batch_size = 1
        self.rollout_max_context_len = 32768 * 2
        self.sglang_reasoning_parser = "qwen3"
        self.sglang_tool_call_parser = "qwen3_coder"
        self.rollout_seed = 20260802
        self.sglang_enable_deterministic_inference = True

        # SGLang engine.
        self.rollout_num_gpus_per_engine = 2
        self.sglang_mem_fraction_static = 0.85
        self.sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))
        self.sglang_mamba_scheduler_strategy = "extra_buffer"
        self.sglang_speculative_algorithm = "EAGLE"
        self.sglang_speculative_num_steps = 3
        self.sglang_speculative_eagle_topk = 1
        self.sglang_speculative_num_draft_tokens = 4
        self.sglang_enable_dp_attention = False
        self.sglang_disable_custom_all_reduce = False
        self.qwen_gdn_backend = "flashqla"

        # DAPO is disabled for the four-group engineering smoke.
        self.dynamic_sampling_filter_path = (
            "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
            if full_topology
            else None
        )

        # Dense 27B training.
        self.tensor_model_parallel_size = 4
        self.sequence_parallel = True
        self.pipeline_model_parallel_size = 1
        self.context_parallel_size = 2
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.use_dynamic_batch_size = True
        self.max_tokens_per_gpu = 32768
        self.log_probs_chunk_size = 1024
        self.recompute_granularity = "full"
        self.recompute_method = "uniform"
        self.recompute_num_layers = 1
        self.attention_dropout = 0.0
        self.hidden_dropout = 0.0
        self.accumulate_allreduce_grads_in_fp32 = True
        self.attention_softmax_in_fp32 = True
        self.attention_backend = "flash"
        self.use_rollout_logprobs = True

        # Checkpointing and optimizer.
        self.save = f"{CHECKPOINTS_PATH}/swe_ckpts/{state_tag}"
        self.load = self.save
        self.save_interval = 5
        self.ckpt_step = resume_ckpt_step
        self.override_opt_param_scheduler = bool(resume)
        self.save_debug_rollout_data = (
            f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{state_tag}/rollout_{{rollout_id}}.pt"
        )
        self.advantage_estimator = "grpo"
        self.use_kl_loss = False
        self.kl_loss_coef = 0.0
        self.kl_loss_type = "low_var_kl"
        self.kl_coef = 0.0
        self.entropy_coef = 0.0
        self.eps_clip = 0.2
        self.eps_clip_high = 0.28
        self.optimizer = "adam"
        self.lr = 1e-6
        self.lr_decay_style = "constant"
        self.weight_decay = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.98
        self.optimizer_cpu_offload = True
        self.overlap_cpu_optimizer_d2h_h2d = True
        self.use_precision_aware_optimizer = True
        self.eval_interval = None
        self.skip_eval_before_train = True

        self.environment = {
            "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "MODAL_ENVIRONMENT": env.get("MODAL_ENVIRONMENT", "junlin-dev"),
            "ASYNC_RL_TASK_ROOT": str(DATA_PATH),
            "SLIME_AGENT_SANDBOX_CPU": "4",
            "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
            "FRONTIER_CS_JUDGE_URL": env.get("FRONTIER_CS_JUDGE_URL", ""),
            "ASYNC_RL_REWARD_SHAPE": env.get("ASYNC_RL_REWARD_SHAPE", "fractional"),
            "ASYNC_RL_OUTCOME_REWARD": arm,
            "ASYNC_RL_SOLVED_BONUS": "0",
            "ASYNC_RL_OUTCOME_GAMMA": env.get("ASYNC_RL_OUTCOME_GAMMA", "0.4"),
            "ASYNC_RL_RETRO_RUN_TAG": state_tag,
            "ASYNC_RL_RETRO_MANIFEST_PATH": manifest_path,
            "ASYNC_RL_RETRO_SNAPSHOT_KIND": env.get("RETRO_SNAPSHOT_KIND", "directory"),
            "ASYNC_RL_RETRO_SNAPSHOT_PATH": "/app",
            "ASYNC_RL_RETRO_SNAPSHOT_TTL": env.get(
                "RETRO_SNAPSHOT_TTL", str(48 * 60 * 60)
            ),
            "ASYNC_RL_RETRO_CAPTURE_STATUS": "tentative",
            "ASYNC_RL_RETRO_MIN_SCORE": env.get("RETRO_MIN_SCORE", "0.1"),
            "ASYNC_RL_RETRO_MAX_SCORE": env.get("RETRO_MAX_SCORE", "0.95"),
            "ASYNC_RL_RETRO_MIN_REMAINING": env.get("RETRO_MIN_REMAINING", "0.0"),
            "ASYNC_RL_RETRO_MIN_TURN": env.get("RETRO_MIN_TURN", "2"),
            "ASYNC_RL_RETRO_REGRESSION_DELTA": env.get(
                "RETRO_REGRESSION_DELTA", "0.1"
            ),
            "ASYNC_RL_RETRO_STAGNANT_SUBMISSIONS": env.get(
                "RETRO_STAGNANT_SUBMISSIONS", "2"
            ),
            "ASYNC_RL_RETRO_TARGET_TRAJECTORY_FRACTION": str(target_fraction),
            "ASYNC_RL_RETRO_MAX_FRACTION_ERROR": str(max_fraction_error),
            "ASYNC_RL_RETRO_SELECTOR_ASSIGNMENT": selector_assignment,
            "ASYNC_RL_RETRO_CAPTURE_PROMISING_RATIO": str(capture_promising_ratio),
            "ASYNC_RL_RETRO_SELECTOR_SEED": env.get("RETRO_SELECTOR_SEED", "20260802"),
            "ASYNC_RL_RETRO_SELECTOR_FALLBACK": env.get("RETRO_SELECTOR_FALLBACK", "0"),
            "ASYNC_RL_RETRO_GROUP_RATIO": str(retro_ratio),
            "ASYNC_RL_RETRO_POOL_PROMISING_RATIO": str(pool_promising_ratio),
            "ASYNC_RL_RETRO_POOL_ORDER": pool_order,
            "ASYNC_RL_RETRO_MIN_POLICY_AGE": env.get("RETRO_MIN_POLICY_AGE", "0"),
            "ASYNC_RL_RETRO_MAX_POLICY_AGE": env.get("RETRO_MAX_POLICY_AGE", "4"),
            "ASYNC_RL_RETRO_MAX_ATTEMPTS": env.get("RETRO_MAX_ATTEMPTS", "3"),
        }
        self.use_wandb = True
        self.wandb_project = env.get("WANDB_PROJECT")
        self.wandb_group = run_tag
        self.disable_wandb_random_suffix = True

        self.run_tag = run_tag
        self.state_tag = state_tag
        self.launch_stamp = launch_stamp
        self.reward_arm = arm
        self.target_fraction = target_fraction

    def cli_args(self) -> list[str]:
        out: list[str] = []
        for key, value in vars(self).items():
            if key.startswith("_") or key in _SLIME_SKIP or key in {
                "run_tag",
                "state_tag",
                "launch_stamp",
                "reward_arm",
                "target_fraction",
            }:
                continue
            if value is None or value is False:
                continue
            flag = f"--{key.replace('_', '-')}"
            if value is True:
                out.append(flag)
            elif isinstance(value, dict) and key in _JSON_CONFIG_FIELDS:
                out += [flag, json.dumps(value)]
            elif isinstance(value, list):
                out += [flag, *[str(item) for item in value]]
            else:
                out += [flag, str(value)]
        return out

    def total_nodes(self) -> int:
        training_gpus = self.actor_num_nodes * self.actor_num_gpus_per_node
        total_gpus = training_gpus if self.colocate else training_gpus + self.rollout_num_gpus
        if total_gpus % self.actor_num_gpus_per_node:
            raise ValueError("retro launch GPU count does not fill complete Modal nodes")
        return math.ceil(total_gpus / self.actor_num_gpus_per_node)

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.hf_checkpoint)

    def download_data(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "junlin-modal/frontier-cs",
            repo_type="dataset",
            local_dir=str(DATA_PATH),
        )
        if os.environ.get("RETRO_DOWNLOAD_USACO", "0") == "1":
            snapshot_download(
                "junlin-modal/usaco",
                repo_type="dataset",
                local_dir=str(DATA_PATH),
            )


def build_launch_configs(
    environ: Mapping[str, str] | None = None,
) -> tuple[ModalLaunchConfig, RetroSlimeConfig]:
    env = dict(os.environ if environ is None else environ)
    slime = RetroSlimeConfig(env)
    image_env = {
        key: value
        for key, value in env.items()
        if key.startswith("RETRO_")
        or key
        in {
            "WANDB_PROJECT",
            "WANDB_GROUP",
            "RESUME",
            "RESUME_CKPT_STEP",
            "ASYNC_RL_REWARD_SHAPE",
            "ASYNC_RL_OUTCOME_GAMMA",
            "THINK_CLOSURE",
            "MODAL_ENVIRONMENT",
            "FRONTIER_CS_JUDGE_URL",
        }
    }
    image_env.update(
        {
            "LAUNCH_STAMP": slime.launch_stamp,
            "MSWEA_SILENT_STARTUP": "1",
        }
    )
    modal = ModalLaunchConfig(
        docker_image="slimerl/slime:nightly-dev-20260529a",
        gpu="H200",
        memory=(1024, int(2 * 1024 * 1024)),
        ephemeral_disk=2 * 1024 * 1024,
        image_run_commands=(
            f"rm -rf {HF_CACHE_PATH}",
            "apt-get update && apt-get install -y --no-install-recommends rdma-core libibverbs1 ibverbs-providers",
            "uv pip install --system modal mini-swe-agent datasets huggingface_hub pyyaml",
        ),
        image_env=image_env,
    )
    return modal, slime


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _bounded_fraction(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(env.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def materialize_yaml_configs(slime: RetroSlimeConfig, directory: str) -> None:
    import yaml

    for field in _YAML_CONFIG_FIELDS:
        value: Any = getattr(slime, field, None)
        if not isinstance(value, dict):
            continue
        path = Path(directory) / f"{field}.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        setattr(slime, field, str(path))
