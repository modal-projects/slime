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
    # De-forked knobs: travel via self.environment as ASYNC_RL_* env vars,
    # read by agentic_rl/core/fully_async.py — no longer slime CLI flags.
    "rollout_prefetch_batches",
    "rollout_max_behavior_lag",
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


class _SlimeConfigBase:
    """Shared launch mechanics: attribute→CLI-flag emission and node math."""

    # Identity/bookkeeping fields that are not slime CLI flags.
    _IDENTITY_SKIP = {
        "run_tag",
        "state_tag",
        "launch_stamp",
        "reward_arm",
        "target_fraction",
        "rollout_mode",
        "arm",
        "eval_id",
        "source_run_tag",
    }

    def cli_args(self) -> list[str]:
        out: list[str] = []
        for key, value in vars(self).items():
            if key.startswith("_") or key in _SLIME_SKIP or key in self._IDENTITY_SKIP:
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
            raise ValueError("launch GPU count does not fill complete Modal nodes")
        return math.ceil(total_gpus / self.actor_num_gpus_per_node)

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.hf_checkpoint)


class RetroSlimeConfig(_SlimeConfigBase):
    """Concrete Slime arguments for the paired 27B retro experiment."""

    def __init__(self, env: Mapping[str, str]):
        arm = env.get("RETRO_REWARD_ARM", "final").strip().lower()
        if arm not in ("final", "best"):
            raise ValueError("RETRO_REWARD_ARM must be final or best")
        groups = max(2, _env_int(env, "RETRO_PHASE2_GROUPS", 4))
        # Canonical production run since 2026-08-24: 100 updates, checkpoint
        # every 10. The bare 4-group engineering smoke keeps its 1-rollout
        # default.
        rollouts = max(1, _env_int(env, "RETRO_PHASE2_ROLLOUTS", 100 if groups >= 32 else 1))
        retro_ratio = _bounded_fraction(env, "RETRO_GROUP_RATIO", 0.25)
        target_fraction = _bounded_fraction(env, "RETRO_TARGET_TRAJECTORY_FRACTION", 0.5)
        max_fraction_error = _bounded_fraction(env, "RETRO_MAX_FRACTION_ERROR", 0.4)
        capture_promising_ratio = _bounded_fraction(env, "RETRO_CAPTURE_PROMISING_RATIO", 0.5)
        pool_promising_ratio = _bounded_fraction(env, "RETRO_POOL_PROMISING_RATIO", 0.5)
        stagnant_submissions = _env_int(env, "RETRO_STAGNANT_SUBMISSIONS", 2)
        promising_consecutive = _env_int(env, "RETRO_PROMISING_CONSECUTIVE", 0)
        if stagnant_submissions < 1:
            raise ValueError("RETRO_STAGNANT_SUBMISSIONS must be at least 1")
        if promising_consecutive < 0:
            raise ValueError("RETRO_PROMISING_CONSECUTIVE must be non-negative")
        selector_assignment = env.get("RETRO_SELECTOR_ASSIGNMENT", "hashed").strip().lower()
        if selector_assignment not in ("hashed", "alternating", "promising", "recovery", "any"):
            raise ValueError(
                "RETRO_SELECTOR_ASSIGNMENT must be hashed, alternating, promising, recovery, or any"
            )
        pool_order = env.get("RETRO_POOL_ORDER", "newest").strip().lower()
        if pool_order not in ("newest", "fifo"):
            raise ValueError("RETRO_POOL_ORDER must be newest or fifo")

        # retro (default) runs the mixed retro rollout; vanilla runs the stock
        # slime fully-async path with the plain agentic generate — no snapshot
        # capture, no retro envs, no manifest store. Everything else (model,
        # data, reward shaping, DAPO, prefetch/lag knobs) is shared, so a
        # vanilla arm is the retro stack's true no-retro control.
        rollout_mode = env.get("ROLLOUT_MODE", "retro").strip().lower()
        if rollout_mode not in ("retro", "vanilla"):
            raise ValueError("ROLLOUT_MODE must be retro or vanilla (eval routes to HeldoutEvalSlimeConfig in build_launch_configs)")

        self.rollout_mode = rollout_mode
        launch_stamp = env.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"
        fraction_tag = f"p{round(target_fraction * 100):02d}"
        if rollout_mode == "vanilla":
            default_name = f"qwen3.6-27b-frontier-cs-vanilla-{arm}"
        else:
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
        # Staleness is split into orthogonal knobs (see agentic_rl/core/fully_async.py):
        # prefetch sizes the in-flight pool (capacity/throughput only), while the
        # per-lane max_behavior_lag values are HARD per-group bounds enforced at
        # batch assembly (rollout t trains only groups whose oldest token is
        # <= lag updates behind version t+1). FRESH_MAX_BEHAVIOR_LAG binds the
        # fresh lane (exported as ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG since the
        # de-fork — the knobs travel as env vars, not slime CLI flags);
        # RETRO_MAX_BEHAVIOR_LAG binds retro
        # continuations (env, > 1 also admits mixed-version continuations).
        # Snapshot *state* age stays a separate axis (ASYNC_RL_RETRO_MIN/MAX_POLICY_AGE).
        # A value of 0 (or negative) DISABLES the corresponding hard gate —
        # the pre-2026-08-18 unenforced behavior, needed to recreate the old
        # P50 cohort faithfully.
        self.rollout_prefetch_batches = _env_int(env, "ROLLOUT_PREFETCH_BATCHES", 1)
        fresh_max_behavior_lag = _env_int(env, "FRESH_MAX_BEHAVIOR_LAG", 1)
        self.rollout_max_behavior_lag = fresh_max_behavior_lag if fresh_max_behavior_lag > 0 else None
        retro_max_behavior_lag = _env_int(env, "RETRO_MAX_BEHAVIOR_LAG", 1)
        if rollout_mode == "vanilla":
            self.rollout_function_path = "agentic_rl.core.fully_async.generate_rollout_fully_async"
        else:
            self.rollout_function_path = "agentic_rl.retro.rollout.generate_retro_mixed"
        self.use_fault_tolerance = True
        self.sglang_server_concurrency = 64 if full_topology else 16
        self.no_check_for_nan_in_loss_and_grad = True

        # Agentic rollout. The retro generate is the plain agentic generate
        # plus a task_type stamp that routes episodes to the capture-capable
        # retro env; vanilla mode uses the plain generate directly, so its
        # episodes never attempt snapshot capture.
        if rollout_mode == "vanilla":
            self.custom_generate_function_path = "agentic_rl.generate.generate"
        else:
            self.custom_generate_function_path = "agentic_rl.retro.generate.generate"
        self.custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
        self.custom_config_path = {
            "agentic_max_steps": 75,
            "agentic_episode_timeout": 1800,
            "agentic_eval_timeout": 600,
            # Per-turn /generate cap. At prefetch > 1 the engines run tens of
            # concurrent streams, so a max-length turn decodes several times
            # slower than in the near-idle regime the old 600s default assumed.
            "agentic_query_timeout": _env_int(env, "AGENTIC_QUERY_TIMEOUT", 1200),
            "agentic_exec_timeout": 60,
            "agentic_close_think_on_length": env.get("THINK_CLOSURE", "0") == "1",
            "agentic_max_think_closures": 2,
            "agentic_think_closure_budget": 4096,
            "router_policy": "consistent_hashing",
        }
        self.metadata_key = "metadata"
        self.prompt_data = f"{DATA_PATH}/frontier_cs/train.jsonl"
        self.input_key = "prompt"
        self.label_key = "label"
        self.apply_chat_template = False
        self.rollout_shuffle = False
        self.rm_type = None
        self.balance_data = True

        # Rollout sizing. Deterministic inference is OFF by default since the
        # 2026-08-18 inference A/B (profiles/inference_ab/REPORT.md): it costs
        # ~1.3x per-stream decode (pytorch sampler + NCCL-tree all-reduce +
        # batch-invariant kernels) and training doesn't need reproducible
        # sampling. Set RETRO_DETERMINISTIC=1 only for seed-replicate arm
        # designs (e.g. the rlag4/rlag8 pair), where matched sampling is the
        # point of the experiment.
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
        self.sglang_enable_deterministic_inference = env.get("RETRO_DETERMINISTIC", "0") == "1"

        # SGLang engine.
        self.rollout_num_gpus_per_engine = 2
        self.sglang_mem_fraction_static = 0.85
        # sglang 0.5.15 split cuda_graph_bs into decode/prefill lists; this is
        # the old decode-capture coverage (prefill graphs keep their default).
        # It also removed mamba_scheduler_strategy ("extra_buffer" here since
        # the knob study) — the reworked mamba radix-cache defaults replace it.
        self.sglang_cuda_graph_bs_decode = [1, 2, 4, 8, 16] + list(range(24, 257, 8))
        self.sglang_speculative_algorithm = "EAGLE"
        self.sglang_speculative_num_steps = 3
        self.sglang_speculative_eagle_topk = 1
        self.sglang_speculative_num_draft_tokens = 4
        self.sglang_enable_dp_attention = False
        self.sglang_disable_custom_all_reduce = False
        self.qwen_gdn_backend = "flashqla"

        # DAPO is disabled for the four-group engineering smoke. DAPO_FILTER=0
        # disables it for a full run (plain GRPO: every generated group trains;
        # zero-std groups contribute zero advantage — safe, the group
        # normalization is mean-centered before the std+1e-6 division — but
        # dilute the batch instead of costing extra generation). Note with the
        # filter off, dynamic_sampling/* metrics (incl. raw_reward_all) are not
        # logged; rollout/raw_reward is then itself the unbiased pre-filter
        # mean, since nothing is filtered.
        self.dynamic_sampling_filter_path = (
            "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
            if full_topology and env.get("DAPO_FILTER", "1") != "0"
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
        # Interval saves land at iters 9, 19, ..., 99. The final one coincides
        # with job teardown and can miss the Modal volume's last commit (5 of 6
        # staleness-cohort runs lost iter-84's metadata this way), so treat the
        # second-to-last checkpoint as the safe endpoint until the durability
        # fix (explicit volume commit after the final save) lands.
        self.save_interval = 10
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
            "ASYNC_RL_ROLLOUT_PREFETCH_BATCHES": str(self.rollout_prefetch_batches),
        }
        if self.rollout_max_behavior_lag is not None:
            self.environment["ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG"] = str(self.rollout_max_behavior_lag)
        if rollout_mode == "retro":
            self.environment.update(
                {
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
                    "ASYNC_RL_RETRO_STAGNANT_SUBMISSIONS": str(stagnant_submissions),
                    "ASYNC_RL_RETRO_PROMISING_CONSECUTIVE": str(promising_consecutive),
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
                    # 0 keeps the made-to-order retro lane (effectively synchronous:
                    # measured behavior lag <= 1, and a straggler barrier that gated
                    # 68-73% of steps in rlag4/rlag8). > 0 pre-generates retro groups
                    # in a background worker so they age across updates, which both
                    # removes the barrier and makes RETRO_MAX_BEHAVIOR_LAG bind.
                    "ASYNC_RL_RETRO_PREFETCH_BATCHES": env.get("RETRO_PREFETCH_BATCHES", "0"),
                    # Pre-2026-08-18 leg schedule: fresh completes, then retro,
                    # with the buffer built after the fresh leg so age-0
                    # snapshots are leasable. Only meaningful at prefetch 0.
                    "ASYNC_RL_RETRO_SEQUENTIAL_LEGS": env.get("RETRO_SEQUENTIAL_LEGS", "0"),
                    "ASYNC_RL_RETRO_MAX_ATTEMPTS": env.get("RETRO_MAX_ATTEMPTS", "3"),
                }
            )
            if retro_max_behavior_lag > 0:
                self.environment["ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG"] = str(retro_max_behavior_lag)
            # <= 0 omits the env, so the retro lane falls back to the fresh
            # flag; with both disabled the lane is ungated (old behavior).
        self.use_wandb = True
        self.wandb_project = env.get("WANDB_PROJECT")
        self.wandb_group = run_tag
        self.disable_wandb_random_suffix = True

        self.run_tag = run_tag
        self.state_tag = state_tag
        self.launch_stamp = launch_stamp
        self.reward_arm = arm
        self.target_fraction = target_fraction

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



class HeldoutEvalSlimeConfig(_SlimeConfigBase):
    """Eval-only slime arguments for the strict Frontier-CS held-out avg@3 protocol.

    Ported from the guide repo's ``w_qwen3_6_27b_frontier_cs_heldout_avg3``
    (RUNBOOK §7 step 4), so held-out evals launch from this repo alone via
    ``ROLLOUT_MODE=eval``. Sampling/limit pins come from
    ``eval/frontier_cs/arms.json`` (the one protocol source of truth);
    topology and the image pin live here. ``num_rollout=0`` +
    ``eval_interval=1`` route slime's train.py through its eval-only branch:
    evaluate once, dump ``rollout_eval_0.pt``, exit — no optimizer step.

    Identity env vars: ``FRONTIER_CS_EVAL_ARM`` (required — a registry key
    fills checkpoint identity automatically; an unregistered ad-hoc arm also
    needs ``FRONTIER_CS_EVAL_RUN_TAG``), optional ``FRONTIER_CS_EVAL_ID``,
    ``FRONTIER_CS_EVAL_CKPT_STEP``, ``FRONTIER_CS_EVAL_LOAD``,
    ``FRONTIER_CS_EVAL_REFRESH_DATA``.
    """

    def __init__(self, env: Mapping[str, str]):
        from agentic_rl.eval.frontier_cs.protocol import load_registry

        protocol, arms = load_registry()
        arm = env.get("FRONTIER_CS_EVAL_ARM", "").strip()
        if not arm:
            raise ValueError("ROLLOUT_MODE=eval requires FRONTIER_CS_EVAL_ARM")
        spec = arms.get(arm)
        source_run_tag = env.get("FRONTIER_CS_EVAL_RUN_TAG", "").strip() or (
            spec.source_run_tag if spec else ""
        )
        if not source_run_tag:
            raise ValueError(
                f"arm {arm!r} is not in the registry; set FRONTIER_CS_EVAL_RUN_TAG explicitly"
            )
        ckpt_step = (
            int(env["FRONTIER_CS_EVAL_CKPT_STEP"])
            if env.get("FRONTIER_CS_EVAL_CKPT_STEP")
            else (spec.checkpoint_step if spec else None)
        )
        load_override = env.get("FRONTIER_CS_EVAL_LOAD") or (spec.load_path if spec else None)
        launch_stamp = env.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"
        eval_id = env.get("FRONTIER_CS_EVAL_ID") or f"frontier-cs-heldout-avg3-{arm}-{launch_stamp}"
        dump_dir = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/frontier_cs/heldout_avg3/{eval_id}"

        # Identity/bookkeeping (skipped by cli_args via _IDENTITY_SKIP).
        self.rollout_mode = "eval"
        self.arm = arm
        self.eval_id = eval_id
        self.source_run_tag = source_run_tag
        self.run_tag = eval_id
        self.state_tag = eval_id
        self.launch_stamp = launch_stamp
        self.reward_arm = "final"
        self.target_fraction = 0.0

        # Model and eval-only topology: TP4×CP2 fits the model on one train
        # node; four TP2 rollout engines fit another (2 nodes total).
        self.slime_model_script = "scripts/models/qwen3.5-27B.sh"
        self.make_vocab_size_divisible_by = 32
        self.hf_checkpoint = "Qwen/Qwen3.6-27B"
        self.ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-27B_torch_dist"
        self.colocate = False
        self.actor_num_nodes = 1
        self.actor_num_gpus_per_node = 8
        self.rollout_num_gpus = 8
        self.update_weights_interval = 1
        self.update_weight_buffer_size = 2147483648
        self.async_mode = False  # sync train.py owns the eval-only branch
        self.sglang_server_concurrency = 16

        # Eval-only branch: no rollouts, one armed eval, dummy 1-iter LR
        # schedule so Megatron's `assert lr_decay_steps > 0` passes.
        self.num_rollout = 0
        self.eval_interval = 1
        self.lr_decay_iters = 1
        self.skip_eval_before_train = True

        # Checkpoint under evaluation. One shared loading path for ordinary-RL
        # and retro arms; load_path overrides for non-training checkpoints
        # (e.g. the vanilla base-model torch_dist conversion).
        self.load = load_override or f"{CHECKPOINTS_PATH}/swe_ckpts/{source_run_tag}"
        self.ckpt_step = ckpt_step
        self.no_load_optim = True
        self.no_load_rng = True
        self.override_opt_param_scheduler = True

        # Agentic episode recipe — pinned by the registry protocol.
        self.custom_generate_function_path = "agentic_rl.generate.generate"
        self.custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
        self.custom_config_path = {
            "agentic_max_steps": protocol.max_steps,
            "agentic_episode_timeout": protocol.episode_timeout_seconds,
            "agentic_eval_timeout": protocol.verifier_timeout_seconds,
            "agentic_exec_timeout": protocol.exec_timeout_seconds,
            "agentic_close_think_on_length": protocol.think_closure,
            "agentic_max_think_closures": 2,
            "agentic_think_closure_budget": 4096,
            "router_policy": "consistent_hashing",
        }
        self.metadata_key = "metadata"
        self.prompt_data = f"{DATA_PATH}/frontier_cs/train.jsonl"
        self.input_key = "prompt"
        self.label_key = "label"
        self.apply_chat_template = False
        self.rollout_shuffle = False
        self.rm_type = None
        self.balance_data = True

        # Sampling — every pin from the shared avg@3 protocol. Deterministic
        # inference is ON here (unlike training): arms must be re-runnable.
        self.num_steps_per_rollout = 1
        self.rollout_batch_size = 32
        self.global_batch_size = 256
        self.micro_batch_size = 1
        self.n_samples_per_prompt = 8
        self.n_samples_per_eval_prompt = protocol.samples_per_task
        self.rollout_max_response_len = protocol.max_response_len
        self.eval_max_response_len = protocol.max_response_len
        self.rollout_max_context_len = protocol.max_context_len
        self.rollout_temperature = protocol.temperature
        self.rollout_top_p = protocol.top_p
        self.rollout_top_k = protocol.top_k
        self.eval_temperature = protocol.temperature
        self.eval_top_p = protocol.top_p
        self.eval_top_k = protocol.top_k
        self.rollout_seed = protocol.rollout_seed
        self.sglang_enable_deterministic_inference = True
        self.eval_config = {
            "defaults": {
                "n_samples_per_eval_prompt": protocol.samples_per_task,
                "temperature": protocol.temperature,
                "top_p": protocol.top_p,
                "top_k": protocol.top_k,
                "max_response_len": protocol.max_response_len,
            },
            "datasets": [
                {
                    "name": "frontier_cs",
                    "path": f"{DATA_PATH}/frontier_cs/eval.jsonl",
                    "metadata_overrides": {"eval_dataset": "frontier_cs"},
                }
            ],
        }
        self.log_passrate = True

        # SGLang engine — wave-1 image (sglang 0.5.12): the OLD ServerArgs
        # names (cuda_graph_bs, mamba_scheduler_strategy). Do NOT use the
        # 0.5.15 names here; the pinned image's --sglang-* bridge rejects them.
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
        self.sglang_reasoning_parser = "qwen3"
        self.sglang_tool_call_parser = "qwen3_coder"
        self.qwen_gdn_backend = "flashqla"

        # Training-shape args slime validates even though no step runs.
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
        self.save = f"{CHECKPOINTS_PATH}/swe_ckpts/{eval_id}"
        self.save_interval = 15
        self.save_debug_rollout_data = f"{dump_dir}/rollout_{{rollout_id}}.pt"

        self.environment = {
            "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "MODAL_ENVIRONMENT": env.get("MODAL_ENVIRONMENT", "junlin-dev"),
            "ASYNC_RL_TASK_ROOT": str(DATA_PATH),
            "SLIME_AGENT_SANDBOX_CPU": "4",
            "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
            "FRONTIER_CS_JUDGE_URL": env.get("FRONTIER_CS_JUDGE_URL", ""),
            # Protocol pins, deliberately NOT env-overridable for eval.
            "ASYNC_RL_REWARD_SHAPE": "fractional",
            "ASYNC_RL_OUTCOME_REWARD": "final",
            "ASYNC_RL_SOLVED_BONUS": "0",
            "ASYNC_RL_OUTCOME_GAMMA": "0.4",
            "FRONTIER_CS_EVAL_ARM": arm,
            "FRONTIER_CS_EVAL_RUN_TAG": source_run_tag,
            "FRONTIER_CS_EVAL_ID": eval_id,
        }
        if ckpt_step is not None:
            self.environment["FRONTIER_CS_EVAL_CKPT_STEP"] = str(ckpt_step)
        if load_override:
            self.environment["FRONTIER_CS_EVAL_LOAD"] = load_override

        self.use_wandb = True
        self.wandb_project = env.get("WANDB_PROJECT")
        self.wandb_group = eval_id
        self.disable_wandb_random_suffix = True
        self._refresh_data = env.get("FRONTIER_CS_EVAL_REFRESH_DATA") == "1"

    def _validated_split(self) -> dict:
        from agentic_rl.eval.frontier_cs.protocol import load_registry
        from agentic_rl.eval.frontier_cs.split import validate_split_files

        protocol, _ = load_registry()
        summary = validate_split_files(
            DATA_PATH / "frontier_cs" / "train.jsonl",
            DATA_PATH / "frontier_cs" / "eval.jsonl",
            expected_eval_tasks=protocol.expected_tasks,
        )
        for key, expected in (
            ("train_sha256", protocol.train_sha256),
            ("eval_sha256", protocol.eval_sha256),
        ):
            if summary[key] != expected:
                raise ValueError(f"Frontier-CS {key} changed: expected {expected}, got {summary[key]}")
        return summary

    def download_data(self) -> None:
        """Pull Frontier-CS and prove the published 150/38 partition is disjoint."""

        train_path = DATA_PATH / "frontier_cs" / "train.jsonl"
        eval_path = DATA_PATH / "frontier_cs" / "eval.jsonl"
        if self._refresh_data or not (train_path.is_file() and eval_path.is_file()):
            from huggingface_hub import snapshot_download

            snapshot_download(
                "junlin-modal/frontier-cs", repo_type="dataset", local_dir=str(DATA_PATH)
            )
        else:
            print(
                "[frontier-cs-eval] reusing existing slime-data split; "
                "set FRONTIER_CS_EVAL_REFRESH_DATA=1 to pull"
            )
        summary = self._validated_split()
        print("[frontier-cs-eval] split validated")
        print(json.dumps(summary, indent=2))

    def post_process_data(self) -> str:
        """Write strict avg@3 results beside ``rollout_eval_0.pt``; return the path."""

        from agentic_rl.eval.frontier_cs.aggregate import aggregate_dump
        from agentic_rl.eval.frontier_cs.protocol import load_registry

        protocol, _ = load_registry()
        split = self._validated_split()
        dump_path = Path(self.save_debug_rollout_data.format(rollout_id="eval_0"))
        summary = aggregate_dump(
            dump_path,
            samples_per_task=protocol.samples_per_task,
            expected_tasks=protocol.expected_tasks,
            strict=True,
            metadata={
                "arm": self.arm,
                "source_run_tag": self.source_run_tag,
                "checkpoint_step": self.ckpt_step,
                "eval_id": self.eval_id,
                "split": split,
                "protocol": {
                    "temperature": protocol.temperature,
                    "top_p": protocol.top_p,
                    "top_k": protocol.top_k,
                    "max_response_len": protocol.max_response_len,
                    "max_context_len": protocol.max_context_len,
                    "max_steps": protocol.max_steps,
                    "episode_timeout_seconds": protocol.episode_timeout_seconds,
                    "verifier_timeout_seconds": protocol.verifier_timeout_seconds,
                    "exec_timeout_seconds": protocol.exec_timeout_seconds,
                    "think_closure": protocol.think_closure,
                    "rollout_seed": protocol.rollout_seed,
                },
            },
        )
        output_path = dump_path.with_name("summary.json")
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[frontier-cs-eval] {summary['metric']}={summary['avg_at_k']:.6f}")
        print(output_path)
        return str(output_path)


def build_launch_configs(
    environ: Mapping[str, str] | None = None,
) -> tuple[ModalLaunchConfig, _SlimeConfigBase]:
    env = dict(os.environ if environ is None else environ)
    if env.get("ROLLOUT_MODE", "").strip().lower() == "eval":
        return _build_eval_launch_configs(env)
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
            "ROLLOUT_PREFETCH_BATCHES",
            "FRESH_MAX_BEHAVIOR_LAG",
            "AGENTIC_QUERY_TIMEOUT",
            "ROLLOUT_MODE",
            "SGLANG_VERSION",
            "DAPO_FILTER",
        }
    }
    image_env.update(
        {
            "LAUNCH_STAMP": slime.launch_stamp,
            "MSWEA_SILENT_STARTUP": "1",
            # The entrypoint is imported in-container at /root/modal_train.py
            # and must resolve agentic_rl from the repo copy at /root/slime.
            # Older Modal clients auto-mounted imported local packages next to
            # the entrypoint; Modal >= 1.0 does not, so the image must put the
            # repo on sys.path itself.
            "PYTHONPATH": f"/root/Megatron-LM/:{SLIME_ROOT}",
        }
    )
    image_run_commands = [
        f"rm -rf {HF_CACHE_PATH}",
        "apt-get update && apt-get install -y --no-install-recommends rdma-core libibverbs1 ibverbs-providers",
        "uv pip install --system modal mini-swe-agent datasets huggingface_hub pyyaml",
    ]
    if sglang_version := env.get("SGLANG_VERSION", "").strip():
        # Upgrade the rollout engine in place (the base image ships
        # 0.5.15.post1 — the newest sglang upstream slime supports; 0.5.18
        # measured another +16% in rollout_sim but requires torch 2.13/cu13,
        # which the pin below rejects by design). Pin torch to whatever the
        # base image already has: Megatron is built against it, and letting
        # sglang's resolver replace it would corrupt training silently. If the
        # pin is unsatisfiable for this sglang version, the image build fails
        # loudly instead.
        image_run_commands.append(
            'TORCH_PIN=$(python -c "import torch; print(torch.__version__.split(\'+\')[0])") '
            f'&& uv pip install --system "sglang[all]=={sglang_version}" "torch==$TORCH_PIN"'
        )
    modal = ModalLaunchConfig(
        # Canonical inference stack since 2026-08-24 (rollout_sim benchmark,
        # profiles/rollout_sim/README.md): sglang 0.5.15.post1 = 844 tok/GPU/s
        # vs 728 on the old 20260529a image (sglang 0.5.12), same torch
        # 2.11+cu129 / megatron-core stack. 0.5.18 (978) needs the cu13 stack
        # and waits on upstream slime.
        docker_image="slimerl/slime:nightly-dev-20260810a-cu129",
        gpu="H200",
        memory=(1024, int(2 * 1024 * 1024)),
        ephemeral_disk=2 * 1024 * 1024,
        image_run_commands=tuple(image_run_commands),
        image_env=image_env,
    )
    return modal, slime



def _build_eval_launch_configs(env: dict[str, str]) -> tuple[ModalLaunchConfig, HeldoutEvalSlimeConfig]:
    """Held-out avg@3 eval launch (ROLLOUT_MODE=eval).

    Protocol pin: every avg@3 arm must run the SAME inference stack. Wave 1
    (2026-08-17) ran on nightly-dev-20260529a (sglang 0.5.12); the training
    image moved to 20260810a-cu129 (sglang 0.5.15.post1) on 2026-08-24, whose
    renamed ServerArgs fields also break this slime's --sglang-* bridge
    (validate_args reads sglang_data_parallel_size). Keep the old image here
    until the protocol is deliberately re-baselined.
    """

    slime = HeldoutEvalSlimeConfig(env)
    image_env = {
        key: value
        for key, value in env.items()
        if key.startswith("FRONTIER_CS_EVAL_")
        or key in {"WANDB_PROJECT", "MODAL_ENVIRONMENT", "FRONTIER_CS_JUDGE_URL", "ROLLOUT_MODE"}
    }
    image_env.update(
        {
            "LAUNCH_STAMP": slime.launch_stamp,
            "FRONTIER_CS_EVAL_ID": slime.eval_id,
            "MSWEA_SILENT_STARTUP": "1",
            "PYTHONPATH": f"/root/Megatron-LM/:{SLIME_ROOT}",
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
