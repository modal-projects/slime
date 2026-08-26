"""Registry of every project-owned environment knob (RUNBOOK §6 debt #3).

One row per env var: type, default (as the READ SITE applies it), the module
that consumes it, and which side of the launch boundary reads it:

* ``launcher`` — read on the operator's machine by ``retro/launch_config.py``
  when composing a run (short names like ``RETRO_MIN_SCORE`` that map onto
  ``ASYNC_RL_*`` runtime knobs, plus run-shape controls).
* ``runtime``  — read inside the training/eval job by the overlay
  (``ASYNC_RL_*``, judge autostart, sandbox sizing).
* ``sandbox``  — read inside task sandboxes by baked-in scripts.

``validate_environment`` is called by ``build_launch_configs`` so a typo'd
knob fails at launch, not silently mid-run. External namespaces (``WANDB_*``,
``MODAL_*``, ``NCCL_*``, …) are not validated. Keep this file in sync with the
code: ``tests/test_agent/test_knobs.py`` scans read sites and fails on any
knob missing here.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import dataclass, field

#: Env-name prefixes this registry owns; anything else is out of scope.
VALIDATED_PREFIXES = (
    "ASYNC_RL_",
    "RETRO_",
    "FRONTIER_CS_",
    "SLIME_AGENT_",
    "AGENTIC_",
    "HARBOR_",
)


@dataclass(frozen=True)
class Knob:
    name: str
    type: str  # int | float | str | bool | path | enum
    default: str | None  # None = no default (required or optional-unset)
    consumer: str  # module that reads it
    scope: str  # launcher | runtime | sandbox
    description: str
    choices: tuple[str, ...] = field(default=())


def _k(name, type_, default, consumer, scope, description, choices=()):
    return Knob(name, type_, default, consumer, scope, description, tuple(choices))


_KNOBS: list[Knob] = [
    # ------------------------------------------------------------------ #
    # Launcher inputs (operator shell → retro/launch_config.py)
    # ------------------------------------------------------------------ #
    _k("ROLLOUT_MODE", "enum", "retro", "retro/launch_config.py", "launcher",
       "retro = mixed retro rollout; vanilla = stock fully-async control arm; "
       "eval = held-out avg@3 protocol (HeldoutEvalSlimeConfig)",
       ("retro", "vanilla", "eval")),
    _k("RETRO_REWARD_ARM", "enum", "final", "retro/launch_config.py", "launcher",
       "Outcome reward arm", ("final", "best")),
    _k("RETRO_PHASE2_GROUPS", "int", "4", "retro/launch_config.py", "launcher",
       "Prompt groups per step (32+ = full 6-node topology + DAPO)"),
    _k("RETRO_PHASE2_ROLLOUTS", "int", "100 if groups>=32 else 1", "retro/launch_config.py", "launcher",
       "Training steps"),
    _k("RETRO_GROUP_RATIO", "fraction", "0.25", "retro/launch_config.py", "launcher",
       "Retro share of each batch → ASYNC_RL_RETRO_GROUP_RATIO"),
    _k("RETRO_TARGET_TRAJECTORY_FRACTION", "fraction", "0.5", "retro/launch_config.py", "launcher",
       "Capture point as a fraction of the trajectory → ASYNC_RL_RETRO_TARGET_TRAJECTORY_FRACTION"),
    _k("RETRO_MAX_FRACTION_ERROR", "fraction", "0.4", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MAX_FRACTION_ERROR"),
    _k("RETRO_CAPTURE_PROMISING_RATIO", "fraction", "0.5", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_CAPTURE_PROMISING_RATIO"),
    _k("RETRO_POOL_PROMISING_RATIO", "fraction", "0.5", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_POOL_PROMISING_RATIO"),
    _k("RETRO_STAGNANT_SUBMISSIONS", "int", "2", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_STAGNANT_SUBMISSIONS"),
    _k("RETRO_PROMISING_CONSECUTIVE", "int", "0", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_PROMISING_CONSECUTIVE"),
    _k("RETRO_SELECTOR_ASSIGNMENT", "enum", "hashed", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_SELECTOR_ASSIGNMENT",
       ("hashed", "alternating", "promising", "recovery", "any")),
    _k("RETRO_SELECTOR_SEED", "int", "20260802", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_SELECTOR_SEED"),
    _k("RETRO_SELECTOR_FALLBACK", "bool", "0", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_SELECTOR_FALLBACK"),
    _k("RETRO_POOL_ORDER", "enum", "newest", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_POOL_ORDER", ("newest", "fifo")),
    _k("RETRO_MIN_SCORE", "float", "0.1", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MIN_SCORE"),
    _k("RETRO_MAX_SCORE", "float", "0.95", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MAX_SCORE"),
    _k("RETRO_MIN_REMAINING", "float", "0.0", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MIN_REMAINING"),
    _k("RETRO_MIN_TURN", "int", "2", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MIN_TURN"),
    _k("RETRO_REGRESSION_DELTA", "float", "0.1", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_REGRESSION_DELTA"),
    _k("RETRO_MIN_POLICY_AGE", "int", "0", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MIN_POLICY_AGE"),
    _k("RETRO_MAX_POLICY_AGE", "int", "4", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MAX_POLICY_AGE"),
    _k("RETRO_MAX_BEHAVIOR_LAG", "int", "1", "retro/launch_config.py", "launcher",
       "Retro-lane hard lag gate; <= 0 omits the env (falls back to the fresh gate) "
       "→ ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG"),
    _k("RETRO_PREFETCH_BATCHES", "int", "0", "retro/launch_config.py", "launcher",
       "Retro prefetch depth (0 = made-to-order) → ASYNC_RL_RETRO_PREFETCH_BATCHES"),
    _k("RETRO_SEQUENTIAL_LEGS", "bool", "0", "retro/launch_config.py", "launcher",
       "Pre-2026-08-18 fresh-then-retro leg schedule → ASYNC_RL_RETRO_SEQUENTIAL_LEGS"),
    _k("RETRO_MAX_ATTEMPTS", "int", "3", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_MAX_ATTEMPTS"),
    _k("RETRO_SNAPSHOT_KIND", "enum", "directory", "retro/launch_config.py", "launcher",
       "→ ASYNC_RL_RETRO_SNAPSHOT_KIND", ("directory",)),
    _k("RETRO_SNAPSHOT_TTL", "int", "172800", "retro/launch_config.py", "launcher",
       "Snapshot TTL seconds → ASYNC_RL_RETRO_SNAPSHOT_TTL"),
    _k("RETRO_DETERMINISTIC", "bool", "0", "retro/launch_config.py", "launcher",
       "sglang deterministic inference for seed-replicate arm designs (~1.3x decode tax)"),
    _k("RETRO_DOWNLOAD_USACO", "bool", "0", "retro/launch_config.py", "launcher",
       "download_data also pulls the USACO dataset"),
    _k("ROLLOUT_PREFETCH_BATCHES", "int", "1", "retro/launch_config.py", "launcher",
       "Fresh-lane in-flight pool = prefetch × rollout_batch_size "
       "→ ASYNC_RL_ROLLOUT_PREFETCH_BATCHES"),
    _k("FRESH_MAX_BEHAVIOR_LAG", "int", "1", "retro/launch_config.py", "launcher",
       "Fresh-lane hard lag gate; <= 0 disables → ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG"),
    _k("DAPO_FILTER", "bool", "1", "retro/launch_config.py", "launcher",
       "Dynamic-sampling zero-std filter (full topology only)"),
    _k("THINK_CLOSURE", "bool", "0", "retro/launch_config.py", "launcher",
       "s1-style forced think closure on length-capped turns"),
    _k("AGENTIC_QUERY_TIMEOUT", "int", "1200", "retro/launch_config.py", "launcher",
       "Per-turn /generate cap (custom-config agentic_query_timeout)"),
    _k("SGLANG_VERSION", "str", None, "retro/launch_config.py", "launcher",
       "In-place sglang upgrade in the train image (torch pinned; unsatisfiable = loud build failure)"),
    _k("LAUNCH_STAMP", "str", "now()", "retro/launch_config.py", "launcher",
       "Shared launch timestamp (pins run_tag / eval_id)"),
    _k("RESUME", "str", None, "retro/launch_config.py", "launcher",
       "state_tag of the run to resume (checkpoints + manifests)"),
    _k("RESUME_CKPT_STEP", "int", None, "retro/launch_config.py", "launcher",
       "Iteration to load when resuming"),
    _k("FRONTIER_CS_EVAL_ARM", "str", None, "retro/launch_config.py", "launcher",
       "Held-out eval arm key (registry key fills checkpoint identity)"),
    _k("FRONTIER_CS_EVAL_RUN_TAG", "str", "from registry", "retro/launch_config.py", "launcher",
       "Checkpoint run tag under /checkpoints/swe_ckpts (required for unregistered arms)"),
    _k("FRONTIER_CS_EVAL_ID", "str", "frontier-cs-heldout-avg3-<arm>-<stamp>", "retro/launch_config.py", "launcher",
       "Stable output/W&B tag; reuse across train + post_process_data"),
    _k("FRONTIER_CS_EVAL_CKPT_STEP", "int", "from registry", "retro/launch_config.py", "launcher",
       "Checkpoint iteration to evaluate"),
    _k("FRONTIER_CS_EVAL_LOAD", "path", "from registry", "retro/launch_config.py", "launcher",
       "Absolute checkpoint root override (non-training checkpoints)"),
    _k("FRONTIER_CS_EVAL_REFRESH_DATA", "bool", "0", "retro/launch_config.py", "launcher",
       "Re-pull the frontier_cs dataset even if the split exists"),
    # ------------------------------------------------------------------ #
    # Runtime knobs (inside the job; mostly exported by launch_config)
    # ------------------------------------------------------------------ #
    _k("ASYNC_RL_TASK_ROOT", "path", None, "environment/harbor.py", "runtime",
       "Directory metadata.task_path resolves against (= /data on Modal); required"),
    _k("ASYNC_RL_REWARD_SHAPE", "enum", "fractional", "environment/rewards.py", "runtime",
       "Reward shaper", ("fractional", "binary", "thresholded")),
    _k("ASYNC_RL_REWARD_THRESHOLD", "float", "0.1", "environment/rewards.py", "runtime",
       "thresholded shaper cutoff"),
    _k("ASYNC_RL_OUTCOME_REWARD", "enum", "final", "environment/rewards.py", "runtime",
       "Episode outcome strategy", ("final", "best", "disc_sum")),
    _k("ASYNC_RL_SOLVED_BONUS", "float", "0", "environment/rewards.py", "runtime",
       "Additive bonus on solved episodes"),
    _k("ASYNC_RL_OUTCOME_GAMMA", "float", "0.4", "environment/rewards.py", "runtime",
       "disc_sum discount"),
    _k("ASYNC_RL_TURN_REWARD", "str", "", "turn_reward.py", "runtime",
       "Per-turn reward strategy ('' = off; e.g. kevin_sum)"),
    _k("ASYNC_RL_TURN_MIX_WEIGHT", "float", "0.3", "turn_advantage.py", "runtime",
       "omega mixing per-turn advantages into GRPO"),
    _k("ASYNC_RL_ROLLOUT_PREFETCH_BATCHES", "int", None, "core/fully_async.py", "runtime",
       "De-forked capacity knob; <= 0 or unset = engine cap"),
    _k("ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG", "int", None, "core/fully_async.py", "runtime",
       "De-forked hard per-group staleness bound; unset/<= 0 = ungated"),
    _k("SLIME_AGENT_SANDBOX_CPU", "float", None, "sandbox.py", "runtime",
       "Task sandbox CPU cores"),
    _k("SLIME_AGENT_SANDBOX_MEMORY_MB", "int", None, "sandbox.py", "runtime",
       "Task sandbox memory"),
    _k("AGENTIC_SANDBOX_VM_RUNTIME", "bool", "0", "environment/harbor.py", "runtime",
       "Boot task sandboxes as VMs instead of gVisor"),
    _k("AGENT_EVAL_TIMEOUT_SEC", "int", "600", "environment/harbor.py", "runtime",
       "Oracle-check CLI eval timeout"),
    _k("HARBOR_VERIFY_DEBUG", "bool", None, "environment/harbor.py", "runtime",
       "Verbose harbor verifier logging"),
    _k("FRONTIER_CS_SERVER_VERIFY", "bool", "1", "environment/frontiercs.py", "runtime",
       "Server-side judge scores (contract #8: sandbox submissions log is untrusted)"),
    _k("FRONTIER_CS_SUBMIT_MAX_POLL_TIME", "int", "600", "environment/frontiercs.py", "runtime",
       "Judge poll window per submit"),
    _k("FRONTIER_CS_JUDGE_URL", "str", "", "environment/frontiercs.py", "runtime",
       "Pre-deployed judge URL ('' = per-worker autostart)"),
    _k("FRONTIER_CS_JUDGE_APP", "str", "frontier-cs-judge", "environment/verifier_server/autostart.py", "runtime",
       "Modal app name for the autostarted judge"),
    _k("FRONTIER_CS_JUDGE_CPU", "float", None, "environment/verifier_server/autostart.py", "runtime",
       "Judge container CPUs"),
    _k("FRONTIER_CS_JUDGE_MEMORY_MB", "int", None, "environment/verifier_server/autostart.py", "runtime",
       "Judge container memory"),
    _k("FRONTIER_CS_JUDGE_IDLE_EXIT_MIN", "int", "25", "environment/verifier_server/autostart.py", "runtime",
       "Judge idle self-shutdown"),
    _k("FRONTIER_CS_DATA_VOLUME", "str", "slime-data", "environment/verifier_server/autostart.py", "runtime",
       "Volume the judge mounts for problems/"),
    _k("FRONTIER_CS_PROBLEMS_ROOT", "path", "/data/frontier_cs/problems", "environment/verifier_server/autostart.py", "runtime",
       "Judge testdata root"),
    _k("ASYNC_RL_RETRO_RUN_TAG", "str", "", "retro/env.py", "runtime",
       "Run identity stamped into manifests"),
    _k("ASYNC_RL_RETRO_MANIFEST_PATH", "path", None, "retro/rollout.py", "runtime",
       "Volume-side manifest JSONL ledger; REQUIRED for retro mode"),
    _k("ASYNC_RL_RETRO_SNAPSHOT_KIND", "enum", "directory", "retro/env.py", "runtime",
       "Snapshot mechanism", ("directory",)),
    _k("ASYNC_RL_RETRO_SNAPSHOT_PATH", "path", "/app", "retro/env.py", "runtime",
       "Sandbox-side SOURCE dir to photograph (P9: rename to _SNAPSHOT_SOURCE_DIR)"),
    _k("ASYNC_RL_RETRO_SNAPSHOT_TTL", "int", "172800", "retro/env.py", "runtime",
       "Modal image TTL seconds"),
    _k("ASYNC_RL_RETRO_CAPTURE_STATUS", "enum", "available", "retro/env.py", "runtime",
       "Status new manifests enter with (tentative = batch-accept commits them)",
       ("tentative", "available")),
    _k("ASYNC_RL_RETRO_MIN_SCORE", "float", "0.1", "retro/env.py", "runtime",
       "EventSelector: minimum submission score to stage"),
    _k("ASYNC_RL_RETRO_MAX_SCORE", "float", "0.95", "retro/env.py", "runtime",
       "EventSelector: solved-enough ceiling"),
    _k("ASYNC_RL_RETRO_MIN_REMAINING", "float", "0.0", "retro/env.py", "runtime",
       "Minimum remaining-budget fraction at capture"),
    _k("ASYNC_RL_RETRO_MIN_TURN", "int", "2", "retro/env.py", "runtime",
       "Earliest capture turn"),
    _k("ASYNC_RL_RETRO_REGRESSION_DELTA", "float", "0.1", "retro/env.py", "runtime",
       "RECOVERY selector: score drop that counts as a regression"),
    _k("ASYNC_RL_RETRO_STAGNANT_SUBMISSIONS", "int", "2", "retro/env.py", "runtime",
       "RECOVERY selector: stagnant submissions before staging"),
    _k("ASYNC_RL_RETRO_PROMISING_CONSECUTIVE", "int", "0", "retro/env.py", "runtime",
       "PROMISING selector: consecutive improvements required"),
    _k("ASYNC_RL_RETRO_TARGET_TRAJECTORY_FRACTION", "fraction", "0.5", "retro/env.py", "runtime",
       "Preferred capture point along the trajectory"),
    _k("ASYNC_RL_RETRO_MAX_FRACTION_ERROR", "fraction", "0.4", "retro/env.py", "runtime",
       "Tolerated |capture point − target|"),
    _k("ASYNC_RL_RETRO_SELECTOR_ASSIGNMENT", "enum", "hashed", "retro/env.py", "runtime",
       "Event-type assignment policy",
       ("hashed", "alternating", "promising", "recovery", "any")),
    _k("ASYNC_RL_RETRO_SELECTOR_SEED", "int", "20260802", "retro/env.py", "runtime",
       "Hashed-assignment seed"),
    _k("ASYNC_RL_RETRO_SELECTOR_FALLBACK", "bool", "0", "retro/env.py", "runtime",
       "Fall back to the other event type when the assigned one never fires"),
    _k("ASYNC_RL_RETRO_CAPTURE_PROMISING_RATIO", "fraction", "0.5", "retro/env.py", "runtime",
       "Capture-side PROMISING share"),
    _k("ASYNC_RL_RETRO_POOL_PROMISING_RATIO", "fraction", "0.5", "retro/rollout.py", "runtime",
       "Lease-side PROMISING quota"),
    _k("ASYNC_RL_RETRO_POOL_ORDER", "enum", "newest", "retro/rollout.py", "runtime",
       "Lease order", ("newest", "fifo")),
    _k("ASYNC_RL_RETRO_GROUP_RATIO", "fraction", "0.25", "retro/rollout.py", "runtime",
       "Retro share of each training batch"),
    _k("ASYNC_RL_RETRO_MIN_POLICY_AGE", "int", "0", "retro/rollout.py", "runtime",
       "Snapshot policy-age lower bound at lease"),
    _k("ASYNC_RL_RETRO_MAX_POLICY_AGE", "int", "4", "retro/rollout.py", "runtime",
       "Snapshot policy-age upper bound at lease"),
    _k("ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG", "int", "fresh gate", "retro/rollout.py", "runtime",
       "Retro-lane branch-generation lag gate"),
    _k("ASYNC_RL_RETRO_MAX_ATTEMPTS", "int", "3", "retro/rollout.py", "runtime",
       "Lease attempts per made-to-order slot"),
    _k("ASYNC_RL_RETRO_PREFETCH_BATCHES", "int", "0", "retro/prefetch.py", "runtime",
       "Retro prefetch depth (0 = no worker, made-to-order)"),
    _k("ASYNC_RL_RETRO_SEQUENTIAL_LEGS", "bool", "0", "retro/rollout.py", "runtime",
       "Old-P50 sequential leg schedule (prefetch 0 only)"),
    _k("ASYNC_RL_RETRO_CODE_FINGERPRINT", "str", "", "retro/env.py", "runtime",
       "Compatibility fingerprint stamped into manifests"),
    _k("ASYNC_RL_RETRO_DATA_FINGERPRINT", "str", None, "retro/env.py", "runtime",
       "Compatibility fingerprint stamped into manifests"),
    _k("ASYNC_RL_RETRO_BASE_IMAGE_FINGERPRINT", "str", None, "retro/env.py", "runtime",
       "Compatibility fingerprint stamped into manifests"),
]

REGISTRY: dict[str, Knob] = {knob.name: knob for knob in _KNOBS}
assert len(REGISTRY) == len(_KNOBS), "duplicate knob names in registry"


def unknown_knobs(env: Mapping[str, str]) -> list[str]:
    """Env names carrying a validated prefix that the registry doesn't know."""

    return sorted(
        name
        for name in env
        if name.startswith(VALIDATED_PREFIXES) and name not in REGISTRY
    )


def validate_environment(env: Mapping[str, str]) -> None:
    """Fail launch on typo'd project knobs, with a did-you-mean hint."""

    problems = []
    for name in unknown_knobs(env):
        hint = difflib.get_close_matches(name, REGISTRY, n=1)
        suffix = f" (did you mean {hint[0]}?)" if hint else ""
        problems.append(f"{name}{suffix}")
    if problems:
        raise ValueError(
            "unknown environment knob(s): "
            + "; ".join(problems)
            + " — every project knob must be registered in agentic_rl/knobs.py"
        )
