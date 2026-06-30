"""Custom rollout metrics for fully-async agentic RL, logged via
--custom-rollout-log-function-path on top of slime's defaults.

  agentic/*  episode behavior — turns, exit-status mix, gen/exec/grade timing, token split.
  async/*    off-policy health of each training batch — version_span (weight versions a
             multi-turn episode straddled), version_lag (updates-stale vs the freshest in
             the batch), versions_in_batch, sample_age_sec (gen-finish → train dwell).
"""

import logging
import re
import time
from collections import Counter

import numpy as np

logger = logging.getLogger("agentic_rl")


def _vnum(v):
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def _agentic_metrics(samples) -> dict:
    stats = [
        s.metadata["agentic"]
        for s in samples
        if isinstance(getattr(s, "metadata", None), dict) and "agentic" in s.metadata
    ]
    if not stats:
        return {}

    def mean(key):
        vals = [s[key] for s in stats if key in s]
        return float(np.mean(vals)) if vals else 0.0

    out = {
        f"agentic/{k}/mean": mean(k)
        for k in (
            "turns",
            "n_calls",
            "format_errors",
            "length_truncations",
            "gen_time",
            "exec_time",
            "boot_time",
            "grade_time",
            "episode_time",
            "output_tokens",
            "response_tokens",
            "reasoning_tokens",
            "total_length",
            "prefix_cache_hit",
            "overlong_penalty",
        )
    }
    out["agentic/format_error_tax/mean"] = mean("n_calls") - mean("turns")  # non-productive calls per episode
    for key in ("turns", "response_tokens", "output_tokens", "episode_time", "total_length"):  # long-tail maxes
        out[f"agentic/{key}/max"] = float(max((s.get(key, 0) for s in stats), default=0))
    turns_list = [s["turns"] for s in stats if "turns" in s]  # turn distribution (decide max_steps)
    for p in (50, 90, 99):
        out[f"agentic/turns/p{p}"] = float(np.percentile(turns_list, p)) if turns_list else 0.0
    # Truncation in the BROAD sense — an episode cut off by a RESOURCE LIMIT before a clean finish:
    #   turns   (hit max_steps → LimitsExceeded),
    #   tokens  (trajectory filled the 128k context window → ContextExceeded), or
    #   time    (wall / hard cap → TimeExceeded).
    # truncated_frac is the union; the sub-fracs say WHICH limit binds so we know what to raise. Separately,
    # length_turn_frac = episodes where a single turn's output was cut at the per-turn cap (finish_reason=length)
    # → the signal for rollout_max_response_len. Target: keep truncated_frac under ~0.10 (ideally 0.05).
    n = len(stats) or 1
    _trunc_exits = {"LimitsExceeded", "TimeExceeded", "ContextExceeded"}
    out["agentic/truncated_frac"] = sum(s.get("exit_status") in _trunc_exits for s in stats) / n
    out["agentic/truncated/step_cap_frac"] = mean("hit_step_cap")  # turns: hit max_steps
    out["agentic/truncated/context_frac"] = sum(s.get("exit_status") == "ContextExceeded" for s in stats) / n
    out["agentic/truncated/time_frac"] = sum(s.get("exit_status") == "TimeExceeded" for s in stats) / n
    out["agentic/truncated/length_turn_frac"] = sum(s.get("length_truncations", 0) > 0 for s in stats) / n
    gens = [(s["output_tokens"], s["gen_time"]) for s in stats if s.get("gen_time", 0) > 0 and "output_tokens" in s]
    if gens:
        out["agentic/decode_tok_per_s/mean"] = float(np.mean([o / g for o, g in gens]))
    frac = [s["gen_time"] / s["episode_time"] for s in stats if s.get("episode_time", 0) > 0 and "gen_time" in s]
    if frac:
        out["agentic/gen_frac/mean"] = float(np.mean(frac))
    # raw 0/1 correctness (sample.reward is shaped by the overlong penalty, so it can't be used here)
    out["agentic/solved_frac"] = float(np.mean([s.get("solved", 0.0) for s in stats]))
    exits = Counter(s.get("exit_status", "?") for s in stats)
    total = sum(exits.values()) or 1
    for status, c in exits.items():
        out[f"agentic/exit_frac/{status}"] = c / total
        # solved-rate WITHIN this exit status: are length/time-limited episodes losing solves they'd
        # otherwise get? (validates the overlong/length hypothesis — e.g. Submitted should solve >> LimitsExceeded)
        grp = [s.get("solved", 0.0) for s in stats if s.get("exit_status", "?") == status]
        out[f"agentic/solved_by_exit/{status}"] = float(np.mean(grp)) if grp else 0.0
    rsrc = Counter(s.get("reward_source", "?") for s in stats)  # curated submission vs git-add-A fallback
    rtotal = sum(rsrc.values()) or 1
    for src, c in rsrc.items():
        out[f"agentic/reward_source_frac/{src}"] = c / rtotal
    return out


def _async_metrics(samples, now) -> dict:
    out = {}
    versioned = [s.weight_versions for s in samples if getattr(s, "weight_versions", None)]
    if versioned:
        spans = [len(set(vs)) for vs in versioned]
        out["async/version_span/mean"] = float(np.mean(spans))
        out["async/version_span/max"] = float(max(spans))
        nums = [ns for ns in ([n for n in (_vnum(v) for v in vs) if n is not None] for vs in versioned) if ns]
        if nums:
            fresh = max(max(ns) for ns in nums)
            lags = [fresh - min(ns) for ns in nums]
            out["async/version_lag/mean"] = float(np.mean(lags))
            out["async/version_lag/max"] = float(max(lags))
            out["async/versions_in_batch"] = float(len({n for ns in nums for n in ns}))
    ages = [
        now - s.metadata["agentic"]["gen_timestamp"]
        for s in samples
        if isinstance(getattr(s, "metadata", None), dict) and s.metadata.get("agentic", {}).get("gen_timestamp")
    ]
    if ages:
        out["async/sample_age_sec/mean"] = float(np.mean(ages))
        out["async/sample_age_sec/max"] = float(max(ages))
    return out


def _dump_trajectories(args, rollout_id) -> None:
    """Persist two JSONL streams to the checkpoint volume for offline analysis (wandb Tables can't be
    logged from slime's shared-mode rollout run): agentic_trajectories/ = the last-8 full(ish) transcripts
    for qualitative spot-checks; agentic_episodes/ = one scalar row per EVERY episode (solve, exit, turns,
    lengths, reward_source, overlong) — the per-(task,time) record for quantitative analysis + curriculum."""
    import json
    import os

    from . import generate

    base = getattr(args, "save", None) or "/checkpoints"

    def _drain(ring, subdir):
        rows = list(ring)
        ring.clear()
        if not rows:
            return
        d = os.path.join(base, subdir)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"rollout_{rollout_id}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    _drain(generate._recent_trajectories, "agentic_trajectories")
    _drain(generate._episode_records, "agentic_episodes")


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    # Telemetry must never crash the rollout: wrap everything (a logging error once killed the run).
    try:
        from slime.ray.rollout import compute_rollout_step
        from slime.utils import logging_utils

        extra = {**_agentic_metrics(samples), **_async_metrics(samples, time.time())}
        if extra:
            logger.info(
                "agentic %s: %s", rollout_id, {k: round(v, 3) for k, v in extra.items() if isinstance(v, (int, float))}
            )
            extra["rollout/step"] = compute_rollout_step(args, rollout_id)
            logging_utils.log(args, extra, step_key="rollout/step")
        _dump_trajectories(args, rollout_id)
    except Exception:
        logger.exception("log_rollout_data failed (non-fatal)")
    return False  # also emit slime's default rollout/* and perf/* metrics
