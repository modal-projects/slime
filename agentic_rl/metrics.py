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
            "format_errors",
            "gen_time",
            "exec_time",
            "boot_time",
            "grade_time",
            "episode_time",
            "output_tokens",
            "response_tokens",
        )
    }
    for key in ("turns", "response_tokens", "output_tokens", "episode_time"):  # long-tail maxes
        out[f"agentic/{key}/max"] = float(max((s.get(key, 0) for s in stats), default=0))
    gens = [(s["output_tokens"], s["gen_time"]) for s in stats if s.get("gen_time", 0) > 0 and "output_tokens" in s]
    if gens:
        out["agentic/decode_tok_per_s/mean"] = float(np.mean([o / g for o, g in gens]))
    frac = [s["gen_time"] / s["episode_time"] for s in stats if s.get("episode_time", 0) > 0 and "gen_time" in s]
    if frac:
        out["agentic/gen_frac/mean"] = float(np.mean(frac))
    out["agentic/solved_frac"] = float(np.mean([1.0 if (s.reward or 0) >= 1.0 else 0.0 for s in samples]))
    exits = Counter(s.get("exit_status", "?") for s in stats)
    total = sum(exits.values()) or 1
    for status, c in exits.items():
        out[f"agentic/exit_frac/{status}"] = c / total
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


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    from slime.ray.rollout import compute_rollout_step
    from slime.utils import logging_utils

    extra = {**_agentic_metrics(samples), **_async_metrics(samples, time.time())}
    if extra:
        logger.info("agentic %s: %s", rollout_id, {k: round(v, 3) for k, v in extra.items()})
        extra["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, extra, step_key="rollout/step")
    return False  # also emit slime's default rollout/* and perf/* metrics
