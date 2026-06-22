"""Custom rollout metrics for fully-async agentic RL, logged via
``--custom-rollout-log-function-path agentic_rl.metrics.log_rollout_data`` on top
of slime's defaults.

  agentic/*  episode behavior — turns, exit-status mix, phase timing, token split,
             solve rate, chains/steps.
  async/*    off-policy health of each training batch — version_span (weight
             versions a multi-turn episode straddled), version_lag (updates-stale
             vs the freshest in the batch), versions_in_batch, sample_age_sec
             (gen-finish -> train dwell).
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter

import numpy as np

logger = logging.getLogger("agentic_rl")

_MEAN_KEYS = ("turns", "format_errors", "gen_time", "output_tokens", "response_tokens", "chains")
_MAX_KEYS = ("turns", "response_tokens", "output_tokens")
_TIMING_KEYS = ("boot", "prep", "agent", "verifier", "grade", "episode")


def _vnum(v):
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def _stats(samples) -> list[dict]:
    return [
        s.metadata["agentic"]
        for s in samples
        if isinstance(getattr(s, "metadata", None), dict) and "agentic" in s.metadata
    ]


def _agentic_metrics(samples) -> dict:
    stats = _stats(samples)
    if not stats:
        return {}

    def mean(getter):
        vals = [getter(s) for s in stats]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    out = {f"agentic/{k}/mean": mean(lambda s, k=k: s.get(k)) for k in _MEAN_KEYS}
    for k in _MAX_KEYS:
        out[f"agentic/{k}/max"] = float(max((s.get(k, 0) for s in stats), default=0))
    for k in _TIMING_KEYS:
        out[f"agentic/timing/{k}/mean"] = mean(lambda s, k=k: (s.get("timing") or {}).get(k))

    gens = [(s["output_tokens"], s["gen_time"]) for s in stats if s.get("gen_time", 0) > 0 and "output_tokens" in s]
    if gens:
        out["agentic/decode_tok_per_s/mean"] = float(np.mean([o / g for o, g in gens]))

    out["agentic/solved_frac"] = float(np.mean([1.0 if s.get("is_solved") else 0.0 for s in stats]))
    exits = Counter(s.get("exit_status", "?") for s in stats)
    total = sum(exits.values()) or 1
    for status, c in exits.items():
        out[f"agentic/exit_frac/{status}"] = c / total
    return out


def _async_metrics(samples, now: float) -> dict:
    out: dict = {}
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
        logger.info("[agentic_rl] %s: %s", rollout_id, {k: round(v, 3) for k, v in extra.items()})
        extra["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, extra, step_key="rollout/step")
    return False  # also emit slime's default rollout/* and perf/* metrics
