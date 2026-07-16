"""Custom rollout metrics for fully-async agentic RL, logged via
``--custom-rollout-log-function-path agentic_rl.metrics.log_rollout_data`` on top
of slime's defaults.

  agentic/*  episode behavior — turns (mean/max/p90), exit-status mix, phase timing,
             token split, solve rate, chains/steps, judge submissions/episode
             (frontier_cs); plus rollout-throughput health:
             elapsed_sec tail (p50/p90/p99/max) + straggler_ratio (a sync step is
             gated by its slowest sample, which the mean hides), budget_hit_frac,
             exec_timeouts (hung bash calls cut by the client-side timeout),
             trained_token_frac, removed_frac (fully-masked, no-gradient samples).
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

_MEAN_KEYS = ("turns", "format_errors", "output_tokens", "response_tokens", "chains")
_MAX_KEYS = ("turns", "response_tokens", "output_tokens")
# "generate" is the LLM-inference time -- a sub-phase of "agent" (the rest of the agent
# leg is in-sandbox tool exec), kept next to it; "episode" is the full boot..grade wall.
_TIMING_KEYS = ("boot", "prep", "agent", "generate", "verifier", "grade", "episode")


def _vnum(v):
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def _stats(samples) -> list[dict]:
    return [
        s.metadata["agentic"]
        for s in samples
        if isinstance(getattr(s, "metadata", None), dict) and "agentic" in s.metadata
    ]


def _pctl_block(prefix: str, vals, ps=(50, 90, 99), include_max=True) -> dict:
    """Percentile (+ max) summary for a tail-sensitive quantity (latency, turns)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=float)
    out = {f"{prefix}/p{p}": float(np.percentile(arr, p)) for p in ps}
    if include_max:
        out[f"{prefix}/max"] = float(arr.max())
    return out


def _agentic_metrics(samples, args) -> dict:
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
    # Per-phase timing: mean for every phase, plus max -- a single hung verifier or
    # sandbox gates a sync step and the mean hides it (e.g. a test suite that burns
    # the full verifier timeout while the agent finished in minutes) -- plus p90 for
    # the two straggler-prone phases (agent leg + verifier/RM).
    for k in _TIMING_KEYS:
        tvals = [v for v in ((s.get("timing") or {}).get(k) for s in stats) if v is not None]
        out[f"agentic/timing/{k}/mean"] = float(np.mean(tvals)) if tvals else 0.0
        if tvals:
            out[f"agentic/timing/{k}/max"] = float(np.max(tvals))
    for k in ("agent", "verifier"):
        out |= _pctl_block(f"agentic/timing/{k}", [(s.get("timing") or {}).get(k) for s in stats], ps=(90,), include_max=False)

    gens = [
        (s["output_tokens"], (s.get("timing") or {}).get("generate"))
        for s in stats
        if "output_tokens" in s and ((s.get("timing") or {}).get("generate") or 0) > 0
    ]
    if gens:
        out["agentic/decode_tok_per_s/mean"] = float(np.mean([o / g for o, g in gens]))

    out["agentic/solved_frac"] = float(np.mean([1.0 if s.get("is_solved") else 0.0 for s in stats]))
    exits = Counter(s.get("exit_status", "?") for s in stats)
    total = sum(exits.values()) or 1
    for status, c in exits.items():
        out[f"agentic/exit_frac/{status}"] = c / total

    # Tail latency -- the mean hides a single multi-hour straggler that gates a sync step
    # (rollout_time tracks the MAX of this, not the mean). straggler_ratio = max/median.
    elapsed = [s.get("elapsed_sec") for s in stats if s.get("elapsed_sec") is not None]
    out |= _pctl_block("agentic/elapsed_sec", elapsed)
    if elapsed:
        med = float(np.median(elapsed))
        out["agentic/straggler_ratio"] = float(max(elapsed) / med) if med > 0 else 0.0
        budget = float(getattr(args, "agentic_episode_timeout", 1800) or 1800)
        out["agentic/budget_hit_frac"] = float(np.mean([1.0 if e >= 0.95 * budget else 0.0 for e in elapsed]))

    # Turn-count tail (no-progress loops / step-cap pile-up; mean+max already above).
    out |= _pctl_block("agentic/turns", [s.get("turns") for s in stats], ps=(90,), include_max=False)

    # Hung-command canary: episodes where a bash call hit the client-side exec timeout.
    et = [s.get("exec_timeouts") for s in stats if s.get("exec_timeouts") is not None]
    if et:
        out["agentic/exec_timeouts/mean"] = float(np.mean(et))
        out["agentic/exec_timeout_frac"] = float(np.mean([1.0 if x > 0 else 0.0 for x in et]))

    # Forced-think-closure canary (agentic_close_think_on_length): how often the
    # scaffold had to close a runaway <think>. A RISING trend across training steps
    # means the policy is learning to lean on the crutch instead of closing its own
    # think — the signal to add length pressure (soft overlong penalty).
    tc = [s.get("think_closures") for s in stats if s.get("think_closures") is not None]
    if tc:
        out["agentic/think_closures/mean"] = float(np.mean(tc))
        out["agentic/think_closure_frac"] = float(np.mean([1.0 if x > 0 else 0.0 for x in tc]))

    # Iterative judge submissions per episode (frontier_cs submit.sh loop). Episodes
    # where the agent never submitted carry no submission_summary and count as 0;
    # omitted entirely for task families that don't log submissions (e.g. SWE).
    # max/p90 are the submission-spam watchdog: the judge has crashed under load
    # before, so if the mean climbs >2x the outcome-baseline (~7 at rollout 19 of
    # the 20260708-023750 run) the reward shaping is likely teaching submit-spam.
    subs = [(s.get("submission_summary") or {}).get("n") for s in stats]
    if any(n is not None for n in subs):
        counts = [n or 0 for n in subs]
        out["agentic/submissions/mean"] = float(np.mean(counts))
        out["agentic/submissions/max"] = float(max(counts))
        out |= _pctl_block("agentic/submissions", counts, ps=(90,), include_max=False)

        # Server-side collection health + forgery tripwire. server_frac < 1 means
        # some episodes fell back to the agent-writable sandbox log (old judge or
        # fetch failures); log_mismatch_frac > 0 means the sandbox log disagreed
        # with the judge's own records (sids the server never saw, or edited
        # scores) — the reward is server-side either way, but a persistent
        # nonzero rate means the policy is trying to game the log.
        summaries = [s.get("submission_summary") for s in stats if s.get("submission_summary")]
        if summaries:
            out["agentic/submissions/server_frac"] = float(
                np.mean([1.0 if sm.get("source") == "server" else 0.0 for sm in summaries])
            )
            out["agentic/submissions/log_mismatch_frac"] = float(
                np.mean([
                    1.0 if ((sm.get("n_log_only") or 0) > 0 or (sm.get("n_score_mismatch") or 0) > 0) else 0.0
                    for sm in summaries
                ])
            )

    # Outcome-reward shaping observability (harbor.py -> rewards.shape_outcome
    # breakdown). uplift = how much best-submitted exceeds the final grade — the
    # signal strategy "best" reclaims; bonus_frac = episodes that got the
    # solved bonus. Under baseline defaults these still log (uplift shows the
    # headroom even when it isn't being trained on).
    outcomes = [s.get("outcome") for s in stats if isinstance(s.get("outcome"), dict)]
    if outcomes:
        finals = [float(o.get("final") or 0.0) for o in outcomes]
        uplifts = [max(0.0, float(o.get("best_submitted") or 0.0) - f) for o, f in zip(outcomes, finals)]
        out["agentic/outcome/reward_final/mean"] = float(np.mean(finals))
        out["agentic/outcome/reward_trained/mean"] = float(np.mean([float(o.get("base") or 0.0) + float(o.get("bonus") or 0.0) for o in outcomes]))
        out["agentic/outcome/uplift/mean"] = float(np.mean(uplifts))
        out["agentic/outcome/best_gt_final_frac"] = float(np.mean([1.0 if u > 1e-9 else 0.0 for u in uplifts]))
        out["agentic/outcome/bonus_frac"] = float(np.mean([1.0 if float(o.get("bonus") or 0.0) > 0 else 0.0 for o in outcomes]))
        # Canary for performance-scored problems whose submit scores arrive in
        # non-[0,1] judge units and get discarded at ingestion (rewards.py) --
        # they'd otherwise leak unbounded values into "best" (observed: 835.76
        # at rollout 13 of the o1-best run).
        out["agentic/outcome/invalid_score_frac"] = float(
            np.mean([1.0 if (o.get("invalid_scores") or 0) > 0 else 0.0 for o in outcomes])
        )

    # Per-turn shaping precursor: fraction of episodes with >=1 positive
    # best-so-far delta across their submission sequence (the r_t > 0 events
    # potential-based shaping would reward). Denominator = every episode in the
    # batch; a never-submitting episode has no delta by definition.
    if any(n is not None for n in subs):
        def has_positive_delta(entries) -> bool:
            for e in entries or ():
                score = e.get("score")
                if score is not None and float(score) > 1e-9:
                    return True
            return False

        out["agentic/submissions/positive_delta_frac"] = float(
            np.mean([1.0 if has_positive_delta(s.get("submissions")) else 0.0 for s in stats])
        )

    # Trained-token fraction: share of generated tokens that carry gradient (the rest is
    # masked tool output) -- low means we pay to generate context we never train on.
    resp = sum(s.get("response_tokens", 0) or 0 for s in stats)
    if resp > 0:
        out["agentic/trained_token_frac"] = sum(s.get("output_tokens", 0) or 0 for s in stats) / resp
    return out


def _outcome_dict(s) -> dict | None:
    """The episode's outcome-shaping breakdown (harbor.py -> shape_outcome),
    or None for samples without one (masked nulls, non-harbor envs)."""
    md = getattr(s, "metadata", None)
    outcome = md.get("agentic", {}).get("outcome") if isinstance(md, dict) else None
    return outcome if isinstance(outcome, dict) else None


def _final_reward(s) -> float:
    """The episode's pre-shaping final-solution reward; falls back to the
    shipped reward for samples without a breakdown (masked nulls carry 0, and
    without shaping reward == final anyway)."""
    outcome = _outcome_dict(s)
    if outcome is not None and outcome.get("final") is not None:
        return float(outcome["final"])
    return float(getattr(s, "reward", 0.0) or 0.0)


def _best_traj_reward(s) -> float:
    """The best reward achieved anywhere in the trajectory: max(final grade,
    best mid-episode submission score) — the quantity O1's "best" strategy
    trains on, logged for every run so O0/O2 show the reclaimable headroom.
    Episodes without submissions degrade to their final grade; samples without
    a breakdown fall back like _final_reward."""
    outcome = _outcome_dict(s)
    if outcome is None:
        return float(getattr(s, "reward", 0.0) or 0.0)
    final = float(outcome.get("final") or 0.0)
    best = outcome.get("best_submitted")
    return max(final, float(best)) if best is not None else final


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

    extra = {**_agentic_metrics(samples, args), **_async_metrics(samples, time.time())}
    # Fraction of the batch shipped fully-masked (remove_sample) -- contributes no
    # gradient, so it shrinks the effective batch. Read off the samples, not the stats.
    if samples:
        extra["agentic/removed_frac"] = float(
            np.mean([1.0 if getattr(s, "remove_sample", False) else 0.0 for s in samples])
        )
        # O0 reference: mean of the UNSHAPED final-solution reward over the same
        # denominator as rollout/raw_reward (all shipped samples; nulls count 0).
        # raw_reward - average_last_reward = what outcome shaping added.
        extra["rollout/average_last_reward"] = float(np.mean([_final_reward(s) for s in samples]))
        # O1 reference: mean best-of-trajectory reward (max of the final grade and
        # the best server-side submission score), same denominator. Under O1 this
        # tracks raw_reward (its training scalar); under O0/O2 the gap to
        # average_last_reward is the headroom "best" would reclaim.
        extra["rollout/best_reward_per_traj"] = float(np.mean([_best_traj_reward(s) for s in samples]))
    if extra:
        logger.info("[agentic_rl] %s: %s", rollout_id, {k: round(v, 3) for k, v in extra.items()})
        extra["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, extra, step_key="rollout/step")
    return False  # also emit slime's default rollout/* and perf/* metrics
