"""Turn-level reward shaping — ROLLOUT side.

Wire-up (O3 config):

    custom_reward_post_process_path = "agentic_rl.rewards.turn_reward.post_process_rewards"
    custom_advantage_function_path  = "agentic_rl.rewards.turn_advantage.compute_turn_advantages"
    environment["ASYNC_RL_TURN_REWARD"] = "next_sub"

The per-token training signal becomes

    A_token = A_outcome(sample) + omega * A_turn(turn(token))

where ``A_outcome`` is the stock GRPO group-normalized episode scalar (unchanged
from O0/O1/O2) and ``A_turn`` is a group-normalized per-turn value painted over
that turn's token span train-side. ``omega`` (``ASYNC_RL_TURN_MIX_WEIGHT``,
default 0.3) is applied train-side so the two components stay separately
inspectable in the shipped metadata.

Turn-reward rule ``next_sub`` (the only one for now): a turn's raw reward is the
score of the episode's NEXT judge submission at-or-after that turn's generation
end — i.e. every turn is credited with the score of the submission its work
culminated in (a turn that itself submits is credited with that submission).
Turns after the last submission get the LAST submission's score. Episodes with
no scored submission ship no turn component (pure outcome — degrades to O1
behavior); so do masked null samples.

Alignment is by wall clock: ``model.py`` records each turn's generation-end
timestamp (worker clock) and the judge records each submission's arrival
(``ts`` = server ``Date.now()`` ms). Both are NTP-synced Modal containers and
inter-turn gaps are tens of seconds, so second-level skew is harmless.

The hook replicates slime's stock GRPO reward normalization bit-for-bit (it
REPLACES ``_post_process_rewards``), then group-normalizes the pooled turn
rewards and ships per-sample ``train_metadata``:

    {"turn_spans": [[s, e], ...],   # response-relative token spans, one per turn
     "turn_rewards": [...],         # raw r_t in [0, 1] (diagnostics)
     "turn_adv": [...]}             # group-normalized r_t (what the train side paints)

Group pooling uses ``sample.group_index`` runs (falls back to contiguous blocks
of ``n_samples_per_prompt``); the normalization stats are over every turn of
every sample in the group, so a turn is "good" relative to the group's other
turns, mirroring how GRPO scores an episode relative to its group.
"""

from __future__ import annotations

import logging
import os
from bisect import bisect_left
from datetime import datetime
from typing import Any

import torch

from .rewards import _valid01

logger = logging.getLogger("agentic_rl")

TURN_REWARD_ENV = "ASYNC_RL_TURN_REWARD"
TURN_MIX_ENV = "ASYNC_RL_TURN_MIX_WEIGHT"  # omega; read TRAIN-side by turn_advantage.py
DEFAULT_TURN_STRATEGY = ""  # off

_STD_EPS = 1e-6


def resolve_turn_strategy() -> str:
    return (os.environ.get(TURN_REWARD_ENV) or DEFAULT_TURN_STRATEGY).strip()


def _ts_seconds(value: Any) -> float | None:
    """Coerce a submission timestamp to epoch seconds: judge rows carry
    ``Date.now()`` ms; the sandbox-log fallback carries ISO-8601 strings."""
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e11 else v
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _scored_submissions(submissions: list[dict] | None) -> list[tuple[float | None, float]]:
    """(ts_seconds | None, score01) per scored submission, in episode order.
    Scores outside [0, 1] are discarded (same _valid01 rule the outcome uses)."""
    out: list[tuple[float | None, float]] = []
    for e in submissions or ():
        if not isinstance(e, dict):
            continue
        score = _valid01(e.get("score"))
        if score is None:
            continue
        ts = _ts_seconds(e.get("ts")) or _ts_seconds(e.get("ts_started")) or _ts_seconds(e.get("ts_done"))
        out.append((ts, score))
    return out


def turn_rewards_next_sub(turn_ts: list[float], submissions: list[dict] | None) -> list[float] | None:
    """The ``next_sub`` rule: r_t = score of the first scored submission at or
    after turn t's generation end; turns past the last submission get the last
    scored submission's score. None when the episode has no scored submission
    (no turn signal). Submissions without a usable timestamp can't be aligned
    and only participate via the last-submission fallback."""
    scored = _scored_submissions(submissions)
    if not scored:
        return None
    timed = sorted((ts, sc) for ts, sc in scored if ts is not None)
    ts_keys = [ts for ts, _ in timed]
    last_score = scored[-1][1]
    rewards = []
    for t in turn_ts:
        i = bisect_left(ts_keys, t)
        rewards.append(timed[i][1] if i < len(timed) else last_score)
    return rewards


TURN_RULES = {"next_sub": turn_rewards_next_sub}


def compute_turn_rewards(agentic: dict[str, Any] | None, strategy: str) -> tuple[list[list[int]], list[float]] | None:
    """(turn_spans, raw turn rewards) for one sample, or None when the sample
    ships no turn signal (null sample, no turns recorded, or no scored subs)."""
    rule = TURN_RULES.get(strategy)
    if rule is None or not isinstance(agentic, dict):
        return None
    spans = agentic.get("turn_spans")
    turn_ts = agentic.get("turn_ts")
    if not spans or not turn_ts or len(spans) != len(turn_ts):
        return None
    rewards = rule(list(turn_ts), agentic.get("submissions"))
    if rewards is None:
        return None
    return [list(s) for s in spans], rewards


def _default_grpo_post_process(args, samples) -> tuple[list, list]:
    """Bit-for-bit copy of slime's stock ``_post_process_rewards`` (the custom
    hook REPLACES it, so the outcome normalization must be reproduced here)."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean
        if args.advantage_estimator in ["grpo", "gspo", "cispo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)
        return raw_rewards, rewards.flatten().tolist()
    return raw_rewards, raw_rewards


def _group_slices(samples, n_per_group: int) -> list[range]:
    """Contiguous index runs of ``sample.group_index`` (samples arrive
    group-contiguous); falls back to fixed blocks when indices are absent."""
    groups: list[range] = []
    start = 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or getattr(samples[i], "group_index", None) != getattr(samples[start], "group_index", None):
            groups.append(range(start, i))
            start = i
    if all(getattr(s, "group_index", None) is None for s in samples) and n_per_group > 1:
        groups = [range(i, min(i + n_per_group, len(samples))) for i in range(0, len(samples), n_per_group)]
    return groups


def _normalize_group(turn_rewards: dict[int, list[float]], use_std: bool) -> dict[int, list[float]]:
    """Pool every turn reward in the group, center (and std-scale) them; a
    degenerate pool (fewer than 2 turns or zero variance) carries no signal."""
    pooled = [v for vals in turn_rewards.values() for v in vals]
    if len(pooled) < 2:
        return {i: [0.0] * len(vals) for i, vals in turn_rewards.items()}
    t = torch.tensor(pooled, dtype=torch.float)
    mean, std = t.mean().item(), t.std().item()
    if std < _STD_EPS:
        return {i: [0.0] * len(vals) for i, vals in turn_rewards.items()}
    scale = (std + _STD_EPS) if use_std else 1.0
    return {i: [(v - mean) / scale for v in vals] for i, vals in turn_rewards.items()}


def post_process_rewards(args, samples):
    """slime ``--custom-reward-post-process-path`` hook: stock GRPO outcome
    normalization + the turn-level component shipped via ``train_metadata``."""
    raw_rewards, rewards = _default_grpo_post_process(args, samples)

    strategy = resolve_turn_strategy()
    if not strategy:
        return raw_rewards, rewards
    if strategy not in TURN_RULES:
        logger.warning("[turn_reward] unknown %s=%r; turn shaping off", TURN_REWARD_ENV, strategy)
        return raw_rewards, rewards

    per_sample = [
        compute_turn_rewards((getattr(s, "metadata", None) or {}).get("agentic"), strategy) for s in samples
    ]
    use_std = bool(getattr(args, "grpo_std_normalization", True))
    for grp in _group_slices(samples, getattr(args, "n_samples_per_prompt", 1) or 1):
        group_rewards = {i: per_sample[i][1] for i in grp if per_sample[i] is not None}
        normalized = _normalize_group(group_rewards, use_std)
        for i in grp:
            md = dict(samples[i].train_metadata or {})
            if per_sample[i] is not None:
                spans, raw_turn = per_sample[i]
                md |= {"turn_spans": spans, "turn_rewards": raw_turn, "turn_adv": normalized[i]}
            # train_data["metadata"] is only emitted when EVERY sample carries
            # train_metadata (gated on samples[0]), so always set at least {}.
            samples[i].train_metadata = md
    return raw_rewards, rewards
