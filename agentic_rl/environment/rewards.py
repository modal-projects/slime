"""Central reward shaping: raw verifier signal -> scalar training reward.

Every env's verifier reports a raw signal (``RewardSignal``); the scalar reward
used for training is a separate, swappable decision made here. Pick the shape
with ``ASYNC_RL_REWARD_SHAPE`` (global) or per-row via ``metadata.reward_shape``;
default ``fractional``. This is the one place to design/A-B reward shapes without
touching envs or re-converting data.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("agentic_rl")

DEFAULT_SHAPE = "fractional"
SHAPE_ENV = "ASYNC_RL_REWARD_SHAPE"
_THRESHOLD_FLOOR = float(os.environ.get("ASYNC_RL_REWARD_THRESHOLD", "0.1"))


@dataclass(frozen=True)
class RewardSignal:
    """What a verifier measured, before any shaping. ``score_raw`` is the judge's
    native 0..100; ``is_solved`` means full marks; ``cases_*`` are diagnostics."""

    score_raw: float
    is_solved: bool
    cases_passed: int = 0
    cases_total: int = 0

    @property
    def fraction(self) -> float:
        return max(0.0, min(1.0, self.score_raw / 100.0))


def _fractional(s: RewardSignal) -> float:
    return s.fraction


def _binary(s: RewardSignal) -> float:
    return 1.0 if s.is_solved else 0.0


def _thresholded(s: RewardSignal) -> float:
    f = s.fraction
    if f >= 1.0:
        return 1.0
    if f < _THRESHOLD_FLOOR:
        return 0.0
    return (f - _THRESHOLD_FLOOR) / (1.0 - _THRESHOLD_FLOOR)


SHAPERS: dict[str, Callable[[RewardSignal], float]] = {
    "fractional": _fractional,
    "binary": _binary,
    "thresholded": _thresholded,
}


def resolve_shape(metadata: dict[str, Any] | None = None) -> str:
    chosen = (metadata or {}).get("reward_shape") or os.environ.get(SHAPE_ENV) or DEFAULT_SHAPE
    if chosen not in SHAPERS:
        logger.warning("[rewards] unknown shape %r; falling back to %r", chosen, DEFAULT_SHAPE)
        return DEFAULT_SHAPE
    return chosen


def shape(signal: RewardSignal, strategy: str = DEFAULT_SHAPE) -> float:
    return SHAPERS.get(strategy, _fractional)(signal)


def signal_from_reward_dict(rewards: dict[str, Any] | None) -> RewardSignal:
    """Build a ``RewardSignal`` from a verifier's parsed ``reward.json``. Rich
    verifiers emit ``score_raw`` + ``cases_*`` + ``is_solved``; legacy harbor
    verifiers write only ``{"reward": x}`` (0..1)."""
    r = rewards or {}
    reward01 = _coerce_scalar(r)
    score_raw = float(r["score_raw"]) if r.get("score_raw") is not None else reward01 * 100.0
    cases_total = int(r.get("cases_total") or 0)
    cases_passed = int(r.get("cases_passed") or 0)
    if "is_solved" in r:
        is_solved = bool(r["is_solved"])
    elif cases_total:
        is_solved = cases_passed >= cases_total
    else:
        is_solved = reward01 >= 1.0 - 1e-6
    return RewardSignal(score_raw=score_raw, is_solved=is_solved, cases_passed=cases_passed, cases_total=cases_total)


def _coerce_scalar(rewards: dict[str, Any]) -> float:
    if not rewards:
        return 0.0
    if "reward" in rewards:
        return float(rewards["reward"])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    return 0.0


# ---------------------------------------------------------------------------
# Episode-level OUTCOME reward — the universal switch-and-test interface.
#
# Orthogonal to the per-step SHAPERS above: a SHAPER maps one verifier grade to
# a scalar; an OUTCOME strategy decides which episode-level scalar to train on,
# using the whole episode's signal (final grade + the mid-episode judge
# submissions FrontierCsEnv collects into artifacts). Applied once per episode
# in ``HarborEnv._episode`` after aggregation + artifact collection.
#
# Two orthogonal knobs, both env-switchable so strategies can be A/B'd without
# code changes (per-row override via ``metadata.outcome_reward``):
#
#   ASYNC_RL_OUTCOME_REWARD   base strategy — "final" (default; bit-identical to
#                             the pre-interface behavior), "best", "disc_sum"
#   ASYNC_RL_SOLVED_BONUS     additive bonus on solved episodes (Kevin,
#                             arXiv:2507.11948, uses 0.3); default 0 = off
#   ASYNC_RL_OUTCOME_GAMMA    disc_sum's discount (Kevin's best config: sum
#                             aggregation with gamma=0.4); default 0.4
#
# Envs without a submissions log (e.g. SWE) and episodes with zero submissions
# degrade to the final grade under every strategy.

DEFAULT_OUTCOME = "final"
OUTCOME_ENV = "ASYNC_RL_OUTCOME_REWARD"
SOLVED_BONUS_ENV = "ASYNC_RL_SOLVED_BONUS"
OUTCOME_GAMMA_ENV = "ASYNC_RL_OUTCOME_GAMMA"


@dataclass(frozen=True)
class EpisodeOutcome:
    """Everything an outcome strategy may consume. ``final`` is the aggregated
    per-step shaped reward (0..1). ``submission_scores`` are the mid-episode
    judge scores (0..1, episode order; possibly truncated to the most recent 50
    by the submissions parser). ``summary_best`` spans ALL attempts (computed
    before truncation), so "best" stays exact even for submit-spam episodes.
    ``n_invalid_scores`` counts submission scores discarded at ingestion for
    being outside [0, 1] (performance-scored problems report unbounded units)."""

    final: float
    is_solved: bool
    submission_scores: tuple[float, ...] = ()
    summary_best: float | None = None
    submitted_solved: bool = False
    n_invalid_scores: int = 0

    @property
    def best_submitted(self) -> float | None:
        candidates = [s for s in (self.summary_best, *(self.submission_scores or ())) if s is not None]
        return max(candidates) if candidates else None


def _outcome_final(o: EpisodeOutcome) -> float:
    return o.final


def _outcome_best(o: EpisodeOutcome) -> float:
    """max(final, best submitted) — plan A. Also what per-turn potential deltas
    telescope to, so this doubles as the outcome-only collapse of plan B."""
    best = o.best_submitted
    return o.final if best is None else max(o.final, best)


def _outcome_disc_sum(o: EpisodeOutcome) -> float:
    """Kevin's turn-1 return R_1 = sum_i gamma^i * r_i over the submission
    sequence — the outcome-only collapse of plan C. NOTE: with gamma < 1 this
    weights EARLY submissions most (turn 1 gets discounted credit for the
    future); the proper per-turn C assigns each turn its own R_t."""
    if not o.submission_scores:
        return o.final
    gamma = float(os.environ.get(OUTCOME_GAMMA_ENV, "0.4"))
    return float(sum(score * gamma**i for i, score in enumerate(o.submission_scores)))


# strategy -> (reward fn, solved fn). Strategies that credit mid-episode
# submissions also count a solved submission toward the bonus; "final" keeps
# the verifier-only semantics.
OUTCOMES: dict[str, tuple[Callable[[EpisodeOutcome], float], Callable[[EpisodeOutcome], bool]]] = {
    "final": (_outcome_final, lambda o: o.is_solved),
    "best": (_outcome_best, lambda o: o.is_solved or o.submitted_solved),
    "disc_sum": (_outcome_disc_sum, lambda o: o.is_solved or o.submitted_solved),
}


def resolve_outcome(metadata: dict[str, Any] | None = None) -> str:
    chosen = (metadata or {}).get("outcome_reward") or os.environ.get(OUTCOME_ENV) or DEFAULT_OUTCOME
    if chosen not in OUTCOMES:
        logger.warning("[rewards] unknown outcome %r; falling back to %r", chosen, DEFAULT_OUTCOME)
        return DEFAULT_OUTCOME
    return chosen


def shape_outcome(outcome: EpisodeOutcome, strategy: str = DEFAULT_OUTCOME) -> tuple[float, dict[str, Any]]:
    """Episode training scalar + a breakdown dict (shipped into
    ``sample.metadata.agentic.outcome`` for the W&B outcome metrics). Defaults
    (strategy "final", bonus 0) reproduce the pre-interface reward exactly."""
    reward_fn, solved_fn = OUTCOMES.get(strategy, OUTCOMES[DEFAULT_OUTCOME])
    bonus = float(os.environ.get(SOLVED_BONUS_ENV, "0") or 0)
    base = reward_fn(outcome)
    applied_bonus = bonus if (bonus and solved_fn(outcome)) else 0.0
    return base + applied_bonus, {
        "strategy": strategy,
        "final": outcome.final,
        "best_submitted": outcome.best_submitted,
        "base": base,
        "bonus": applied_bonus,
        "invalid_scores": outcome.n_invalid_scores,
    }


_SCORE_EPS = 1e-6


def _valid01(value: Any) -> float | None:
    """A submission score usable as reward, or None. Valid scores live in
    [0, 1] (tiny float overshoot clamped); anything else is a different unit."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(v, 1.0) if 0.0 <= v <= 1.0 + _SCORE_EPS else None


def episode_outcome_from_artifacts(final: float, is_solved: bool, artifacts: dict[str, Any] | None) -> EpisodeOutcome:
    """Build the :class:`EpisodeOutcome` off a harbor episode's collected
    artifacts (``submissions``/``submission_summary`` from FrontierCsEnv;
    absent for envs that don't log submissions).

    Submission scores are only trusted inside [0, 1]: performance-scored
    problems report ``score`` in unbounded judge units (observed: 92.0, 835.76
    where the verifier's clamped final was ~0), and feeding those to "best"
    poisons the training scalar. Out-of-range scores are discarded (counted in
    ``n_invalid_scores``) and, since the episode's units are then untrusted,
    ``submitted_solved`` is recomputed from the surviving valid scores instead
    of the summary's raw ``score_raw >= 100`` flag."""
    subs = (artifacts or {}).get("submissions") or []
    scores: list[float] = []
    n_invalid = 0
    for entry in subs:
        try:
            raw = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue  # unscored/malformed attempt, same as before
        score = _valid01(raw)
        if score is None:
            n_invalid += 1
        else:
            scores.append(score)
    summary = (artifacts or {}).get("submission_summary") or {}
    summary_best_raw = summary.get("best_score")
    summary_best = _valid01(summary_best_raw)
    if summary_best_raw is not None and summary_best is None:
        n_invalid += 1
    submitted_solved = bool(summary.get("solved"))
    if n_invalid:
        submitted_solved = any(s >= 1.0 - _SCORE_EPS for s in scores)
    return EpisodeOutcome(
        final=final,
        is_solved=is_solved,
        submission_scores=tuple(scores),
        summary_best=summary_best,
        submitted_solved=submitted_solved,
        n_invalid_scores=n_invalid,
    )
