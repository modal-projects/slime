"""Central reward shaping: raw verifier signal -> scalar training reward.

Every env's verifier reports a raw signal (``RewardSignal``); the scalar reward
used for training is a separate, swappable decision made here. Pick the shape
with ``AGENTIC_REWARD_SHAPE`` (global) or per-row via ``metadata.reward_shape``;
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
SHAPE_ENV = "AGENTIC_REWARD_SHAPE"
_THRESHOLD_FLOOR = float(os.environ.get("AGENTIC_REWARD_THRESHOLD", "0.1"))


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
