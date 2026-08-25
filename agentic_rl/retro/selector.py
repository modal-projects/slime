"""Deterministic trajectory-relative branch-point selection for Frontier-CS.

The selector watches the sandbox-local submission log to classify coherent
post-tool states.  A caller stages every matching state, then retains the one
closest to a configured fraction of the *realized* source trajectory.  The log
is never trusted for training reward; final and branch rewards remain
server-side.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum

from agentic_rl.environment.submissions import parse_submissions_log

_EPS = 1e-6


class BranchEventType(str, Enum):
    PROMISING = "promising"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class SelectionConfig:
    min_score: float = 0.1
    max_score: float = 0.95
    min_remaining_fraction: float = 0.0
    min_turn: int = 2
    regression_delta: float = 0.1
    stagnant_submissions: int = 2
    # Number of *additional* consecutive qualifying new-best submissions to
    # require after the first. Zero preserves the original immediate-new-best
    # behavior; one requires two consecutive qualifying improvements.
    promising_consecutive: int = 0
    target_fraction: float = 0.5
    max_fraction_error: float = 0.4
    preferred_event: BranchEventType | None = None
    allow_fallback: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_fraction <= 1.0:
            raise ValueError("retro target_fraction must be in [0, 1]")
        if not 0.0 <= self.max_fraction_error <= 1.0:
            raise ValueError("retro max_fraction_error must be in [0, 1]")
        if self.stagnant_submissions < 1:
            raise ValueError("retro stagnant_submissions must be at least 1")
        if self.promising_consecutive < 0:
            raise ValueError("retro promising_consecutive must be non-negative")


@dataclass(frozen=True)
class BranchEvent:
    event_type: BranchEventType
    turn_index: int
    score: float
    best_score: float
    remaining_steps: int
    remaining_seconds: int
    submission_count: int
    reason: str
    elapsed_seconds: float = 0.0
    source_total_turns: int | None = None
    source_total_seconds: float | None = None
    trajectory_fraction: float | None = None
    fraction_error: float | None = None


class EventSelector:
    """Classify matching post-submission states, then select one at episode end."""

    def __init__(self, config: SelectionConfig | None = None):
        self.config = config or SelectionConfig()
        self._seen_scored = 0
        self._previous_score: float | None = None
        self._best_score: float | None = None
        self._stagnant = 0
        self._promising_streak = 0

    def observe_log(
        self,
        text: str,
        *,
        turn_index: int,
        max_steps: int,
        elapsed_seconds: float,
        wall_time_seconds: int,
    ) -> BranchEvent | None:
        scores = valid_score_trace(text)
        if len(scores) <= self._seen_scored:
            return None

        # A single tool action can append more than one completed record. Consume
        # each in order so regression/stagnation state remains deterministic. The
        # filesystem exists only after the whole tool action, so only the final new
        # submission may label this coherent post-tool state.
        event: BranchEvent | None = None
        for score in scores[self._seen_scored :]:
            event = self._observe_score(
                score,
                turn_index=turn_index,
                max_steps=max_steps,
                elapsed_seconds=elapsed_seconds,
                wall_time_seconds=wall_time_seconds,
                submission_count=self._seen_scored + 1,
            )
            self._seen_scored += 1
        return event

    def select(self, candidates: list[BranchEvent], *, total_turns: int, total_seconds: float) -> BranchEvent | None:
        """Retain the matching candidate nearest the configured trajectory fraction."""

        if total_turns < 1 or total_seconds <= 0:
            return None
        ranked: list[tuple[float, int, BranchEvent]] = []
        for candidate in candidates:
            remaining_steps = total_turns - candidate.turn_index
            remaining_seconds = int(total_seconds - candidate.elapsed_seconds)
            if remaining_steps < 1 or remaining_seconds < 1:
                continue
            fraction = candidate.turn_index / total_turns
            error = abs(fraction - self.config.target_fraction)
            ranked.append((error, candidate.turn_index, candidate))
        if not ranked:
            return None
        error, _, candidate = min(ranked, key=lambda item: (item[0], item[1]))
        if error > self.config.max_fraction_error + _EPS:
            return None
        return replace(
            candidate,
            remaining_steps=total_turns - candidate.turn_index,
            remaining_seconds=max(1, int(total_seconds - candidate.elapsed_seconds)),
            source_total_turns=total_turns,
            source_total_seconds=total_seconds,
            trajectory_fraction=candidate.turn_index / total_turns,
            fraction_error=error,
        )

    def _observe_score(
        self,
        score: float,
        *,
        turn_index: int,
        max_steps: int,
        elapsed_seconds: float,
        wall_time_seconds: int,
        submission_count: int,
    ) -> BranchEvent | None:
        best_before = self._best_score
        improved = best_before is None or score > best_before + _EPS
        if improved:
            self._best_score = score
            self._stagnant = 0
        else:
            self._stagnant += 1
        qualifying_improvement = improved and self.config.min_score <= score <= self.config.max_score
        if qualifying_improvement:
            self._promising_streak += 1
        else:
            self._promising_streak = 0
        self._previous_score = score

        remaining_steps = max(0, max_steps - turn_index)
        remaining_seconds = max(0, int(wall_time_seconds - elapsed_seconds))
        step_fraction = remaining_steps / max(1, max_steps)
        time_fraction = remaining_seconds / max(1, wall_time_seconds)
        if (
            turn_index < self.config.min_turn
            or min(step_fraction, time_fraction) < self.config.min_remaining_fraction
            or score >= 1.0 - _EPS
        ):
            return None

        regressed = best_before is not None and best_before - score >= self.config.regression_delta
        stalled = self._stagnant >= self.config.stagnant_submissions
        if regressed or stalled:
            candidate = BranchEvent(
                event_type=BranchEventType.RECOVERY,
                turn_index=turn_index,
                score=score,
                best_score=max(score, self._best_score or score),
                remaining_steps=remaining_steps,
                remaining_seconds=remaining_seconds,
                submission_count=submission_count,
                reason="score_regression" if regressed else "score_plateau",
                elapsed_seconds=elapsed_seconds,
            )
            if self._accept(candidate.event_type):
                return candidate

        required_promising_streak = self.config.promising_consecutive + 1
        if qualifying_improvement and self._promising_streak >= required_promising_streak:
            candidate = BranchEvent(
                event_type=BranchEventType.PROMISING,
                turn_index=turn_index,
                score=score,
                best_score=max(score, self._best_score or score),
                remaining_steps=remaining_steps,
                remaining_seconds=remaining_seconds,
                submission_count=submission_count,
                reason="intermediate_progress",
                elapsed_seconds=elapsed_seconds,
            )
            if self._accept(candidate.event_type):
                return candidate
        return None

    def _accept(self, event_type: BranchEventType) -> bool:
        preferred = self.config.preferred_event
        return preferred is None or event_type == preferred or self.config.allow_fallback


def assign_event_type(
    *,
    assignment: str,
    promising_ratio: float,
    seed: int,
    group_index: int | None,
    sample_index: int | None,
) -> BranchEventType | None:
    """Assign a selector per rollout, never per problem instance."""

    mode = assignment.strip().lower()
    if not 0.0 <= promising_ratio <= 1.0:
        raise ValueError("retro promising_ratio must be in [0, 1]")
    if mode == "promising":
        return BranchEventType.PROMISING
    if mode == "recovery":
        return BranchEventType.RECOVERY
    if mode == "any":
        return None
    identity = f"{seed}:{group_index}:{sample_index}"
    if mode == "alternating":
        value = int(sample_index if sample_index is not None else _stable_u64(identity))
        return BranchEventType.PROMISING if value % 2 == 0 else BranchEventType.RECOVERY
    if mode != "hashed":
        raise ValueError(
            "retro selector assignment must be one of hashed, alternating, promising, recovery, any"
        )
    unit = _stable_u64(identity) / 2**64
    return BranchEventType.PROMISING if unit < promising_ratio else BranchEventType.RECOVERY


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def valid_score_trace(text: str) -> list[float]:
    """Sandbox-log scores that are already normalized to [0, 1]."""

    parsed = parse_submissions_log(text)
    scores: list[float] = []
    for submission in parsed["submissions"]:
        try:
            score = float(submission["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0 + _EPS:
            scores.append(min(score, 1.0))
    return scores
