"""Lease-time branch-point policies over all-turns manifests.

Winner-mode capture commits to ONE branch point in-episode; an all-turns
manifest carries every post-tool boundary (``turns``) plus the boundary-
attributed score trace, so the branch point becomes a policy decision made at
group-build time with the whole trajectory visible. The chosen turn is stamped
into the branch samples as ``retro_branch_turn`` (consumed by
``RetroFrontierCsEnv``); returning ``None`` leaves the manifest's capture-time
default (``manifest.turn_index``) in force — the parity spine.

Policies (``ASYNC_RL_RETRO_LEASE_POLICY``):

- ``capture_default``: never override — bit-identical to winner-mode arms.
- ``target_fraction``: re-rank ALL recorded turns by distance to the (lease-
  time) target trajectory fraction. Same rule the in-episode selector applies,
  but over every boundary instead of only event-qualifying ones — and
  re-targetable over existing inventory without recapturing.
- ``hindsight_gain``: branch where the realized subsequent improvement was
  largest — max(best score after t) − (best score up to t). Only computable
  because the full trace is known at lease time.
"""

from __future__ import annotations

import os
from typing import Any

from .manifest import RetroSnapshotManifest
from .turns import TurnRecord

POLICIES = ("capture_default", "target_fraction", "hindsight_gain")


def choose_branch_turn(manifest: RetroSnapshotManifest, policy: str | None = None) -> int | None:
    """The turn to branch at, or None to keep the manifest's capture default."""

    if policy is None:
        policy = os.environ.get("ASYNC_RL_RETRO_LEASE_POLICY", "capture_default").strip().lower()
        policy = policy or "capture_default"
    if policy not in POLICIES:
        raise ValueError(f"retro lease policy must be one of {POLICIES}, got {policy!r}")
    if policy == "capture_default" or not manifest.artifact or not manifest.turns:
        return None  # winner manifests carry a single restorable state
    total_turns = manifest.source_total_turns
    if not total_turns:
        return None
    records = [_record(value) for value in manifest.turns]
    eligible = [record for record in records if total_turns - record.turn_index >= 1]
    if not eligible:
        return None

    if policy == "target_fraction":
        target = _env_float(
            "ASYNC_RL_RETRO_TARGET_TRAJECTORY_FRACTION",
            manifest.target_fraction if manifest.target_fraction is not None else 0.5,
        )
        best = min(eligible, key=lambda r: (abs(r.turn_index / total_turns - target), r.turn_index))
        return _maybe_override(best.turn_index, manifest)

    # hindsight_gain: branch at a demonstrated-progress state whose future beat
    # it the most. Turns before the first scored submission are ineligible —
    # with an empty past the "gain" is the whole trajectory and the policy
    # degenerates to replay-from-start.
    trace = [(int(e["turn_index"]), float(e["score"])) for e in manifest.score_trace]
    if not trace:
        return None
    first_scored = min(t for t, _ in trace)
    scored = [record for record in eligible if record.turn_index >= first_scored]
    if not scored:
        return None

    def gain(record: TurnRecord) -> float:
        past = max(s for t, s in trace if t <= record.turn_index)
        future = max((s for t, s in trace if t > record.turn_index), default=0.0)
        return future - past

    best = max(scored, key=lambda r: (gain(r), -r.turn_index))  # earliest max-gain turn
    if gain(best) <= 0.0:
        return None  # no realized improvement after any scored state → capture default
    return _maybe_override(best.turn_index, manifest)


def _maybe_override(turn_index: int, manifest: RetroSnapshotManifest) -> int | None:
    return None if turn_index == manifest.turn_index else int(turn_index)


def _record(value: Any) -> TurnRecord:
    return TurnRecord.from_dict(value) if isinstance(value, dict) else value


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default
