"""Serializable association between a Frontier-CS turn and a Modal snapshot.

The sandbox filesystem is only half of a resumable agent state.  A manifest also
stores the head-process checkpoint (mini-swe messages + exact RecordingModel
token prefix), policy provenance, remaining budget, and compatibility
fingerprints.  The schema is deliberately pure stdlib so rollout dumps and CPU
tests can load it without Modal, Ray, or Torch.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1

_SECRET_KEYS = {
    "agent_id",
    "api_key",
    "authorization",
    "frontier_cs_judge_url",
    "judge_url",
    "token",
}


class SnapshotKind(str, Enum):
    DIRECTORY = "directory"


class SnapshotStatus(str, Enum):
    TENTATIVE = "tentative"
    AVAILABLE = "available"
    LEASED = "leased"
    CONSUMED = "consumed"
    DELETED = "deleted"
    INVALID = "invalid"


@dataclass(frozen=True)
class Compatibility:
    """Inputs whose mismatch makes a snapshot unsafe to replay."""

    base_image: str = ""
    task_data: str = ""
    code: str = ""
    modal_version: str = ""
    mini_swe_version: str = ""

    def assert_matches(self, other: Compatibility) -> None:
        mismatched = [
            name
            for name in ("base_image", "task_data", "code", "modal_version", "mini_swe_version")
            if getattr(self, name) and getattr(other, name) and getattr(self, name) != getattr(other, name)
        ]
        if mismatched:
            raise ValueError(f"retro snapshot compatibility mismatch: {', '.join(mismatched)}")


@dataclass
class RetroSnapshotManifest:
    snapshot_id: str
    snapshot_kind: SnapshotKind
    snapshot_path: str
    created_at: float
    expires_at: float | None

    task_type: str
    instance_id: str
    problem_id: str
    event_type: str
    turn_index: int
    remaining_steps: int
    remaining_seconds: int

    source_run_tag: str = ""
    source_rollout_id: int | None = None
    source_group_index: int | None = None
    source_sample_index: int | None = None
    source_weight_version: str = ""
    source_update: int | None = None
    source_total_turns: int | None = None
    source_total_seconds: float | None = None

    target_fraction: float | None = None
    trajectory_fraction: float | None = None
    fraction_error: float | None = None

    score: float | None = None
    best_score: float | None = None
    agent_state: dict[str, Any] = field(default_factory=dict)
    sample_metadata: dict[str, Any] = field(default_factory=dict)
    compatibility: Compatibility = field(default_factory=Compatibility)

    # All-turns capture (RETRO_CAPTURE_MODE=all_turns) — additive, default-empty
    # so winner-mode rows are unchanged. ``turns`` holds TurnRecord dicts (every
    # post-tool boundary of the source trajectory), ``score_trace`` the
    # boundary-attributed submission scores, and ``artifact`` describes the
    # packed staging tarball inside the snapshot ({"member", "compression",
    # "member_root"}); empty artifact = plain directory snapshot (winner mode).
    # ``turn_index``/``remaining_*`` stay the DEFAULT branch point (parity with
    # winner selection); a lease-time policy may override via retro_branch_turn.
    turns: list = field(default_factory=list)
    score_trace: list = field(default_factory=list)
    artifact: dict[str, Any] = field(default_factory=dict)

    status: SnapshotStatus = SnapshotStatus.AVAILABLE
    leased_by: str = ""
    consumed_by_rollout: int | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported retro manifest schema {self.schema_version}")
        if not self.snapshot_id:
            raise ValueError("retro manifest missing snapshot_id")
        if self.turn_index < 0:
            raise ValueError("retro manifest turn_index must be non-negative")
        if self.remaining_steps < 1 or self.remaining_seconds < 1:
            raise ValueError("retro manifest requires positive remaining budget")
        for name in ("score", "best_score"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"retro manifest {name} outside [0, 1]: {value}")
        for name in ("target_fraction", "trajectory_fraction", "fraction_error"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"retro manifest {name} outside [0, 1]: {value}")
        self.agent_state = _sanitize(self.agent_state)
        self.sample_metadata = _sanitize(self.sample_metadata)

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: str,
        snapshot_kind: SnapshotKind,
        snapshot_path: str,
        ttl_seconds: int | None,
        **kwargs: Any,
    ) -> RetroSnapshotManifest:
        now = time.time()
        return cls(
            snapshot_id=snapshot_id,
            snapshot_kind=snapshot_kind,
            snapshot_path=snapshot_path,
            created_at=now,
            expires_at=None if ttl_seconds is None else now + ttl_seconds,
            **kwargs,
        )

    def is_expired(self, now: float | None = None) -> bool:
        return self.expires_at is not None and (time.time() if now is None else now) >= self.expires_at

    def policy_age(self, current_update: int | None) -> int | None:
        if self.source_update is None or current_update is None:
            return None
        return max(0, int(current_update) - int(self.source_update))

    def is_eligible(
        self,
        *,
        current_update: int | None = None,
        min_policy_age: int | None = None,
        max_policy_age: int | None = None,
    ) -> bool:
        if self.status != SnapshotStatus.AVAILABLE or self.is_expired():
            return False
        age = self.policy_age(current_update)
        if age is not None and min_policy_age is not None and age < min_policy_age:
            return False
        return not (age is not None and max_policy_age is not None and age > max_policy_age)

    def activate(self) -> None:
        if self.status != SnapshotStatus.TENTATIVE:
            raise ValueError(f"cannot activate retro snapshot in status {self.status.value}")
        self.status = SnapshotStatus.AVAILABLE

    def invalidate(self) -> None:
        if self.status not in (SnapshotStatus.TENTATIVE, SnapshotStatus.AVAILABLE):
            raise ValueError(f"cannot invalidate retro snapshot in status {self.status.value}")
        self.status = SnapshotStatus.INVALID
        self.leased_by = ""

    def lease(self, consumer_id: str) -> None:
        if not self.is_eligible():
            raise ValueError(f"retro snapshot {self.snapshot_id} is not available")
        if not consumer_id:
            raise ValueError("retro snapshot lease requires consumer_id")
        self.status = SnapshotStatus.LEASED
        self.leased_by = consumer_id

    def release(self) -> None:
        if self.status != SnapshotStatus.LEASED:
            raise ValueError(f"cannot release retro snapshot in status {self.status.value}")
        self.status = SnapshotStatus.AVAILABLE
        self.leased_by = ""

    def consume(self, rollout_id: int | None = None) -> None:
        if self.status not in (SnapshotStatus.AVAILABLE, SnapshotStatus.LEASED):
            raise ValueError(f"cannot consume retro snapshot in status {self.status.value}")
        self.status = SnapshotStatus.CONSUMED
        self.consumed_by_rollout = rollout_id
        self.leased_by = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["snapshot_kind"] = self.snapshot_kind.value
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RetroSnapshotManifest:
        data = copy.deepcopy(value)
        data["snapshot_kind"] = SnapshotKind(data["snapshot_kind"])
        data["status"] = SnapshotStatus(data.get("status", SnapshotStatus.AVAILABLE.value))
        data["compatibility"] = Compatibility(**data.get("compatibility", {}))
        return cls(**data)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_KEYS
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    return value
