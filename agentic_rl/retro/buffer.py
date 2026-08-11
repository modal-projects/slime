"""Bounded recent-snapshot queue with append-only crash recovery."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

from .manifest import RetroSnapshotManifest, SnapshotStatus

_STORE_LOCKS: dict[str, threading.Lock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


class ManifestStore:
    """Append state transitions; last record per snapshot wins on reload."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        key = str(self.path.resolve())
        with _STORE_LOCKS_GUARD:
            self._lock = _STORE_LOCKS.setdefault(key, threading.Lock())

    def append(self, manifest: RetroSnapshotManifest) -> None:
        """Append a complete manifest exactly once at snapshot creation."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(manifest.to_dict(), separators=(",", ":"), sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def append_transition(self, manifest: RetroSnapshotManifest) -> None:
        """Append only mutable lifecycle fields, avoiding checkpoint duplication."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        transition = {
            "_transition": True,
            "snapshot_id": manifest.snapshot_id,
            "status": manifest.status.value,
            "leased_by": manifest.leased_by,
            "consumed_by_rollout": manifest.consumed_by_rollout,
        }
        line = json.dumps(transition, separators=(",", ":"), sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def load_latest(self) -> list[RetroSnapshotManifest]:
        if not self.path.is_file():
            return []
        latest: dict[str, RetroSnapshotManifest] = {}
        # Appends are one-line atomic writes. Do not hold the writer lock while
        # scanning a large historical manifest; an incomplete trailing line is
        # safely ignored and will be visible on the next reload.
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
                if not isinstance(value, dict):
                    continue
                if value.get("_transition"):
                    manifest = latest.get(str(value.get("snapshot_id") or ""))
                    if manifest is None:
                        continue
                    try:
                        manifest.status = SnapshotStatus(value["status"])
                    except (KeyError, ValueError):
                        continue
                    manifest.leased_by = str(value.get("leased_by") or "")
                    manifest.consumed_by_rollout = value.get("consumed_by_rollout")
                    continue
                try:
                    manifest = RetroSnapshotManifest.from_dict(value)
                except (ValueError, TypeError, KeyError):
                    continue
                latest[manifest.snapshot_id] = manifest
        return list(latest.values())


class RetroBuffer:
    def __init__(self, *, max_items: int = 1024, store: ManifestStore | None = None):
        if max_items < 1:
            raise ValueError("RetroBuffer max_items must be positive")
        self.max_items = max_items
        self.store = store
        self._items: deque[RetroSnapshotManifest] = deque()
        self._lock = threading.Lock()
        if store is not None:
            for manifest in sorted(store.load_latest(), key=lambda item: item.created_at):
                if manifest.status == SnapshotStatus.LEASED:
                    manifest.release()
                if manifest.is_eligible():
                    self._items.append(manifest)
            self._trim()

    def add(self, manifest: RetroSnapshotManifest) -> list[RetroSnapshotManifest]:
        """Add an available snapshot and return evicted manifests for deletion."""

        if not manifest.is_eligible():
            raise ValueError("RetroBuffer only accepts available, unexpired snapshots")
        with self._lock:
            self._items.append(manifest)
            evicted = self._trim()
            self._persist_full(manifest)
            for item in evicted:
                item.status = SnapshotStatus.INVALID
                self._persist_transition(item)
            return evicted

    def lease(
        self,
        consumer_id: str,
        *,
        current_update: int | None = None,
        min_policy_age: int | None = None,
        max_policy_age: int | None = None,
        instance_id: str | None = None,
        event_type: str | None = None,
        order: str = "fifo",
    ) -> RetroSnapshotManifest | None:
        with self._lock:
            candidates = [
                manifest
                for manifest in self._items
                if (instance_id is None or manifest.instance_id == instance_id)
                and (event_type is None or manifest.event_type == event_type)
                and manifest.is_eligible(
                    current_update=current_update,
                    min_policy_age=min_policy_age,
                    max_policy_age=max_policy_age,
                )
            ]
            if order == "newest":
                candidates.sort(
                    key=lambda item: (
                        item.source_update if item.source_update is not None else -1,
                        item.created_at,
                        item.snapshot_id,
                    ),
                    reverse=True,
                )
            elif order != "fifo":
                raise ValueError("retro pool order must be fifo or newest")
            for manifest in candidates:
                if instance_id is not None and manifest.instance_id != instance_id:
                    continue
                manifest.lease(consumer_id)
                self._persist_transition(manifest)
                return manifest
        return None

    def release(self, snapshot_id: str) -> None:
        with self._lock:
            manifest = self._get(snapshot_id)
            manifest.release()
            self._persist_transition(manifest)

    def consume(self, snapshot_id: str, rollout_id: int | None = None) -> RetroSnapshotManifest:
        with self._lock:
            manifest = self._get(snapshot_id)
            manifest.consume(rollout_id)
            self._items.remove(manifest)
            self._persist_transition(manifest)
            return manifest

    def available(
        self,
        *,
        current_update: int | None = None,
        min_policy_age: int | None = None,
        max_policy_age: int | None = None,
        event_type: str | None = None,
    ) -> int:
        with self._lock:
            return sum(
                (event_type is None or item.event_type == event_type)
                and item.is_eligible(
                    current_update=current_update,
                    min_policy_age=min_policy_age,
                    max_policy_age=max_policy_age,
                )
                for item in self._items
            )

    def manifests(self) -> list[RetroSnapshotManifest]:
        with self._lock:
            return list(self._items)

    def _get(self, snapshot_id: str) -> RetroSnapshotManifest:
        for manifest in self._items:
            if manifest.snapshot_id == snapshot_id:
                return manifest
        raise KeyError(f"retro snapshot not in buffer: {snapshot_id}")

    def _trim(self) -> list[RetroSnapshotManifest]:
        evicted: list[RetroSnapshotManifest] = []
        while len(self._items) > self.max_items:
            evicted.append(self._items.popleft())
        return evicted

    def _persist_full(self, manifest: RetroSnapshotManifest) -> None:
        if self.store is not None:
            self.store.append(manifest)

    def _persist_transition(self, manifest: RetroSnapshotManifest) -> None:
        if self.store is not None:
            self.store.append_transition(manifest)
