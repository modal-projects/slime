"""ReplayPool + Lease: the ONE owner of manifest lifecycle and snapshot GC.

RUNBOOK §7.1 (P2/P5): every status transition —

    capture ─offer→ TENTATIVE ─commit→ AVAILABLE ─lease→ LEASED ─consume→ CONSUMED → GC
                        └─abort→ INVALID → GC          └─release→ AVAILABLE

— and every snapshot deletion goes through this module; callers never touch
``delete_snapshot`` or ``ManifestStore.append_transition`` directly. The
JSONL persistence and in-memory queue stay in :mod:`.buffer` (append-only
last-record-wins, lease-release-on-reload — proven, unchanged); this class is
the API in front of them.

A :class:`Lease` is a context manager: falling out of a ``with`` block without
``consume()`` auto-releases the snapshot back to AVAILABLE, so the
crash-safety idiom is impossible to get wrong::

    lease = pool.lease("worker-1", event_type="promising", order="newest")
    if lease is not None:
        with lease:
            group = make_branch_group(template_from_manifest(lease.manifest), lease.manifest, ...)
            out = await generate(group)
            if accept(out):
                await lease.consume(rollout_id)   # → CONSUMED + snapshot GC
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from .buffer import ManifestStore, RetroBuffer
from .manifest import RetroSnapshotManifest, SnapshotStatus

logger = logging.getLogger("agentic_rl.retro.pool")


def _default_gc(snapshot_id: str) -> None:
    from .backends.modal_snapshot import delete_snapshot

    delete_snapshot(snapshot_id)


class Lease:
    """A leased snapshot. Exactly one of ``consume()`` / ``release()`` ends it;
    leaving a ``with`` block without either releases it back to AVAILABLE."""

    def __init__(self, pool: "ReplayPool", manifest: RetroSnapshotManifest):
        self._pool = pool
        self.manifest = manifest
        self._closed = False

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id

    @property
    def event_type(self) -> str:
        return self.manifest.event_type

    async def consume(self, rollout_id: int | None = None, *, gc: bool = True) -> None:
        """LEASED → CONSUMED, then best-effort snapshot deletion."""

        if self._closed:
            return
        self._closed = True
        self._pool._buffer.consume(self.manifest.snapshot_id, rollout_id)
        if gc:
            await self._pool.gc_snapshot(self.manifest.snapshot_id)

    def release(self) -> None:
        """LEASED → AVAILABLE (retryable later). Raises KeyError if the
        manifest already left the pool — same contract as RetroBuffer."""

        if self._closed:
            return
        self._closed = True
        self._pool._buffer.release(self.manifest.snapshot_id)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            try:
                self.release()
            except KeyError:
                logger.warning("lease exit: snapshot %s not in pool", self.manifest.snapshot_id)


class ReplayPool:
    """Facade over ``ManifestStore`` + ``RetroBuffer`` owning transitions + GC.

    ``gc`` is the snapshot deleter (Modal image delete by default); injectable
    for tests and for alternative :class:`~.protocols.SnapshotBackend`\\ s.
    GC failures are logged, never raised — a leaked snapshot ages out via TTL.
    """

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        store: ManifestStore | None = None,
        max_items: int = 1024,
        gc: Callable[[str], None] | None = None,
    ):
        if store is None:
            if manifest_path is None:
                raise ValueError("ReplayPool needs a manifest_path or a store")
            store = ManifestStore(manifest_path)
        self.store = store
        self._buffer = RetroBuffer(max_items=max_items, store=store)
        self._gc = gc or _default_gc

    # -- acquisition ---------------------------------------------------------

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
    ) -> Lease | None:
        manifest = self._buffer.lease(
            consumer_id,
            current_update=current_update,
            min_policy_age=min_policy_age,
            max_policy_age=max_policy_age,
            instance_id=instance_id,
            event_type=event_type,
            order=order,
        )
        return None if manifest is None else Lease(self, manifest)

    def refresh(self) -> int:
        """Adopt snapshots activated since construction (long-lived pools)."""

        return self._buffer.refresh_from_store()

    def available(
        self,
        *,
        current_update: int | None = None,
        min_policy_age: int | None = None,
        max_policy_age: int | None = None,
        event_type: str | None = None,
    ) -> int:
        return self._buffer.available(
            current_update=current_update,
            min_policy_age=min_policy_age,
            max_policy_age=max_policy_age,
            event_type=event_type,
        )

    def manifests(self) -> list[RetroSnapshotManifest]:
        return self._buffer.manifests()

    # -- candidate (TENTATIVE) lifecycle --------------------------------------

    async def commit_candidates(self, group: Any) -> None:
        await commit_candidates(self.store, group)

    async def abort_candidates(self, group: Any) -> None:
        await abort_candidates(self.store, group, gc=self._gc)

    # -- GC -------------------------------------------------------------------

    async def gc_snapshot(self, snapshot_id: str) -> None:
        await gc_snapshot(snapshot_id, gc=self._gc)


async def commit_candidates(store: ManifestStore, group: Any) -> None:
    """Batch-accept hook: every TENTATIVE candidate carried by ``group``
    becomes AVAILABLE (it entered training, so it may seed replay).

    Store-only on purpose: the accept/reject hooks fire during the fresh leg,
    which can run before any ReplayPool exists (pool construction time decides
    the minimum leasable snapshot age — see rollout's sequential-legs switch).
    """

    for manifest in _candidate_manifests(group):
        if manifest.status != SnapshotStatus.TENTATIVE:
            continue
        manifest.activate()
        store.append_transition(manifest)


async def abort_candidates(
    store: ManifestStore, group: Any, *, gc: Callable[[str], None] | None = None
) -> None:
    """Batch-reject hook: TENTATIVE candidates → INVALID + snapshot GC."""

    for manifest in _candidate_manifests(group):
        if manifest.status != SnapshotStatus.TENTATIVE:
            continue
        manifest.invalidate()
        store.append_transition(manifest)
        await gc_snapshot(manifest.snapshot_id, gc=gc)


async def gc_snapshot(snapshot_id: str, *, gc: Callable[[str], None] | None = None) -> None:
    """Best-effort snapshot deletion off the event loop; failures only warn."""

    try:
        await asyncio.to_thread(gc or _default_gc, snapshot_id)
    except RuntimeError as exc:
        logger.warning("retro snapshot cleanup %s: %s", snapshot_id, exc)


def _candidate_manifests(group: Any) -> list[RetroSnapshotManifest]:
    """Collect the retro candidates a fresh group carried in sample metadata."""

    manifests: dict[str, RetroSnapshotManifest] = {}

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        metadata = getattr(value, "metadata", None) or {}
        candidates = (metadata.get("agentic") or {}).get("retro_candidates") or []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                manifest = RetroSnapshotManifest.from_dict(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            manifests[manifest.snapshot_id] = manifest

    visit(group)
    return list(manifests.values())
