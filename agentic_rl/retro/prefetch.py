"""Cross-step prefetch buffer for the retro lane.

The retro lane was made-to-order: each step leased snapshots, generated eight
continuations, and trained on them within the same collection window. That
made it *effectively synchronous* — measured behavior lag 0.46, never above 1,
single behavior version per group — with two consequences:

1. **It was a straggler barrier.** The step could not finish until the slowest
   of its 64 retro episodes did, so the retro leg was the critical path in
   68-73% of steps (rlag4/rlag8, 85 steps each) even though the fresh lane
   delivered in half the time.
2. **Retro behavior-lag bounds were inert.** ``RETRO_MAX_BEHAVIOR_LAG`` of 4 or
   8 can never bind a quantity that never exceeds 1, which is why the
   rlag4-vs-rlag8 arms came out as seed replicates.

This module gives the retro lane the same queue-ahead structure the fresh lane
already has (:class:`agentic_rl.core.fully_async.AsyncRolloutWorker`): a
background thread keeps a pool of pre-generated retro groups, so the step
drains a ready queue instead of waiting on generation. Groups then age across
weight updates, which is what makes realized retro lag tunable — the depth is
the treatment variable, and the lag gate remains the enforced bound.

Depth 0 keeps the legacy made-to-order path byte-for-byte, so the control arm
is unchanged.

**Concurrency constraint.** ``GenerateState`` is a process singleton holding one
``asyncio.Semaphore`` sized ``sglang_server_concurrency * num_engines``. It is
already awaited from two event loops (the fresh worker's thread and the main
rollout loop); that is only safe because the capacity is never exhausted, so
``acquire()`` never creates a cross-loop waiter future. This worker adds a
third loop, so the pool sizes must keep total in-flight episodes strictly under
that capacity — :func:`check_concurrency_headroom` warns when they do not.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import queue
import threading
import uuid

from agentic_rl.core.fully_async import behavior_lag
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

from .group import make_branch_group, template_from_manifest
from .selection import choose_branch_turn
from .pool import Lease, ReplayPool

logger = logging.getLogger("agentic_rl.retro.prefetch")

# Group/sample indices must be unique across the whole run, not just within a
# step: prefetched groups outlive the rollout that generated them.
_INDEX_BASE = 2_000_000_000


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def retro_prefetch_groups(args) -> int:
    """In-flight/queued retro group budget. 0 disables prefetch entirely."""

    batches = _env_int("ASYNC_RL_RETRO_PREFETCH_BATCHES", 0)
    if batches <= 0:
        return 0
    ratio = min(1.0, max(0.0, _env_float("ASYNC_RL_RETRO_GROUP_RATIO", 0.25)))
    total = int(getattr(args, "rollout_batch_size", 8))
    per_batch = max(1, min(total - 1, round(total * ratio)) if total > 1 else 0)
    return batches * per_batch


def check_concurrency_headroom(args, retro_groups: int) -> None:
    """Warn if fresh + retro pools can exhaust the shared GenerateState semaphore.

    Exhausting it would make two event loops park waiters on one semaphore,
    which is not cross-loop safe (see the module docstring).
    """

    n_samples = int(getattr(args, "n_samples_per_prompt", 8))
    capacity = int(getattr(args, "sglang_server_concurrency", 0)) * get_rollout_num_engines(args)
    prefetch = getattr(args, "rollout_prefetch_batches", None) or 1
    fresh_groups = prefetch * int(getattr(args, "rollout_batch_size", 8))
    in_flight = (fresh_groups + retro_groups) * n_samples
    if capacity and in_flight >= capacity:
        logger.warning(
            "retro prefetch: fresh(%d groups) + retro(%d groups) = %d episodes >= "
            "GenerateState semaphore capacity %d. The semaphore is shared across event "
            "loops and is only safe while it never blocks; raise "
            "sglang_server_concurrency or lower the pools.",
            fresh_groups,
            retro_groups,
            in_flight,
            capacity,
        )


class RetroPrefetchWorker:
    """Background thread keeping ``depth`` retro groups generated-and-waiting.

    Owns the only long-lived :class:`ReplayPool`: constructing a second one
    would release this worker's in-flight leases (the pool's reload releases
    every LEASED manifest it finds).
    """

    def __init__(self, args, *, manifest_path: str, depth: int):
        self.args = args
        self.depth = depth
        self.pool = ReplayPool(
            manifest_path,
            max_items=max(1024, int(getattr(args, "rollout_batch_size", 8)) * 16),
        )
        self.store = self.pool.store
        # (lease, group) pairs, generated and awaiting a training step.
        self.ready: queue.Queue[tuple[Lease, list[Sample]]] = queue.Queue()
        self.running = True
        self.thread: threading.Thread | None = None
        self._index_lock = threading.Lock()
        self._next_index = 0
        self.stats = {"leased": 0, "generated": 0, "failed": 0, "lease_misses": 0}

    # -- public ------------------------------------------------------------

    def start(self) -> None:
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(
                target=lambda: asyncio.run(self._loop()),
                name="retro-prefetch",
                daemon=True,
            )
            self.thread.start()
            logger.info("retro prefetch worker started (depth=%d groups)", self.depth)

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def queue_size(self) -> int:
        return self.ready.qsize()

    def drain(self, limit: int) -> list[tuple[Lease, list[Sample]]]:
        """Take up to ``limit`` ready (lease, group) pairs."""

        out: list[tuple[Lease, list[Sample]]] = []
        while len(out) < limit:
            try:
                out.append(self.ready.get_nowait())
            except queue.Empty:
                break
        return out

    def give_back(self, item: tuple[Lease, list[Sample]]) -> None:
        """Return a ready group unconsumed (e.g. it did not match the event
        quota this step). It keeps its snapshot lease and ages another update,
        which is the intended prefetch behavior."""

        self.ready.put(item)

    def release(self, lease: Lease) -> None:
        try:
            lease.release()
        except KeyError:
            logger.warning("retro prefetch: snapshot %s not in pool", lease.snapshot_id)

    # -- internals ---------------------------------------------------------

    def _claim_index(self) -> int:
        with self._index_lock:
            n = self._next_index
            self._next_index += 1
            return n

    def _quota_targets(self) -> dict[str, int]:
        ratio = min(1.0, max(0.0, _env_float("ASYNC_RL_RETRO_POOL_PROMISING_RATIO", 0.5)))
        promising = int(math.floor(self.depth * ratio + 0.5))
        return {"promising": promising, "recovery": self.depth - promising}

    def _queued_by_type(self, active_types: list[str]) -> dict[str, int]:
        counts = {"promising": 0, "recovery": 0}
        for event_type in active_types:
            if event_type in counts:
                counts[event_type] += 1
        with self.ready.mutex:
            pending = list(self.ready.queue)
        for lease, _ in pending:
            event_type = getattr(lease, "event_type", None)
            if event_type in counts:
                counts[event_type] += 1
        return counts

    async def _loop(self) -> None:
        state = GenerateState(self.args)
        active: dict[asyncio.Task, tuple[Lease, str]] = {}
        pool_order = os.environ.get("ASYNC_RL_RETRO_POOL_ORDER", "newest").strip().lower()
        min_age = _env_int("ASYNC_RL_RETRO_MIN_POLICY_AGE", 0)
        max_age = _env_int("ASYNC_RL_RETRO_MAX_POLICY_AGE", 4)

        while self.running:
            try:
                for task in [t for t in active if t.done()]:
                    lease, _ = active.pop(task)
                    try:
                        result = task.result()
                    except Exception as exc:  # noqa: BLE001
                        self.stats["failed"] += 1
                        self.release(lease)
                        logger.warning("retro prefetch generate failed (%s): %r", lease.snapshot_id, exc)
                        continue
                    if not isinstance(result, list) or len(result) != 8:
                        self.stats["failed"] += 1
                        self.release(lease)
                        continue
                    self.stats["generated"] += 1
                    self.ready.put((lease, result))

                self.pool.refresh()
                targets = self._quota_targets()
                counts = self._queued_by_type([t for _, t in active.values()])
                # The trainer's version is not visible here; policy-age filtering
                # uses the newest snapshot generation the buffer knows about, and
                # the authoritative check is the lag gate at drain time.
                for event_type in ("promising", "recovery"):
                    while (
                        self.running
                        and counts[event_type] < targets[event_type]
                        and len(active) + self.ready.qsize() < self.depth
                    ):
                        consumer_id = f"prefetch-{uuid.uuid4().hex[:8]}"
                        lease = self.pool.lease(
                            consumer_id,
                            min_policy_age=min_age,
                            max_policy_age=max_age,
                            event_type=event_type,
                            order=pool_order,
                        )
                        if lease is None:
                            self.stats["lease_misses"] += 1
                            break
                        self.stats["leased"] += 1
                        counts[event_type] += 1
                        n = self._claim_index()
                        try:
                            group = make_branch_group(
                                template_from_manifest(lease.manifest),
                                lease.manifest,
                                group_index=_INDEX_BASE // 8 + n,
                                first_sample_index=_INDEX_BASE + n * 8,
                                branch_turn=choose_branch_turn(lease.manifest),
                            )
                        except Exception:  # noqa: BLE001
                            self.release(lease)
                            logger.exception("retro prefetch: could not build branch group")
                            continue
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        active[task] = (lease, event_type)

                await asyncio.sleep(1)
            except Exception:  # noqa: BLE001
                logger.exception("retro prefetch loop iteration error")
                await asyncio.sleep(1)

        for task in active:
            task.cancel()


_worker: RetroPrefetchWorker | None = None
_worker_lock = threading.Lock()


def get_worker(args, *, manifest_path: str) -> RetroPrefetchWorker | None:
    """Process-wide retro prefetch worker, or None when prefetch is disabled."""

    global _worker
    depth = retro_prefetch_groups(args)
    if depth <= 0:
        return None
    with _worker_lock:
        if _worker is None or not (_worker.thread and _worker.thread.is_alive()):
            check_concurrency_headroom(args, depth)
            _worker = RetroPrefetchWorker(args, manifest_path=manifest_path, depth=depth)
            _worker.start()
        return _worker


def drain_ready_groups(
    worker: RetroPrefetchWorker,
    *,
    args,
    rollout_id: int,
    target: int,
    event_targets: dict[str, int],
    max_behavior_lag: int | None,
) -> tuple[list[tuple[Lease, list[Sample]]], dict[str, int]]:
    """Pull up to ``target`` groups honoring the event quota and the lag gate.

    Groups that do not fit this step's quota are handed back to the queue
    (they age one more update, as prefetch intends). Groups over the lag bound
    are dropped and their snapshots released for a later attempt.
    """

    taken: list[tuple[Lease, list[Sample]]] = []
    accepted_by_event = {"promising": 0, "recovery": 0}
    counters = {"lag_rejected": 0, "quota_deferred": 0, "aborted": 0}
    # Bound the scan so a queue full of wrong-type groups cannot spin.
    for item in worker.drain(max(target * 4, target + 8)):
        lease, group = item
        if len(taken) >= target:
            worker.give_back(item)
            continue
        if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in group):
            counters["aborted"] += 1
            worker.release(lease)
            continue
        if max_behavior_lag is not None:
            lag = behavior_lag(group, rollout_id)
            if lag is not None and lag > max_behavior_lag:
                counters["lag_rejected"] += 1
                worker.release(lease)
                continue
        event_type = getattr(lease, "event_type", None)
        if event_type in accepted_by_event and accepted_by_event[event_type] >= event_targets.get(event_type, 0):
            counters["quota_deferred"] += 1
            worker.give_back(item)
            continue
        if event_type in accepted_by_event:
            accepted_by_event[event_type] += 1
        taken.append(item)
    return taken, counters
