"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`. When ``args.rollout_max_staleness`` is
set, the in-flight pool (generating + completed-but-unshipped) is instead
capped at ``rollout_max_staleness * rollout_batch_size`` groups, so by
Little's law a group is trained at most ~``rollout_max_staleness`` weight
updates after it started generating.

``args.dynamic_sampling_filter_path`` (DAPO) is honored at collection time:
completed groups that fail the filter (e.g. zero reward std) are dropped and
the collector keeps pulling until ``rollout_batch_size`` passing groups are
gathered. Over-generation is free here — the pool generates continuously —
but the *logged* ``rollout/raw_reward`` becomes biased by the filter, so the
unbiased pre-filter mean is emitted as ``dynamic_sampling/raw_reward_all``.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _pool_size(args) -> int:
    """In-flight group budget for the async worker.

    The engine-side cap (sglang_server_concurrency x engines) is what the
    serving stack can sustain; the staleness window (rollout_max_staleness x
    rollout_batch_size) is what the trainer can consume before samples go
    stale. Take the min so neither bound is violated.
    """
    engine_cap = args.sglang_server_concurrency * get_rollout_num_engines(args)
    staleness = getattr(args, "rollout_max_staleness", None)
    if staleness is None:
        return engine_cap
    window = staleness * args.rollout_batch_size
    if window < engine_cap:
        logger.info(
            "fully-async: staleness window caps in-flight pool at %d groups "
            "(rollout_max_staleness=%d x rollout_batch_size=%d; engine cap was %d)",
            window,
            staleness,
            args.rollout_batch_size,
            engine_cap,
        )
    return min(window, engine_cap)


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=_pool_size(args))
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                # Top up. Completed-but-unshipped groups (output_queue backlog)
                # count against the budget: they are already generated and only
                # get staler while they wait, so generating past them would let
                # staleness grow beyond the window unboundedly.
                while len(active_tasks) + self.output_queue.qsize() < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid))
                        active_tasks.add(task)

                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            # Aborted group → requeue, don't ship to training.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    # DAPO dynamic sampling: drop groups failing the filter (e.g. zero reward
    # std) and keep collecting. Over-generation is free — the pool keeps
    # producing — the only cost is a longer wait for `target` passing groups.
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    n_completed = 0
    n_dropped = 0
    prefilter_reward_sum = 0.0
    prefilter_reward_n = 0
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull whatever's done.
        drained = 0
        for gid, group in worker.get_completed_groups():
            drained += 1
            n_completed += 1
            for s in group:
                r = s.get_reward_value(args)
                if r is not None:
                    prefilter_reward_sum += float(r)
                    prefilter_reward_n += 1
            filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                n_dropped += 1
                continue
            collected[gid] = group

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d (dropped %d), queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                n_dropped,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)[:target]
    metrics = metric_gatherer.collect()
    if dynamic_filter is not None:
        metrics["dynamic_sampling/completed_groups"] = n_completed
        metrics["dynamic_sampling/dropped_groups"] = n_dropped
        if prefilter_reward_n > 0:
            # Unbiased mean reward over ALL completed groups this step; the
            # post-filter rollout/raw_reward is biased toward mixed outcomes.
            metrics["dynamic_sampling/raw_reward_all"] = prefilter_reward_sum / prefilter_reward_n
    logger.info(
        "fully-async rollout %d: done in %.1fs (dropped %d/%d), queue_left=%d",
        rollout_id,
        time.time() - started,
        n_dropped,
        n_completed,
        worker.queue_size(),
    )
    return RolloutFnTrainOutput(samples=out, metrics=metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
