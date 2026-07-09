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
:mod:`slime.rollout.sglang_rollout`.

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


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
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
        # Windowed-FIFO staleness control (Forge-style). The staleness window throttles GENERATION, not
        # consumption: the worker won't submit a group more than ``max_lead`` gids ahead of the OLDEST
        # in-flight group (``inflight_gids``), so a slow straggler holds back new generation — pinning the
        # window — but never blocks the trainer. The collector keeps sampling the freshest completed groups
        # (oldest-completed first) from the larger generation pool. Because generation can't run more than
        # max_lead = max_staleness × rollout_batch_size ahead of the oldest unfinished request, every trained
        # sample (including a slow head, once it finishes) is within max_staleness weight versions.
        self.completed_buffer: dict[int, list[Sample]] = {}
        self.inflight_gids: set[int] = set()  # gids currently generating; worker-thread-only
        max_staleness = getattr(args, "rollout_max_staleness", None)
        self.max_lead = max_staleness * args.rollout_batch_size if max_staleness else None

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

                # Top up — but not past the staleness window (see __init__): don't submit a group more than
                # max_lead gids ahead of the oldest in-flight group.
                while len(active_tasks) < max_concurrent and self.running:
                    if self.max_lead is not None and self.inflight_gids:
                        if gid_counter - min(self.inflight_gids) >= self.max_lead:
                            break
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        self.inflight_gids.add(gid)
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
            self.inflight_gids.discard(gid)  # no longer generating → unpins the staleness window
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
            # Aborted group → requeue for redo under a fresh gid, don't ship to training. Reset EVERY
            # sibling to PENDING: on re-pull, generate_and_rm short-circuits COMPLETED samples (returns
            # them verbatim, no regeneration), which would ship the siblings' old-policy trajectories under
            # a fresh gid — stale data outside the staleness window. PENDING forces a full-group redo.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                for s in result:
                    s.status = Sample.Status.PENDING
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    # Dynamic sampling (DAPO): if a filter is configured, discard groups it rejects (e.g. zero reward-variance
    # → no GRPO gradient) and keep pulling until `target` groups PASS. Over-generation is free here — the
    # windowed-FIFO already runs the generation pool ahead of the trainer — so we just filter the pool rather
    # than launch extra rounds. raw_reward is logged over ALL examined groups (pre-filter) so the metric is the
    # unbiased generation signal, not the selected-only (upward-biased) subset.
    dyn_filter = None
    fpath = getattr(args, "dynamic_sampling_filter_path", None)
    if fpath:
        dyn_filter = load_function(fpath)
    # Starvation guard: if the policy can't produce `target` passing groups even after a large over-sample,
    # accept rejected groups rather than hang the trainer (and the low kept_frac will show in the metrics).
    over_sample_cap = target * 8

    # Windowed-FIFO consumption: sample the oldest-completed groups first from the generation pool; never
    # block on an in-flight straggler (staleness is bounded on the generation side — see the worker's
    # __init__). Leftover completed groups stay buffered across steps.
    buf = worker.completed_buffer
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    collected: list[list[Sample]] = []
    n_examined = 0  # groups passed through the filter this rollout (pre-filter denominator)
    n_kept = 0  # groups the filter accepted
    reward_sum_all = 0.0  # sum of per-group mean reward over ALL examined groups (pre-filter)
    while len(collected) < target:
        for gid, group in worker.get_completed_groups():
            buf[gid] = group

        for gid in sorted(buf):  # oldest-completed first
            if len(collected) >= target:
                break
            group = buf.pop(gid)
            if dyn_filter is None:
                collected.append(group)
                continue
            n_examined += 1
            reward_sum_all += _group_mean_reward(group, args)
            if dyn_filter(args, group).keep:
                collected.append(group)
                n_kept += 1
            elif n_examined >= over_sample_cap:
                collected.append(group)  # starvation fallback — take a rejected group rather than hang

        if len(collected) < target:
            await asyncio.sleep(0.05)  # pool not yet deep enough — wait for more completions

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, examined=%d kept=%d, buffered=%d, in_flight=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                n_examined,
                n_kept,
                len(buf),
                len(worker.inflight_gids),
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

    out = sorted(collected, key=_key)
    if dyn_filter is not None and n_examined:
        _log_dynamic_sampling(args, rollout_id, n_examined, n_kept, reward_sum_all / n_examined)
    logger.info(
        "fully-async rollout %d: done in %.1fs, buffered_left=%d, in_flight=%d",
        rollout_id,
        time.time() - started,
        len(buf),
        len(worker.inflight_gids),
    )
    return out


def _group_mean_reward(group: list[Sample], args) -> float:
    rs = [s.get_reward_value(args) for s in group if getattr(s, "reward", None) is not None]
    return sum(rs) / len(rs) if rs else 0.0


def _log_dynamic_sampling(args, rollout_id: int, n_examined: int, n_kept: int, raw_reward_all: float) -> None:
    """Emit dynamic-sampling telemetry. raw_reward_all is the UNBIASED mean reward over every generated group
    (pre-filter) — the true learning signal; slime's rollout/raw_reward is over the kept (non-zero-std) subset
    and reads high once the filter is active."""
    try:
        from slime.ray.rollout import compute_rollout_step
        from slime.utils import logging_utils

        kept_frac = n_kept / n_examined
        metrics = {
            "dynamic_sampling/kept_frac": kept_frac,
            "dynamic_sampling/filtered_frac": 1.0 - kept_frac,
            "dynamic_sampling/groups_examined": float(n_examined),
            "dynamic_sampling/raw_reward_all": raw_reward_all,
            "rollout/step": compute_rollout_step(args, rollout_id),
        }
        logging_utils.log(args, metrics, step_key="rollout/step")
        logger.info(
            "fully-async rollout %d: dynamic-sampling kept %d/%d (%.0f%%), raw_reward_all=%.3f",
            rollout_id,
            n_kept,
            n_examined,
            100 * kept_frac,
            raw_reward_all,
        )
    except Exception:  # noqa: BLE001 — telemetry must never crash the rollout
        logger.exception("dynamic-sampling logging failed (non-fatal)")


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
