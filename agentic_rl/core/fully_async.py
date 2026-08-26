"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path agentic_rl.core.fully_async.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

De-forked from ``slime/rollout/fully_async_rollout.py`` (RUNBOOK §7 step 3):
this is a project-owned copy, and the slime file is back at upstream. The two
CLI flags the fork added are env knobs here, normalized onto ``args`` at entry
by :func:`resolve_async_rl_rollout_knobs` so every downstream ``args.<attr>``
read (retro's lane splitter, the prefetch worker, metrics) stays consistent:

* ``ASYNC_RL_ROLLOUT_PREFETCH_BATCHES`` → ``args.rollout_prefetch_batches``
* ``ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG`` → ``args.rollout_max_behavior_lag``

For both, unset/empty defers to any value already on ``args`` (custom-config
YAML or a fork-era flag), and <= 0 means disabled (None).

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`. When ``args.rollout_prefetch_batches``
(or its deprecated alias ``rollout_max_staleness``) is set, the in-flight pool
(generating + completed-but-unshipped) is instead capped at
``rollout_prefetch_batches * rollout_batch_size`` groups. That is a capacity
knob only — it bounds behavior-policy lag on average (Little's law), never per
sample. The enforced bound is ``args.rollout_max_behavior_lag``: at batch
assembly, a group whose oldest token was generated more than that many weight
updates before the current rollout's publishing version (rollout t generates
under absolute version t + 1) is discarded and its prompt requeued for
regeneration.

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
import os
import inspect
import logging
import queue
import re
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
    "behavior_lag",
    "generate_rollout_fully_async",
    "group_min_weight_version",
    "reset_group_for_regeneration",
    "resolve_async_rl_rollout_knobs",
]

logger = logging.getLogger("agentic_rl.core.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _env_int_knob(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def resolve_async_rl_rollout_knobs(args) -> None:
    """Normalize the de-forked env knobs onto ``args`` (idempotent).

    Env wins, then whatever is already on ``args``, then None. <= 0 disables
    (prefetch falls back to the engine cap; the lag gate turns off). Called at
    every rollout entry point so args-attribute reads downstream can never
    disagree with what the worker actually uses.
    """
    for env_name, attr in (
        ("ASYNC_RL_ROLLOUT_PREFETCH_BATCHES", "rollout_prefetch_batches"),
        ("ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG", "rollout_max_behavior_lag"),
    ):
        value = _env_int_knob(env_name)
        if value is not None:
            setattr(args, attr, value if value > 0 else None)
        elif not hasattr(args, attr):
            setattr(args, attr, None)


def _pool_size(args) -> int:
    """In-flight group budget for the async worker.

    The engine-side cap (sglang_server_concurrency x engines) is what the
    serving stack can sustain; the prefetch window (rollout_prefetch_batches x
    rollout_batch_size) is how far generation may run ahead of training. Take
    the min so neither bound is violated.

    Capacity only: this bounds behavior-policy lag on *average* (Little's law:
    lag ~= in-flight / consumed-per-step) but never per sample. The enforced
    per-group bound is ``rollout_max_behavior_lag``, checked at batch assembly.
    """
    engine_cap = args.sglang_server_concurrency * get_rollout_num_engines(args)
    prefetch = getattr(args, "rollout_prefetch_batches", None)
    if prefetch is None:
        prefetch = getattr(args, "rollout_max_staleness", None)
        if prefetch is not None:
            logger.warning(
                "fully-async: rollout_max_staleness=%d is a deprecated alias for "
                "rollout_prefetch_batches (it sizes the pool, it does not bound staleness); "
                "set rollout_prefetch_batches, and rollout_max_behavior_lag for an enforced bound",
                prefetch,
            )
    if prefetch is None:
        return engine_cap
    window = prefetch * args.rollout_batch_size
    if window < engine_cap:
        logger.info(
            "fully-async: prefetch window caps in-flight pool at %d groups "
            "(%d batches x rollout_batch_size=%d; engine cap was %d)",
            window,
            prefetch,
            args.rollout_batch_size,
            engine_cap,
        )
    return min(window, engine_cap)


def _version_num(version) -> int | None:
    """Parse an engine-reported weight version ("81", "weight_v000081") to an int."""
    m = re.search(r"\d+", str(version))
    return int(m.group()) if m else None


def group_min_weight_version(group: list[Sample]) -> int | None:
    """Oldest weight version among all tokens of a group, or None if unrecorded."""
    nums = [
        n
        for s in group
        for n in (_version_num(v) for v in getattr(s, "weight_versions", None) or [])
        if n is not None
    ]
    return min(nums) if nums else None


def reset_group_for_regeneration(group: list[Sample]) -> list[Sample]:
    """Strip a completed group back to a pristine prompt-group.

    A lag-rejected group must actually REGENERATE when requeued —
    ``generate_and_rm`` short-circuits COMPLETED/TRUNCATED samples untouched,
    so requeueing as-is spins the group through buffer -> worker -> reject
    forever (observed: 8553 rejections in one step on rlag4).
    """
    for s in group:
        s.status = Sample.Status.PENDING
        s.response = ""
        s.response_length = 0
        s.tokens = []
        s.reward = None
        s.loss_mask = None
        s.weight_versions = []
        s.rollout_log_probs = None
        s.remove_sample = False
        s.spec_info = Sample.SpecInfo()
        if isinstance(s.metadata, dict):
            s.metadata.pop("agentic", None)
    return group


def behavior_lag(group: list[Sample], rollout_id: int) -> int | None:
    """Behavior-policy lag of a group vs the weights that generate rollout_id.

    Weight versions are absolute and rollout t generates under version t + 1
    (the updater increments before publishing; resumes preserve this). Lag is
    measured against the group's *oldest* token so mixed-version episodes are
    bounded by their stalest part. None when no version was recorded.
    """
    oldest = group_min_weight_version(group)
    return None if oldest is None else (rollout_id + 1) - oldest


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


async def _call_group_hook(hook, group: list[Sample]) -> None:
    if hook is None:
        return
    result = hook(group)
    if inspect.isawaitable(result):
        await result


async def _generate_rollout_async(
    args,
    rollout_id: int,
    data_buffer,
    *,
    group_accept_hook=None,
    group_reject_hook=None,
) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    resolve_async_rl_rollout_knobs(args)
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
    max_behavior_lag = getattr(args, "rollout_max_behavior_lag", None)
    n_lag_rejected = 0
    max_rejected_lag = 0
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
            if max_behavior_lag is not None:
                lag = behavior_lag(group, rollout_id)
                if lag is not None and lag > max_behavior_lag:
                    # Too stale to train on. Tokens are unsalvageable (the group
                    # only gets staler); invalidate side effects and requeue the
                    # prompt for regeneration under the current weights.
                    n_lag_rejected += 1
                    max_rejected_lag = max(max_rejected_lag, lag)
                    await _call_group_hook(group_reject_hook, group)
                    try:
                        data_buffer.add_samples([reset_group_for_regeneration(group)])
                    except Exception:  # noqa: BLE001
                        logger.exception("fully-async: failed to requeue lag-rejected group")
                    continue
            for s in group:
                r = s.get_reward_value(args)
                if r is not None:
                    prefilter_reward_sum += float(r)
                    prefilter_reward_n += 1
            filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                n_dropped += 1
                await _call_group_hook(group_reject_hook, group)
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

    ordered = sorted(collected.items(), key=lambda item: _key(item[1]))
    selected = ordered[:target]
    out = [group for _, group in selected]
    selected_ids = {gid for gid, _ in selected}
    for gid, group in ordered:
        await _call_group_hook(group_accept_hook if gid in selected_ids else group_reject_hook, group)
    metrics = metric_gatherer.collect()
    if max_behavior_lag is not None:
        metrics["behavior_lag/rejected_groups"] = n_lag_rejected
        metrics["behavior_lag/max_rejected_lag"] = max_rejected_lag
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
