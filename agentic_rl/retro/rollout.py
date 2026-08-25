"""Phase-1 debug rollout: evaluate eight siblings from saved snapshots."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import time
import uuid

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.rollout.fully_async_rollout import _generate_rollout_async, behavior_lag
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

from .buffer import ManifestStore, RetroBuffer
from .group import make_branch_group, template_from_manifest
from .manifest import RetroSnapshotManifest, SnapshotStatus
from .prefetch import drain_ready_groups, get_worker as get_prefetch_worker, retro_prefetch_groups
from .snapshot import delete_snapshot

logger = logging.getLogger("agentic_rl.retro")


async def _generate_retro_survey(args, rollout_id: int) -> RolloutFnTrainOutput:
    manifest_path = os.environ.get("ASYNC_RL_RETRO_MANIFEST_PATH")
    if not manifest_path:
        raise ValueError("ASYNC_RL_RETRO_MANIFEST_PATH is required for the retro survey")

    store = ManifestStore(manifest_path)
    buffer = RetroBuffer(max_items=max(1, int(getattr(args, "rollout_batch_size", 1)) * 4), store=store)
    target = int(getattr(args, "rollout_batch_size", 1))
    state = GenerateState(args)
    leased = []
    groups = []
    for group_offset in range(target):
        consumer_id = f"survey-{rollout_id}-{group_offset}-{uuid.uuid4().hex[:8]}"
        manifest = buffer.lease(consumer_id)
        if manifest is None:
            break
        leased.append(manifest)
        try:
            template = template_from_manifest(manifest)
            groups.append(
                make_branch_group(
                    template,
                    manifest,
                    group_index=group_offset,
                    first_sample_index=group_offset * 8,
                )
            )
        except Exception:
            buffer.release(manifest.snapshot_id)
            raise

    if not groups:
        raise ValueError(f"no available retro manifests in {manifest_path}")
    logger.info("retro survey rollout %d: generating %d snapshot groups", rollout_id, len(groups))

    tasks = [
        asyncio.create_task(
            generate_and_rm_group(
                args,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
        )
        for group in groups
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    completed = []
    for manifest, result in zip(leased, results, strict=True):
        if isinstance(result, BaseException):
            buffer.release(manifest.snapshot_id)
            logger.error("retro survey snapshot %s: %s", manifest.snapshot_id, result)
            continue
        if not isinstance(result, list) or len(result) != 8:
            buffer.release(manifest.snapshot_id)
            logger.error(
                "retro survey snapshot %s returned %s siblings",
                manifest.snapshot_id,
                len(result) if isinstance(result, list) else type(result).__name__,
            )
            continue
        versions = {
            str(version)
            for sample in result
            for version in getattr(sample, "weight_versions", ())
            if version is not None
        }
        if len(versions) > 1:
            buffer.release(manifest.snapshot_id)
            logger.warning(
                "retro survey snapshot %s crossed behavior versions %s; dropping group",
                manifest.snapshot_id,
                sorted(versions),
            )
            continue
        buffer.consume(manifest.snapshot_id, rollout_id)
        completed.append(result)

    if not completed:
        raise RuntimeError("retro survey produced no valid eight-sibling groups")
    return RolloutFnTrainOutput(
        samples=completed,
        metrics={
            "retro/survey/requested_groups": len(groups),
            "retro/survey/completed_groups": len(completed),
        },
    )


def generate_retro_survey(args, rollout_id, _data_buffer, evaluation: bool = False):
    if evaluation:
        raise ValueError("retro survey rollout does not support evaluation mode")
    return run(_generate_retro_survey(args, rollout_id))


async def _generate_retro_groups(
    args,
    *,
    rollout_id: int,
    target: int,
    buffer: RetroBuffer,
    delete_consumed: bool,
    prefetch_worker=None,
) -> tuple[list[list[Sample]], dict[str, int | float]]:
    state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None)
        else None
    )
    completed: list[list[Sample]] = []
    max_behavior_lag = _retro_max_behavior_lag(args)
    # With a bound of <=1 (or none) keep the legacy invariant that a retro
    # group is single-behavior-version; a looser bound admits mixed-version
    # continuations and relies on the lag gate alone.
    allow_mixed_versions = max_behavior_lag is not None and max_behavior_lag > 1
    stats: dict[str, int | float] = {
        "requested": target,
        "generated": 0,
        "accepted": 0,
        "dropped": 0,
        "failed": 0,
        "lag_rejected": 0,
        "version_crossed_admitted": 0,
        "accepted_promising": 0,
        "accepted_recovery": 0,
    }
    promising_ratio = min(1.0, max(0.0, _env_float("ASYNC_RL_RETRO_POOL_PROMISING_RATIO", 0.5)))
    promising_target = min(target, int(math.floor(target * promising_ratio + 0.5)))
    event_targets = {
        "promising": promising_target,
        "recovery": target - promising_target,
    }
    accepted_by_event = {"promising": 0, "recovery": 0}
    pool_order = os.environ.get("ASYNC_RL_RETRO_POOL_ORDER", "newest").strip().lower()
    min_policy_age = _env_int("ASYNC_RL_RETRO_MIN_POLICY_AGE", 0)
    max_policy_age = _env_int("ASYNC_RL_RETRO_MAX_POLICY_AGE", 4)
    current_update = rollout_id + 1
    snapshot_policy_ages: list[int] = []
    synthetic_base = 1_000_000_000 + rollout_id * 1_000_000
    max_attempts = max(target, target * _env_int("ASYNC_RL_RETRO_MAX_ATTEMPTS", 3))
    next_group_number = 0

    # Prefetch lane: take whatever the background worker already generated
    # before falling back to made-to-order generation for the remainder. These
    # groups were produced under an earlier policy version, so their behavior
    # lag is genuinely > 0 — that is the point of the buffer.
    if prefetch_worker is not None and target > 0:
        drained, drain_counters = drain_ready_groups(
            prefetch_worker,
            args=args,
            rollout_id=rollout_id,
            target=target,
            event_targets=event_targets,
            max_behavior_lag=max_behavior_lag,
        )
        stats["prefetch_lag_rejected"] = drain_counters["lag_rejected"]
        stats["prefetch_quota_deferred"] = drain_counters["quota_deferred"]
        stats["prefetch_aborted"] = drain_counters["aborted"]
        stats["lag_rejected"] += drain_counters["lag_rejected"]
        for manifest, group in drained:
            policy_age = manifest.policy_age(current_update)
            if policy_age is not None:
                snapshot_policy_ages.append(policy_age)
            filter_output = call_dynamic_filter(dynamic_filter, args, group)
            buffer.consume(manifest.snapshot_id, rollout_id)
            if delete_consumed:
                try:
                    await asyncio.to_thread(delete_snapshot, manifest.snapshot_id)
                except RuntimeError as exc:
                    logger.warning("retro snapshot cleanup %s: %s", manifest.snapshot_id, exc)
            stats["generated"] += 1
            if not filter_output.keep:
                stats["dropped"] += 1
                continue
            completed.append(group)
            stats["accepted"] += 1
            accepted_by_event[manifest.event_type] += 1
            stats[f"accepted_{manifest.event_type}"] += 1
        stats["prefetch_taken"] = len(drained)
        stats["prefetch_queue_left"] = prefetch_worker.queue_size()

    while len(completed) < target and stats["generated"] < max_attempts:
        desired_types = [
            event_type
            for event_type in ("promising", "recovery")
            for _ in range(event_targets[event_type] - accepted_by_event[event_type])
        ]
        desired_types = desired_types[: max_attempts - int(stats["generated"])]
        leased = []
        groups = []
        for offset, event_type in enumerate(desired_types):
            consumer_id = f"train-{rollout_id}-{len(completed)}-{offset}-{uuid.uuid4().hex[:8]}"
            manifest = buffer.lease(
                consumer_id,
                current_update=current_update,
                min_policy_age=min_policy_age,
                max_policy_age=max_policy_age,
                event_type=event_type,
                order=pool_order,
            )
            if manifest is None:
                continue
            leased.append(manifest)
            policy_age = manifest.policy_age(current_update)
            if policy_age is not None:
                snapshot_policy_ages.append(policy_age)
            try:
                # IDs must be unique across *attempts*, not merely accepted
                # groups. If an earlier group is dropped, len(completed) can
                # reuse an accepted sibling group's IDs on the refill pass.
                group_number = next_group_number
                next_group_number += 1
                groups.append(
                    make_branch_group(
                        template_from_manifest(manifest),
                        manifest,
                        group_index=synthetic_base // 8 + group_number,
                        first_sample_index=synthetic_base + group_number * 8,
                    )
                )
            except Exception:
                buffer.release(manifest.snapshot_id)
                raise
        if not groups:
            break

        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    generate_and_rm_group(
                        args,
                        group,
                        sampling_params=state.sampling_params.copy(),
                        evaluation=False,
                    )
                )
                for group in groups
            ],
            return_exceptions=True,
        )
        stats["generated"] += len(results)

        for manifest, result in zip(leased, results, strict=True):
            if isinstance(result, BaseException):
                buffer.release(manifest.snapshot_id)
                stats["failed"] += 1
                logger.error("retro mixed snapshot %s: %s", manifest.snapshot_id, result)
                continue
            if (
                not isinstance(result, list)
                or len(result) != 8
                or any(getattr(sample, "status", None) == Sample.Status.ABORTED for sample in result)
            ):
                buffer.release(manifest.snapshot_id)
                stats["failed"] += 1
                continue
            versions = {
                str(version)
                for sample in result
                for version in getattr(sample, "weight_versions", ())
                if version is not None
            }
            if len(versions) > 1 and not allow_mixed_versions:
                buffer.release(manifest.snapshot_id)
                stats["failed"] += 1
                logger.warning(
                    "retro mixed snapshot %s crossed behavior versions %s",
                    manifest.snapshot_id,
                    sorted(versions),
                )
                continue
            if len(versions) > 1:
                stats["version_crossed_admitted"] += 1
            if max_behavior_lag is not None:
                lag = behavior_lag(result, rollout_id)
                if lag is not None and lag > max_behavior_lag:
                    # Continuation too stale to train on. The snapshot itself is
                    # still valid — release it so a refill pass can retry it.
                    buffer.release(manifest.snapshot_id)
                    stats["lag_rejected"] += 1
                    logger.warning(
                        "retro mixed snapshot %s continuation behavior lag %d > %d; releasing",
                        manifest.snapshot_id,
                        lag,
                        max_behavior_lag,
                    )
                    continue

            filter_output = call_dynamic_filter(dynamic_filter, args, result)
            buffer.consume(manifest.snapshot_id, rollout_id)
            if delete_consumed:
                try:
                    await asyncio.to_thread(delete_snapshot, manifest.snapshot_id)
                except RuntimeError as exc:
                    logger.warning("retro snapshot cleanup %s: %s", manifest.snapshot_id, exc)
            if not filter_output.keep:
                stats["dropped"] += 1
                continue
            completed.append(result)
            stats["accepted"] += 1
            accepted_by_event[manifest.event_type] += 1
            stats[f"accepted_{manifest.event_type}"] += 1
            if len(completed) >= target:
                break
    if snapshot_policy_ages:
        stats["snapshot_policy_age_mean"] = sum(snapshot_policy_ages) / len(snapshot_policy_ages)
        stats["snapshot_policy_age_max"] = max(snapshot_policy_ages)
    return completed, stats


def _candidate_manifests(group) -> list[RetroSnapshotManifest]:
    manifests: dict[str, RetroSnapshotManifest] = {}

    def visit(value) -> None:
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


async def _transition_group_candidates(
    group,
    *,
    store: ManifestStore,
    activate: bool,
) -> None:
    for manifest in _candidate_manifests(group):
        if manifest.status != SnapshotStatus.TENTATIVE:
            continue
        if activate:
            manifest.activate()
            store.append_transition(manifest)
            continue
        manifest.invalidate()
        store.append_transition(manifest)
        try:
            await asyncio.to_thread(delete_snapshot, manifest.snapshot_id)
        except RuntimeError as exc:
            logger.warning("retro tentative snapshot cleanup %s: %s", manifest.snapshot_id, exc)


def retro_group_split(total: int, ratio: float) -> tuple[int, int]:
    """(retro_target, fresh_target) for a batch of ``total`` groups.

    ratio 0 disables the retro lane entirely (fresh gets the whole batch),
    which is the retro-off control arm: the mixed path then reduces to the
    stock fully-async rollout plus in-episode snapshot capture.
    """
    ratio = min(1.0, max(0.0, ratio))
    retro_target = min(total - 1, round(total * ratio)) if total > 1 else 0
    return retro_target, total - retro_target


async def _generate_retro_mixed(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    rollout_started = time.perf_counter()
    total = int(args.rollout_batch_size)
    ratio = _env_float("ASYNC_RL_RETRO_GROUP_RATIO", 0.25)
    retro_target, fresh_target = retro_group_split(total, ratio)
    manifest_path = os.environ.get("ASYNC_RL_RETRO_MANIFEST_PATH")
    if not manifest_path:
        raise ValueError("ASYNC_RL_RETRO_MANIFEST_PATH is required for mixed retro rollout")
    store = ManifestStore(manifest_path)

    async def accept_primary(group) -> None:
        await _transition_group_candidates(group, store=store, activate=True)

    async def reject_candidate_group(group) -> None:
        await _transition_group_candidates(group, store=store, activate=False)

    fresh_args = copy.copy(args)
    fresh_args.rollout_batch_size = fresh_target
    if getattr(fresh_args, "rollout_prefetch_batches", None) is None:
        # Capacity guard: without an explicit prefetch the worker pool defaults
        # to the engine cap (hundreds of groups), which overwhelms the router.
        # One batch matches this path's historical behavior; raise it via
        # --rollout-prefetch-batches once --rollout-max-behavior-lag is set.
        fresh_args.rollout_prefetch_batches = 1
    fresh_args.rollout_max_staleness = None  # superseded by rollout_prefetch_batches

    # Old-P50-compatibility switch: run the legs in the pre-2026-08-18
    # sequential order (fresh completes, then retro), with the RetroBuffer
    # built AFTER the fresh leg so this step's own activated captures are
    # leasable (snapshot policy age >= 0). The concurrent schedule can never
    # offer age-0 snapshots — they don't exist yet when its retro leg runs —
    # so this switch is the only way to reproduce the old retro data
    # distribution for attribution runs. Costs the old straggler barrier.
    sequential_legs = os.environ.get("ASYNC_RL_RETRO_SEQUENTIAL_LEGS", "0") == "1"

    # With prefetch enabled the worker owns the only long-lived RetroBuffer —
    # constructing another here would release its in-flight leases. (Depth > 0
    # with sequential legs is contradictory — the worker pre-generates across
    # steps, so buffer timing no longer decides snapshot age — but harmless:
    # the worker's buffer wins.)
    prefetch_worker = get_prefetch_worker(args, manifest_path=manifest_path)

    def _make_buffer() -> RetroBuffer:
        return RetroBuffer(
            max_items=max(1024, total * 16),
            store=store,
        )

    buffer = None
    if prefetch_worker is not None:
        buffer = prefetch_worker.buffer
    elif not sequential_legs:
        # The buffer view is taken here, before this step's fresh captures are
        # activated, so the retro lane can only lease snapshots from earlier
        # updates: effective snapshot policy age is >= 1 (was >= 0 when the legs
        # ran sequentially and the buffer was built after the fresh leg).
        buffer = _make_buffer()

    async def _timed(coro):
        t0 = time.perf_counter()
        result = await coro
        return result, time.perf_counter() - t0

    if sequential_legs:
        fresh_output, fresh_seconds = await _timed(
            _generate_rollout_async(
                fresh_args,
                rollout_id,
                data_buffer,
                group_accept_hook=accept_primary,
                group_reject_hook=reject_candidate_group,
            )
        )
        if buffer is None:
            buffer = _make_buffer()
        (retro_groups, retro_stats), retro_seconds = await _timed(
            _generate_retro_groups(
                args,
                rollout_id=rollout_id,
                target=retro_target,
                buffer=buffer,
                delete_consumed=True,
                prefetch_worker=prefetch_worker,
            )
        )
    else:
        # Fresh and retro legs run concurrently on the shared engine fleet; the
        # sequential fresh-then-retro schedule left the engines mostly idle and
        # made the retro leg pure added wall time.
        (fresh_output, fresh_seconds), ((retro_groups, retro_stats), retro_seconds) = await asyncio.gather(
            _timed(
                _generate_rollout_async(
                    fresh_args,
                    rollout_id,
                    data_buffer,
                    group_accept_hook=accept_primary,
                    group_reject_hook=reject_candidate_group,
                )
            ),
            _timed(
                _generate_retro_groups(
                    args,
                    rollout_id=rollout_id,
                    target=retro_target,
                    buffer=buffer,
                    delete_consumed=True,
                    prefetch_worker=prefetch_worker,
                )
            ),
        )
    fresh_groups = list(fresh_output.samples)

    missing = total - len(fresh_groups) - len(retro_groups)
    fallback_groups: list[list[Sample]] = []
    fallback_metrics: dict = {}
    fallback_seconds = 0.0
    if missing > 0:
        fallback_args = copy.copy(args)
        fallback_args.rollout_batch_size = missing
        if getattr(fallback_args, "rollout_prefetch_batches", None) is None:
            fallback_args.rollout_prefetch_batches = 1
        fallback_args.rollout_max_staleness = None  # superseded by rollout_prefetch_batches
        fallback_started = time.perf_counter()
        fallback_output = await _generate_rollout_async(
            fallback_args,
            rollout_id,
            data_buffer,
            group_accept_hook=reject_candidate_group,
            group_reject_hook=reject_candidate_group,
        )
        fallback_seconds = time.perf_counter() - fallback_started
        fallback_groups = list(fallback_output.samples)
        fallback_metrics = dict(fallback_output.metrics)

    groups = fresh_groups + retro_groups + fallback_groups
    if len(groups) != total:
        raise RuntimeError(f"retro mixed rollout produced {len(groups)} groups, expected {total}")
    metrics = {
        **fresh_output.metrics,
        **{f"retro/fallback/{key}": value for key, value in fallback_metrics.items()},
        "retro/mix/target_groups": retro_target,
        "retro/mix/accepted_groups": len(retro_groups),
        "retro/mix/fresh_groups": len(fresh_groups) + len(fallback_groups),
        "retro/mix/buffer_available": buffer.available(
            current_update=rollout_id + 1,
            min_policy_age=_env_int("ASYNC_RL_RETRO_MIN_POLICY_AGE", 0),
            max_policy_age=_env_int("ASYNC_RL_RETRO_MAX_POLICY_AGE", 4),
        ),
        "retro/mix/buffer_promising": buffer.available(
            current_update=rollout_id + 1,
            min_policy_age=_env_int("ASYNC_RL_RETRO_MIN_POLICY_AGE", 0),
            max_policy_age=_env_int("ASYNC_RL_RETRO_MAX_POLICY_AGE", 4),
            event_type="promising",
        ),
        "retro/mix/buffer_recovery": buffer.available(
            current_update=rollout_id + 1,
            min_policy_age=_env_int("ASYNC_RL_RETRO_MIN_POLICY_AGE", 0),
            max_policy_age=_env_int("ASYNC_RL_RETRO_MAX_POLICY_AGE", 4),
            event_type="recovery",
        ),
        "retro/staleness/fresh_prefetch_batches": fresh_args.rollout_prefetch_batches,
        "retro/staleness/fresh_max_behavior_lag": (
            -1
            if getattr(args, "rollout_max_behavior_lag", None) is None
            else int(args.rollout_max_behavior_lag)
        ),
        "retro/staleness/retro_max_behavior_lag": (
            -1 if _retro_max_behavior_lag(args) is None else int(_retro_max_behavior_lag(args))
        ),
        # 0 = made-to-order retro lane (effectively synchronous, lag <= 1);
        # > 0 = groups pre-generated and aged across updates, so realized retro
        # behavior lag can actually reach the bound above.
        "retro/staleness/retro_prefetch_groups": retro_prefetch_groups(args),
        "retro/staleness/snapshot_min_policy_age_updates": _env_int("ASYNC_RL_RETRO_MIN_POLICY_AGE", 0),
        "retro/staleness/snapshot_max_policy_age_updates": _env_int("ASYNC_RL_RETRO_MAX_POLICY_AGE", 4),
        "retro/staleness/sequential_legs": int(sequential_legs),
        "retro/mix/pool_promising_ratio": _env_float("ASYNC_RL_RETRO_POOL_PROMISING_RATIO", 0.5),
        "retro/timing/fresh_seconds": fresh_seconds,
        "retro/timing/retro_seconds": retro_seconds,
        "retro/timing/fallback_seconds": fallback_seconds,
        "retro/timing/total_seconds": time.perf_counter() - rollout_started,
        **{f"retro/mix/{key}": value for key, value in retro_stats.items()},
    }
    logger.info(
        "retro mixed rollout %d: fresh=%d retro=%d fallback=%d target=%d",
        rollout_id,
        len(fresh_groups),
        len(retro_groups),
        len(fallback_groups),
        total,
    )
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)


def generate_retro_mixed(args, rollout_id, data_buffer, evaluation: bool = False):
    if evaluation:
        raise ValueError("retro mixed rollout does not support evaluation mode")
    return run(_generate_retro_mixed(args, rollout_id, data_buffer))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _retro_max_behavior_lag(args) -> int | None:
    """Retro-lane behavior-lag bound: ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG when set,
    else the fresh lane's --rollout-max-behavior-lag, else None (unbounded)."""
    raw = os.environ.get("ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG", "").strip()
    if raw:
        return int(raw)
    return getattr(args, "rollout_max_behavior_lag", None)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default
