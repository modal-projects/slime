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
from slime.rollout.fully_async_rollout import _generate_rollout_async
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

from .buffer import ManifestStore, RetroBuffer
from .group import make_branch_group, template_from_manifest
from .manifest import RetroSnapshotManifest, SnapshotStatus
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
) -> tuple[list[list[Sample]], dict[str, int | float]]:
    state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None)
        else None
    )
    completed: list[list[Sample]] = []
    stats: dict[str, int | float] = {
        "requested": target,
        "generated": 0,
        "accepted": 0,
        "dropped": 0,
        "failed": 0,
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
            if len(versions) > 1:
                buffer.release(manifest.snapshot_id)
                stats["failed"] += 1
                logger.warning(
                    "retro mixed snapshot %s crossed behavior versions %s",
                    manifest.snapshot_id,
                    sorted(versions),
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


async def _generate_retro_mixed(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    rollout_started = time.perf_counter()
    total = int(args.rollout_batch_size)
    ratio = min(1.0, max(0.0, _env_float("ASYNC_RL_RETRO_GROUP_RATIO", 0.25)))
    retro_target = min(total - 1, round(total * ratio)) if total > 1 else 0
    fresh_target = total - retro_target
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
    # The mixed path spends a second phase generating retro groups. A 4×
    # staleness window would let the background fresh worker pre-generate
    # 96 groups (768 episodes) during that phase, overwhelming the router and
    # causing 600s request timeouts. Keep one fresh step warm instead.
    fresh_args.rollout_max_staleness = 1
    fresh_started = time.perf_counter()
    fresh_output = await _generate_rollout_async(
        fresh_args,
        rollout_id,
        data_buffer,
        group_accept_hook=accept_primary,
        group_reject_hook=reject_candidate_group,
    )
    fresh_seconds = time.perf_counter() - fresh_started
    fresh_groups = list(fresh_output.samples)

    buffer = RetroBuffer(
        max_items=max(1024, total * 16),
        store=store,
    )
    retro_started = time.perf_counter()
    retro_groups, retro_stats = await _generate_retro_groups(
        args,
        rollout_id=rollout_id,
        target=retro_target,
        buffer=buffer,
        delete_consumed=True,
    )
    retro_seconds = time.perf_counter() - retro_started

    missing = total - len(fresh_groups) - len(retro_groups)
    fallback_groups: list[list[Sample]] = []
    fallback_metrics: dict = {}
    fallback_seconds = 0.0
    if missing > 0:
        fallback_args = copy.copy(args)
        fallback_args.rollout_batch_size = missing
        fallback_args.rollout_max_staleness = 1
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
        "retro/staleness/fresh_pool_max_updates": fresh_args.rollout_max_staleness,
        "retro/staleness/snapshot_min_policy_age_updates": _env_int("ASYNC_RL_RETRO_MIN_POLICY_AGE", 0),
        "retro/staleness/snapshot_max_policy_age_updates": _env_int("ASYNC_RL_RETRO_MAX_POLICY_AGE", 4),
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default
