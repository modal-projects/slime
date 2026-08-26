"""Frontier-CS environment with isolated snapshot capture/restore hooks."""

from __future__ import annotations

import os
import shlex
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agentic_rl.envs.base import EpisodeLimits, RewardResult
from agentic_rl.envs.frontier_cs.env import SUBMISSIONS_LOG, FrontierCsEnv

from .agent import SnapshottingAgent
from .buffer import ManifestStore
from .manifest import Compatibility, RetroSnapshotManifest, SnapshotKind, SnapshotStatus
from .model import ChainCheckpoint, capture_checkpoint, restore_agent, restore_recording_model
from .selector import BranchEvent, EventSelector, SelectionConfig, assign_event_type
from .snapshot import RestoreResult, SnapshotResult, restore_directory, snapshot_sandbox

RETRO_MANIFEST_KEY = "retro_manifest"
_STAGING_ROOT = "/tmp/retro_candidates"


@dataclass(frozen=True)
class _StagedCandidate:
    event: BranchEvent
    checkpoint: ChainCheckpoint
    workspace_path: str


@dataclass
class _EpisodeContext:
    md: dict[str, Any]
    sample_metadata: dict[str, Any]
    source: dict[str, Any]
    restore_manifest: RetroSnapshotManifest | None
    selector: EventSelector | None
    staged: list[_StagedCandidate] = field(default_factory=list)
    captures: list[RetroSnapshotManifest] = field(default_factory=list)
    snapshot_results: list[SnapshotResult] = field(default_factory=list)
    restore_result: RestoreResult | None = None


class RetroFrontierCsEnv(FrontierCsEnv):
    """Explicit opt-in environment selected by ``agentic_rl.retro.generate``."""

    name = "frontier_cs_retro"

    def __init__(self):
        self._local = threading.local()

    def normalize_metadata(self, sample) -> dict[str, Any]:
        md = super().normalize_metadata(sample)
        source_metadata = dict(sample.metadata or {})
        restore = source_metadata.get(RETRO_MANIFEST_KEY)
        md["_retro_manifest"] = RetroSnapshotManifest.from_dict(restore) if isinstance(restore, dict) else None
        md["_retro_sample_metadata"] = {
            key: value
            for key, value in source_metadata.items()
            if key not in (RETRO_MANIFEST_KEY, "retro_sibling", "agentic")
        }
        md["_retro_sample_metadata"]["_retro_prompt"] = sample.prompt
        md["_retro_sample_metadata"]["_retro_label"] = sample.label
        md["_retro_source"] = {
            "rollout_id": getattr(sample, "rollout_id", None),
            "group_index": getattr(sample, "group_index", None),
            "sample_index": getattr(sample, "index", None),
        }
        return md

    def rollout(self, md: dict[str, Any], *, model, limits: EpisodeLimits) -> RewardResult:
        restore_manifest = md.get("_retro_manifest")
        source = md.get("_retro_source") or {}
        selector = None if restore_manifest is not None else EventSelector(_selection_config(source))
        ctx = _EpisodeContext(
            md=md,
            sample_metadata=md.get("_retro_sample_metadata") or {},
            source=source,
            restore_manifest=restore_manifest,
            selector=selector,
        )
        self._local.context = ctx
        branch_limits = limits
        if restore_manifest is not None:
            restore_manifest.compatibility.assert_matches(_compatibility(md))
            branch_limits = replace(
                limits,
                max_steps=restore_manifest.remaining_steps,
                episode_timeout=restore_manifest.remaining_seconds,
            )
        try:
            result = super().rollout(md, model=model, limits=branch_limits)
            extra = dict(result.extra)
            if ctx.captures:
                extra["retro_candidates"] = [manifest.to_dict() for manifest in ctx.captures]
                extra["retro_snapshot"] = {
                    "count": len(ctx.captures),
                    "latency_seconds": sum(item.latency_seconds for item in ctx.snapshot_results),
                    "estimated_bytes": sum(item.estimated_bytes or 0 for item in ctx.snapshot_results),
                    "estimated_files": sum(item.estimated_files or 0 for item in ctx.snapshot_results),
                }
            if restore_manifest is not None:
                extra["retro_branch"] = {
                    "snapshot_id": restore_manifest.snapshot_id,
                    "event_type": restore_manifest.event_type,
                    "turn_index": restore_manifest.turn_index,
                    "inherited_score": restore_manifest.score,
                    "inherited_best": restore_manifest.best_score,
                    "source_weight_version": restore_manifest.source_weight_version,
                    "source_update": restore_manifest.source_update,
                    "target_fraction": restore_manifest.target_fraction,
                    "trajectory_fraction": restore_manifest.trajectory_fraction,
                    "fraction_error": restore_manifest.fraction_error,
                }
                if ctx.restore_result is not None:
                    extra["retro_restore"] = {
                        "latency_seconds": ctx.restore_result.latency_seconds,
                        "copied_to_writable": ctx.restore_result.copied_to_writable,
                    }
            return RewardResult(reward=result.reward, is_solved=result.is_solved, extra=extra)
        finally:
            del self._local.context

    def _sandbox(self, md: dict[str, Any], *, lifetime: int, exec_timeout: int):
        sandbox = super()._sandbox(md, lifetime=lifetime, exec_timeout=exec_timeout)
        ctx = self._context()
        manifest = ctx.restore_manifest
        if manifest is None:
            return sandbox
        if manifest.snapshot_kind != SnapshotKind.DIRECTORY:
            sandbox.terminate()
            raise RuntimeError(
                "retro filesystem restore is not enabled in the agent path; "
                "select directory snapshots after the Phase-0 smoke"
            )
        ctx.restore_result = restore_directory(
            sandbox,
            manifest.snapshot_id,
            target_path=manifest.snapshot_path,
        )
        return sandbox

    def _pre_agent_setup(self, sb, task_dir: Path, md: dict[str, Any]) -> None:
        super()._pre_agent_setup(sb, task_dir, md)
        if self._context().restore_manifest is not None:
            # Branches receive a new server-side AGENT_ID in FrontierCsEnv.rollout.
            # Clear the inherited sandbox-written log so local diagnostics also
            # contain branch-local submissions only.
            sb.exec(f"rm -f {SUBMISSIONS_LOG}", check=False, timeout=30)

    def run_agent_leg(self, model, sandbox, task: str, *, max_steps: int, wall_time_sec: int) -> dict:
        ctx = self._context()
        checkpoint = None
        step_limit = max_steps
        if ctx.restore_manifest is not None:
            checkpoint = ChainCheckpoint.from_dict(ctx.restore_manifest.agent_state["checkpoint"])
            restore_recording_model(model, checkpoint)
            step_limit = checkpoint.n_calls + max_steps

        started = time.monotonic()
        callback = None if checkpoint is not None else self._turn_callback(model, sandbox, wall_time_sec, started)
        agent = SnapshottingAgent(
            model,
            sandbox,
            system_template=_system_template(),
            instance_template=_instance_template(),
            step_limit=step_limit,
            cost_limit=0.0,
            wall_time_limit_seconds=wall_time_sec,
            turn_callback=callback,
        )
        sandbox.deadline = time.monotonic() + wall_time_sec
        try:
            try:
                if checkpoint is not None:
                    restore_agent(agent, checkpoint)
                    result = agent.resume(task=task) or {}
                else:
                    result = agent.run(task=task) or {}
            finally:
                sandbox.deadline = None
            if checkpoint is None and not getattr(model, "aborted", False):
                self._finalize_capture(
                    agent,
                    sandbox,
                    total_seconds=time.monotonic() - started,
                )
            return result
        finally:
            sandbox.deadline = None
            if checkpoint is None:
                self._cleanup_staging(sandbox)

    def _turn_callback(self, model, sandbox, wall_time_sec: int, started: float):
        def callback(agent, _message, _observations) -> None:
            ctx = self._context()
            if ctx.selector is None:
                return
            event = ctx.selector.observe_log(
                sandbox.read_file(SUBMISSIONS_LOG),
                turn_index=agent.n_calls,
                max_steps=agent.config.step_limit,
                elapsed_seconds=time.monotonic() - started,
                wall_time_seconds=wall_time_sec,
            )
            if event is not None:
                self._stage_candidate(agent, model, sandbox, event)

        return callback

    def _stage_candidate(self, agent, model, sandbox, event: BranchEvent) -> None:
        ctx = self._context()
        checkpoint = capture_checkpoint(agent, model)
        source_path = os.environ.get("ASYNC_RL_RETRO_SNAPSHOT_PATH", "/app")
        candidate_path = f"{_STAGING_ROOT}/{event.turn_index}-{event.submission_count}"
        q = shlex.quote
        sandbox.exec(
            f"rm -rf {q(candidate_path)} && mkdir -p {q(candidate_path)} "
            f"&& cp -a {q(source_path)}/. {q(candidate_path)}/",
            cwd="/",
            check=True,
            timeout=60,
        )
        ctx.staged.append(
            _StagedCandidate(
                event=event,
                checkpoint=checkpoint,
                workspace_path=candidate_path,
            )
        )

    def _finalize_capture(self, agent, sandbox, *, total_seconds: float) -> None:
        ctx = self._context()
        if ctx.selector is None or not ctx.staged:
            return
        selected = ctx.selector.select(
            [candidate.event for candidate in ctx.staged],
            total_turns=int(agent.n_calls),
            total_seconds=total_seconds,
        )
        if selected is None:
            return
        staged = next(
            candidate
            for candidate in ctx.staged
            if candidate.event.turn_index == selected.turn_index
            and candidate.event.submission_count == selected.submission_count
        )
        kind = SnapshotKind(os.environ.get("ASYNC_RL_RETRO_SNAPSHOT_KIND", SnapshotKind.DIRECTORY.value))
        if kind != SnapshotKind.DIRECTORY:
            raise ValueError("trajectory-percentile retro capture requires directory snapshots")
        ttl = _env_int("ASYNC_RL_RETRO_SNAPSHOT_TTL", 48 * 60 * 60)
        target_path = os.environ.get("ASYNC_RL_RETRO_SNAPSHOT_PATH", "/app")
        snapshot = snapshot_sandbox(
            sandbox,
            kind=kind,
            path=staged.workspace_path,
            ttl_seconds=ttl,
        )
        source_version = staged.checkpoint.source_weight_version
        capture_status = SnapshotStatus(
            os.environ.get("ASYNC_RL_RETRO_CAPTURE_STATUS", SnapshotStatus.AVAILABLE.value)
        )
        if capture_status not in (SnapshotStatus.AVAILABLE, SnapshotStatus.TENTATIVE):
            raise ValueError("retro capture status must be available or tentative")
        manifest = RetroSnapshotManifest.create(
            snapshot_id=snapshot.snapshot_id,
            snapshot_kind=kind,
            snapshot_path=target_path,
            ttl_seconds=ttl,
            task_type="frontier_cs",
            instance_id=str(ctx.md["instance_id"]),
            problem_id=str((ctx.md.get("verifier") or {}).get("env", {}).get("PROBLEM_ID") or ""),
            event_type=selected.event_type.value,
            turn_index=selected.turn_index,
            remaining_steps=selected.remaining_steps,
            remaining_seconds=selected.remaining_seconds,
            source_run_tag=os.environ.get("ASYNC_RL_RETRO_RUN_TAG", ""),
            source_rollout_id=ctx.source.get("rollout_id"),
            source_group_index=ctx.source.get("group_index"),
            source_sample_index=ctx.source.get("sample_index"),
            source_weight_version=source_version,
            source_update=_version_number(source_version),
            source_total_turns=selected.source_total_turns,
            source_total_seconds=selected.source_total_seconds,
            target_fraction=ctx.selector.config.target_fraction,
            trajectory_fraction=selected.trajectory_fraction,
            fraction_error=selected.fraction_error,
            score=selected.score,
            best_score=selected.best_score,
            agent_state={"checkpoint": staged.checkpoint.to_dict()},
            sample_metadata=ctx.sample_metadata,
            compatibility=_compatibility(ctx.md),
            status=capture_status,
        )
        ctx.captures.append(manifest)
        ctx.snapshot_results.append(snapshot)
        manifest_path = os.environ.get("ASYNC_RL_RETRO_MANIFEST_PATH")
        if manifest_path:
            ManifestStore(manifest_path).append(manifest)

    @staticmethod
    def _cleanup_staging(sandbox) -> None:
        sandbox.exec(f"rm -rf {shlex.quote(_STAGING_ROOT)}", cwd="/", check=False, timeout=60)

    def _context(self) -> _EpisodeContext:
        ctx = getattr(self._local, "context", None)
        if ctx is None:
            raise RuntimeError("retro episode context is unavailable")
        return ctx


def _selection_config(source: dict[str, Any]) -> SelectionConfig:
    preferred = assign_event_type(
        assignment=os.environ.get("ASYNC_RL_RETRO_SELECTOR_ASSIGNMENT", "hashed"),
        promising_ratio=_env_float("ASYNC_RL_RETRO_CAPTURE_PROMISING_RATIO", 0.5),
        seed=_env_int("ASYNC_RL_RETRO_SELECTOR_SEED", 20260802),
        group_index=source.get("group_index"),
        sample_index=source.get("sample_index"),
    )
    return SelectionConfig(
        min_score=_env_float("ASYNC_RL_RETRO_MIN_SCORE", 0.1),
        max_score=_env_float("ASYNC_RL_RETRO_MAX_SCORE", 0.95),
        min_remaining_fraction=_env_float("ASYNC_RL_RETRO_MIN_REMAINING", 0.0),
        min_turn=_env_int("ASYNC_RL_RETRO_MIN_TURN", 2),
        regression_delta=_env_float("ASYNC_RL_RETRO_REGRESSION_DELTA", 0.1),
        stagnant_submissions=_env_int("ASYNC_RL_RETRO_STAGNANT_SUBMISSIONS", 2),
        promising_consecutive=_env_int("ASYNC_RL_RETRO_PROMISING_CONSECUTIVE", 0),
        target_fraction=_env_float("ASYNC_RL_RETRO_TARGET_TRAJECTORY_FRACTION", 0.5),
        max_fraction_error=_env_float("ASYNC_RL_RETRO_MAX_FRACTION_ERROR", 0.4),
        preferred_event=preferred,
        allow_fallback=_env_bool("ASYNC_RL_RETRO_SELECTOR_FALLBACK", False),
    )


def _compatibility(md: dict[str, Any] | None = None) -> Compatibility:
    try:
        import modal

        modal_version = str(getattr(modal, "__version__", ""))
    except ImportError:
        modal_version = ""
    try:
        import minisweagent

        mini_version = str(getattr(minisweagent, "__version__", ""))
    except ImportError:
        mini_version = ""
    return Compatibility(
        base_image=(
            os.environ.get("ASYNC_RL_RETRO_BASE_IMAGE_FINGERPRINT")
            or str((md or {}).get("docker_image") or (md or {}).get("dockerfile") or "")
        ),
        task_data=(
            os.environ.get("ASYNC_RL_RETRO_DATA_FINGERPRINT")
            or str(((md or {}).get("verifier") or {}).get("env", {}).get("PROBLEM_ID") or "")
        ),
        code=os.environ.get("ASYNC_RL_RETRO_CODE_FINGERPRINT", ""),
        modal_version=modal_version,
        mini_swe_version=mini_version,
    )


def _system_template() -> str:
    from agentic_rl.core.prompts import SYSTEM_TEMPLATE

    return SYSTEM_TEMPLATE


def _instance_template() -> str:
    from agentic_rl.core.prompts import INSTANCE_TEMPLATE

    return INSTANCE_TEMPLATE


def _version_number(value: str) -> int | None:
    digits = "".join(character if character.isdigit() else " " for character in value).split()
    return int(digits[-1]) if digits else None


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
