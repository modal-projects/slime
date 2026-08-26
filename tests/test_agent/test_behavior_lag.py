"""Behavior-lag enforcement and the prefetch/staleness split.

``rollout_max_staleness`` only ever sized the fully-async worker's in-flight
pool, so it bounded behavior-policy lag on average (Little's law) but never per
sample: on the retro P50 lineage W&B showed fresh version lag up to 3 and four
behavior versions in one batch despite a "staleness 1" fresh pool. These tests
pin the replacement semantics:

  * ``rollout_prefetch_batches`` is the capacity knob (``rollout_max_staleness``
    stays as a deprecated alias, ignored when prefetch is set);
  * ``rollout_max_behavior_lag`` is a HARD per-group bound enforced at batch
    assembly against the trainer's publishing version (rollout t generates
    under absolute weight version t + 1) — violating groups are discarded and
    their prompts requeued for regeneration;
  * the retro mixed rollout runs its fresh and retro legs concurrently (the
    sequential schedule made the retro leg pure added wall time on a
    near-idle engine fleet).

Importing the rollout modules drags in the slime training stack; on a CPU env
we stub exactly the modules that aren't installed, as in test_no_abort_leak.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_with_stubs(modname: str):
    """Import ``modname``, stubbing each *missing* dependency (permissively) and
    retrying. Only modules that fail to import get stubbed, so a full GPU/CI env
    uses the real ones untouched."""
    for _ in range(60):
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            missing = exc.name
            if not missing or missing in sys.modules:
                raise
            stub = types.ModuleType(missing)
            stub.__path__ = []  # treat as a package so submodule imports resolve
            stub.__getattr__ = lambda _name: MagicMock()  # noqa: B023 - permissive attrs
            sys.modules[missing] = stub
    return importlib.import_module(modname)


try:
    far = _import_with_stubs("slime.rollout.fully_async_rollout")
except Exception as exc:  # pragma: no cover - unsatisfiable import env
    pytest.skip(f"fully_async_rollout unimportable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Version parsing and lag arithmetic
# ---------------------------------------------------------------------------


def _group(*version_lists):
    return [SimpleNamespace(weight_versions=list(vs)) for vs in version_lists]


def test_group_min_weight_version_parses_engine_formats():
    group = _group(["81", "82"], ["weight_v000080"])
    assert far.group_min_weight_version(group) == 80
    assert far.group_min_weight_version(_group([], [])) is None
    assert far.group_min_weight_version(_group(["nope"])) is None


def test_behavior_lag_is_vs_trainer_publishing_version():
    # Rollout t generates under absolute version t+1: an on-policy group at
    # rollout 80 carries version 81 -> lag 0. A version-79 straggler -> lag 2.
    assert far.behavior_lag(_group(["81"]), rollout_id=80) == 0
    assert far.behavior_lag(_group(["81", "80"], ["81"]), rollout_id=80) == 1
    assert far.behavior_lag(_group(["79"]), rollout_id=80) == 2
    assert far.behavior_lag(_group([]), rollout_id=80) is None


# ---------------------------------------------------------------------------
# _pool_size: prefetch is capacity; staleness is a deprecated alias
# ---------------------------------------------------------------------------


def _pool_args(**kw):
    base = dict(
        sglang_server_concurrency=64,
        rollout_num_gpus=32,
        rollout_num_gpus_per_engine=2,  # -> 16 engines, cap 1024
        rollout_batch_size=24,
        rollout_prefetch_batches=None,
        rollout_max_staleness=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_pool_size_prefers_prefetch_over_staleness_alias():
    assert far._pool_size(_pool_args(rollout_prefetch_batches=2, rollout_max_staleness=4)) == 48


def test_pool_size_falls_back_to_staleness_then_engine_cap():
    assert far._pool_size(_pool_args(rollout_max_staleness=4)) == 96
    assert far._pool_size(_pool_args()) == 1024


# ---------------------------------------------------------------------------
# Batch-assembly lag gate: reject, requeue prompt, admit the rest
# ---------------------------------------------------------------------------


class _FakeWorker:
    def __init__(self, batches):
        self._batches = list(batches)

    def get_completed_groups(self):
        return self._batches.pop(0) if self._batches else []

    def queue_size(self):
        return 0


class _FakeBuffer:
    def __init__(self):
        self.requeued = []

    def add_samples(self, groups):
        self.requeued.extend(groups)


def _traj_sample(version, index=0, reward=0.5):
    return SimpleNamespace(
        weight_versions=[str(version)],
        index=index,
        get_reward_value=lambda args, _r=reward: _r,
        status=None,
        response="stale response",
        response_length=3,
        tokens=[1, 2, 3],
        reward=reward,
        loss_mask=[1, 1, 1],
        rollout_log_probs=[0.0],
        remove_sample=False,
        spec_info=None,
        metadata={"agentic": {"turns": 5}, "instance_id": "x"},
    )


def test_collection_rejects_stale_group_and_requeues_prompt(monkeypatch):
    rollout_id = 80  # trainer publishing version 81
    stale = [_traj_sample(79, index=0)]  # lag 2
    fresh = [_traj_sample(81, index=1)]  # lag 0
    worker = _FakeWorker([[(0, stale), (1, fresh)]])
    monkeypatch.setattr(far, "_get_global_worker", lambda args, buf: worker)

    rejected = []
    buffer = _FakeBuffer()
    args = SimpleNamespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=1,
        rollout_max_behavior_lag=1,
    )
    out = asyncio.run(
        far._generate_rollout_async(
            args,
            rollout_id,
            buffer,
            group_reject_hook=rejected.append,
        )
    )

    assert out.samples == [fresh]
    assert buffer.requeued == [stale]  # prompt goes back for regeneration
    assert rejected == [stale]  # side effects (e.g. retro candidates) invalidated
    assert out.metrics["behavior_lag/rejected_groups"] == 1
    assert out.metrics["behavior_lag/max_rejected_lag"] == 2
    # The requeued group must be reset to PENDING with generation state
    # cleared — generate_and_rm short-circuits COMPLETED samples, so an
    # un-reset requeue spins through buffer -> worker -> reject forever.
    s = stale[0]
    assert s.status == far.Sample.Status.PENDING
    assert s.response == "" and s.response_length == 0 and s.tokens == []
    assert s.weight_versions == [] and s.reward is None and s.loss_mask is None
    assert "agentic" not in s.metadata and s.metadata["instance_id"] == "x"


def test_collection_admits_unversioned_groups_ungated(monkeypatch):
    # A group with no recorded weight versions cannot be judged; it must pass.
    unversioned = [SimpleNamespace(weight_versions=[], index=0, get_reward_value=lambda args: 0.0, metadata={})]
    worker = _FakeWorker([[(0, unversioned)]])
    monkeypatch.setattr(far, "_get_global_worker", lambda args, buf: worker)

    args = SimpleNamespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=1,
        rollout_max_behavior_lag=1,
    )
    out = asyncio.run(far._generate_rollout_async(args, 80, _FakeBuffer()))
    assert out.samples == [unversioned]
    assert out.metrics["behavior_lag/rejected_groups"] == 0


# ---------------------------------------------------------------------------
# Retro mixed rollout: fresh and retro legs overlap in time
# ---------------------------------------------------------------------------


def test_retro_mixed_runs_fresh_and_retro_concurrently(monkeypatch, tmp_path):
    retro_rollout = _import_with_stubs("agentic_rl.retro.rollout")
    monkeypatch.setenv("ASYNC_RL_RETRO_MANIFEST_PATH", str(tmp_path / "manifests.jsonl"))

    intervals = {}

    def _make_leg(name, result, duration=0.05):
        async def _leg(*a, **kw):
            start = time.monotonic()
            await asyncio.sleep(duration)
            intervals[name] = (start, time.monotonic())
            return result

        return _leg

    fresh_groups = [[_traj_sample(81, index=i)] for i in range(3)]
    retro_groups = [[_traj_sample(81, index=3)]]
    fresh_output = SimpleNamespace(samples=fresh_groups, metrics={})
    monkeypatch.setattr(retro_rollout, "_generate_rollout_async", _make_leg("fresh", fresh_output))
    monkeypatch.setattr(retro_rollout, "_generate_retro_groups", _make_leg("retro", (retro_groups, {})))
    monkeypatch.setattr(retro_rollout, "RetroBuffer", lambda **kw: SimpleNamespace(available=lambda **k: 0))

    args = SimpleNamespace(
        rollout_batch_size=4,  # default 0.25 retro ratio -> 1 retro + 3 fresh
        rollout_prefetch_batches=None,
        rollout_max_staleness=4,
        rollout_max_behavior_lag=1,
    )
    out = asyncio.run(retro_rollout._generate_retro_mixed(args, 80, data_buffer=None))

    assert len(out.samples) == 4
    fresh_start, fresh_end = intervals["fresh"]
    retro_start, retro_end = intervals["retro"]
    assert fresh_start < retro_end and retro_start < fresh_end, "legs must overlap"
    assert out.metrics["retro/staleness/fresh_max_behavior_lag"] == 1
    assert out.metrics["retro/staleness/retro_max_behavior_lag"] == 1
    assert out.metrics["retro/staleness/fresh_prefetch_batches"] == 1  # capacity guard default
    assert out.metrics["retro/staleness/sequential_legs"] == 0


# ---------------------------------------------------------------------------
# Per-lane bound resolution: retro env overrides the fresh flag
# ---------------------------------------------------------------------------


def test_retro_lag_bound_env_overrides_fresh_flag(monkeypatch):
    retro_rollout = _import_with_stubs("agentic_rl.retro.rollout")
    args = SimpleNamespace(rollout_max_behavior_lag=4)

    monkeypatch.delenv("ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG", raising=False)
    assert retro_rollout._retro_max_behavior_lag(args) == 4  # falls back to fresh flag

    monkeypatch.setenv("ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG", "8")
    assert retro_rollout._retro_max_behavior_lag(args) == 8

    monkeypatch.setenv("ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG", "1")
    assert retro_rollout._retro_max_behavior_lag(args) == 1


# ---------------------------------------------------------------------------
# Retro prefetch buffer: depth 0 keeps the legacy lane; depth > 0 ages groups
# ---------------------------------------------------------------------------


def _manifest_stub(snapshot_id, event_type="promising", age=1):
    return SimpleNamespace(
        snapshot_id=snapshot_id,
        event_type=event_type,
        policy_age=lambda current_update, _a=age: _a,
    )


def test_retro_prefetch_depth_zero_disables_worker(monkeypatch):
    pf = _import_with_stubs("agentic_rl.retro.prefetch")
    monkeypatch.delenv("ASYNC_RL_RETRO_PREFETCH_BATCHES", raising=False)
    args = SimpleNamespace(rollout_batch_size=32)
    assert pf.retro_prefetch_groups(args) == 0
    assert pf.get_worker(args, manifest_path="/nonexistent/manifests.jsonl") is None


def test_retro_prefetch_depth_scales_with_group_ratio(monkeypatch):
    pf = _import_with_stubs("agentic_rl.retro.prefetch")
    monkeypatch.setenv("ASYNC_RL_RETRO_GROUP_RATIO", "0.25")
    args = SimpleNamespace(rollout_batch_size=32)  # 8 retro groups per batch
    monkeypatch.setenv("ASYNC_RL_RETRO_PREFETCH_BATCHES", "1")
    assert pf.retro_prefetch_groups(args) == 8
    monkeypatch.setenv("ASYNC_RL_RETRO_PREFETCH_BATCHES", "4")
    assert pf.retro_prefetch_groups(args) == 32


class _FakeWorkerQ:
    """Minimal stand-in for RetroPrefetchWorker's queue surface."""

    def __init__(self, items):
        self._items = list(items)
        self.given_back = []
        self.released = []

    def drain(self, limit):
        out, self._items = self._items[:limit], self._items[limit:]
        return out

    def give_back(self, item):
        self.given_back.append(item)

    def release(self, manifest):
        self.released.append(manifest.snapshot_id)

    def queue_size(self):
        return len(self._items)


def test_drain_applies_lag_gate_quota_and_defers_extras():
    pf = _import_with_stubs("agentic_rl.retro.prefetch")
    rollout_id = 80  # trainer version 81
    fresh = (_manifest_stub("ok-1"), [_traj_sample(80)])        # lag 1 -> admitted
    stale = (_manifest_stub("stale-1"), [_traj_sample(75)])     # lag 6 -> rejected
    extra = (_manifest_stub("ok-2"), [_traj_sample(81)])        # over quota -> deferred
    worker = _FakeWorkerQ([fresh, stale, extra])

    taken, counters = pf.drain_ready_groups(
        worker,
        args=SimpleNamespace(),
        rollout_id=rollout_id,
        target=4,
        event_targets={"promising": 1, "recovery": 1},
        max_behavior_lag=4,
    )
    assert [m.snapshot_id for m, _ in taken] == ["ok-1"]
    assert counters["lag_rejected"] == 1 and worker.released == ["stale-1"]
    assert counters["quota_deferred"] == 1
    assert [m.snapshot_id for m, _ in worker.given_back] == ["ok-2"]


def test_drain_releases_aborted_groups():
    pf = _import_with_stubs("agentic_rl.retro.prefetch")
    s = _traj_sample(81)
    s.status = far.Sample.Status.ABORTED
    worker = _FakeWorkerQ([(_manifest_stub("dead"), [s])])
    taken, counters = pf.drain_ready_groups(
        worker, args=SimpleNamespace(), rollout_id=80,
        target=4, event_targets={"promising": 4, "recovery": 4}, max_behavior_lag=4,
    )
    assert taken == [] and counters["aborted"] == 1 and worker.released == ["dead"]


def test_concurrency_headroom_warns_when_pools_exceed_semaphore(caplog):
    pf = _import_with_stubs("agentic_rl.retro.prefetch")
    args = SimpleNamespace(
        n_samples_per_prompt=8, sglang_server_concurrency=64,
        rollout_num_gpus=32, rollout_num_gpus_per_engine=2,  # capacity 1024
        rollout_prefetch_batches=4, rollout_batch_size=24,   # fresh 96 groups = 768 eps
    )
    with caplog.at_level("WARNING"):
        pf.check_concurrency_headroom(args, retro_groups=32)  # +256 eps = 1024 -> at the edge
    assert "semaphore capacity" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        pf.check_concurrency_headroom(args, retro_groups=8)   # +64 eps = 832 -> fine
    assert "semaphore capacity" not in caplog.text


# ---------------------------------------------------------------------------
# Baseline parity: ratio 0 reduces the mixed path to fresh-only, and the
# sequential-legs switch reproduces the pre-2026-08-18 schedule (fresh leg,
# THEN buffer build — age-0 snapshots leasable — then retro leg).
# ---------------------------------------------------------------------------


def test_retro_group_split_arithmetic():
    retro_rollout = _import_with_stubs("agentic_rl.retro.rollout")
    assert retro_rollout.retro_group_split(32, 0.0) == (0, 32)
    assert retro_rollout.retro_group_split(32, 0.25) == (8, 24)
    assert retro_rollout.retro_group_split(32, 1.0) == (31, 1)  # fresh never empty
    assert retro_rollout.retro_group_split(1, 0.9) == (0, 1)
    assert retro_rollout.retro_group_split(4, -3.0) == (0, 4)  # clamped


def test_retro_mixed_ratio_zero_reduces_to_fresh_only(monkeypatch, tmp_path):
    retro_rollout = _import_with_stubs("agentic_rl.retro.rollout")
    monkeypatch.setenv("ASYNC_RL_RETRO_MANIFEST_PATH", str(tmp_path / "manifests.jsonl"))
    monkeypatch.setenv("ASYNC_RL_RETRO_GROUP_RATIO", "0")

    seen = {}

    fresh_groups = [[_traj_sample(81, index=i)] for i in range(4)]
    fresh_output = SimpleNamespace(samples=fresh_groups, metrics={})

    async def _fresh(fresh_args, rollout_id, data_buffer, **kw):
        seen["fresh_batch"] = fresh_args.rollout_batch_size
        seen["hooks"] = (kw.get("group_accept_hook"), kw.get("group_reject_hook"))
        return fresh_output

    async def _retro(args, **kw):
        seen["retro_target"] = kw["target"]
        return [], {}

    monkeypatch.setattr(retro_rollout, "_generate_rollout_async", _fresh)
    monkeypatch.setattr(retro_rollout, "_generate_retro_groups", _retro)
    monkeypatch.setattr(retro_rollout, "RetroBuffer", lambda **kw: SimpleNamespace(available=lambda **k: 0))

    args = SimpleNamespace(
        rollout_batch_size=4,
        rollout_prefetch_batches=4,
        rollout_max_staleness=None,
        rollout_max_behavior_lag=4,
    )
    out = asyncio.run(retro_rollout._generate_retro_mixed(args, 80, data_buffer=None))

    # The fresh leg owns the whole batch and keeps its capture hooks; the
    # retro leg is asked for exactly zero groups. That makes this arm the
    # vanilla stock path plus in-episode snapshot capture, nothing else.
    assert seen["fresh_batch"] == 4
    assert seen["retro_target"] == 0
    assert all(hook is not None for hook in seen["hooks"])
    assert out.samples == fresh_groups
    assert out.metrics["retro/mix/target_groups"] == 0


def test_retro_mixed_sequential_legs_restores_old_schedule(monkeypatch, tmp_path):
    retro_rollout = _import_with_stubs("agentic_rl.retro.rollout")
    monkeypatch.setenv("ASYNC_RL_RETRO_MANIFEST_PATH", str(tmp_path / "manifests.jsonl"))
    monkeypatch.setenv("ASYNC_RL_RETRO_SEQUENTIAL_LEGS", "1")

    intervals = {}
    buffer_built_at = []

    def _make_leg(name, result, duration=0.05):
        async def _leg(*a, **kw):
            start = time.monotonic()
            await asyncio.sleep(duration)
            intervals[name] = (start, time.monotonic())
            return result

        return _leg

    fresh_groups = [[_traj_sample(81, index=i)] for i in range(3)]
    retro_groups = [[_traj_sample(81, index=3)]]
    fresh_output = SimpleNamespace(samples=fresh_groups, metrics={})
    monkeypatch.setattr(retro_rollout, "_generate_rollout_async", _make_leg("fresh", fresh_output))
    monkeypatch.setattr(retro_rollout, "_generate_retro_groups", _make_leg("retro", (retro_groups, {})))

    def _buffer(**kw):
        buffer_built_at.append(time.monotonic())
        return SimpleNamespace(available=lambda **k: 0)

    monkeypatch.setattr(retro_rollout, "RetroBuffer", _buffer)

    args = SimpleNamespace(
        rollout_batch_size=4,  # default 0.25 retro ratio -> 1 retro + 3 fresh
        rollout_prefetch_batches=1,
        rollout_max_staleness=None,
        rollout_max_behavior_lag=None,
    )
    out = asyncio.run(retro_rollout._generate_retro_mixed(args, 80, data_buffer=None))

    assert len(out.samples) == 4
    fresh_start, fresh_end = intervals["fresh"]
    retro_start, retro_end = intervals["retro"]
    assert fresh_end <= retro_start, "legs must not overlap in sequential mode"
    # The buffer view is taken after the fresh leg, so snapshots activated by
    # this step's own accepted groups (age 0) are leasable — the old-P50
    # retro data distribution.
    assert len(buffer_built_at) == 1
    assert buffer_built_at[0] >= fresh_end
    assert out.metrics["retro/staleness/sequential_legs"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
