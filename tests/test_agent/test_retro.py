from __future__ import annotations

import importlib
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentic_rl.retro.buffer import ManifestStore, RetroBuffer
from agentic_rl.retro.manifest import (
    Compatibility,
    RetroSnapshotManifest,
    SnapshotKind,
    SnapshotStatus,
)
from agentic_rl.retro.selector import (
    BranchEventType,
    EventSelector,
    SelectionConfig,
    assign_event_type,
)
from agentic_rl.retro.snapshot import sandbox_compute_cost, snapshot_sandbox
from agentic_rl.retro.survey import BranchOutcome, BranchSurveyGroup, summarize_groups


def _import_with_stubs(modname: str):
    for _ in range(60):
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            missing = exc.name
            if not missing or missing in sys.modules:
                raise
            stub = types.ModuleType(missing)
            stub.__path__ = []
            stub.__getattr__ = lambda _name: MagicMock()  # noqa: B023
            sys.modules[missing] = stub
    return importlib.import_module(modname)


def _manifest(snapshot_id="im-1", **overrides):
    values = dict(
        snapshot_id=snapshot_id,
        snapshot_kind=SnapshotKind.DIRECTORY,
        snapshot_path="/app",
        ttl_seconds=3600,
        task_type="frontier_cs",
        instance_id="p1",
        problem_id="1",
        event_type="promising",
        turn_index=3,
        remaining_steps=7,
        remaining_seconds=300,
        source_update=10,
        score=0.4,
        best_score=0.5,
        agent_state={"checkpoint": {"tokens": [1], "authorization": "secret"}},
        sample_metadata={"task_path": "x", "AGENT_ID": "secret"},
    )
    values.update(overrides)
    return RetroSnapshotManifest.create(**values)


def _submission_log(*scores):
    return "\n".join(
        json.dumps(
            {
                "submission_uuid": f"s{i}",
                "status": "done",
                "score": score,
                "score_raw": score * 100,
            }
        )
        for i, score in enumerate(scores)
    )


def test_manifest_roundtrip_sanitizes_secrets_and_checks_age():
    manifest = _manifest()
    assert "authorization" not in manifest.agent_state["checkpoint"]
    assert "AGENT_ID" not in manifest.sample_metadata
    restored = RetroSnapshotManifest.from_dict(manifest.to_dict())
    assert restored.snapshot_kind == SnapshotKind.DIRECTORY
    assert restored.policy_age(13) == 3
    assert restored.is_eligible(current_update=14, max_policy_age=4)
    assert not restored.is_eligible(current_update=15, max_policy_age=4)


def test_manifest_lease_consume_and_compatibility():
    manifest = _manifest()
    manifest.lease("worker-1")
    assert manifest.status == SnapshotStatus.LEASED
    manifest.consume(7)
    assert manifest.status == SnapshotStatus.CONSUMED
    assert manifest.consumed_by_rollout == 7

    Compatibility(code="a").assert_matches(Compatibility(code="a"))
    with pytest.raises(ValueError, match="code"):
        Compatibility(code="a").assert_matches(Compatibility(code="b"))


def test_event_selector_promising_and_recovery():
    promising = EventSelector(
        SelectionConfig(preferred_event=BranchEventType.PROMISING, allow_fallback=False)
    )
    event = promising.observe_log(
        _submission_log(0.4),
        turn_index=3,
        max_steps=10,
        elapsed_seconds=10,
        wall_time_seconds=100,
    )
    assert event is not None and event.event_type == BranchEventType.PROMISING
    assert event.remaining_steps == 7

    recovery = EventSelector(
        SelectionConfig(
            preferred_event=BranchEventType.RECOVERY,
            allow_fallback=False,
            regression_delta=0.1,
        )
    )
    assert (
        recovery.observe_log(
            _submission_log(0.6),
            turn_index=3,
            max_steps=10,
            elapsed_seconds=10,
            wall_time_seconds=100,
        )
        is None
    )
    event = recovery.observe_log(
        _submission_log(0.6, 0.3),
        turn_index=4,
        max_steps=10,
        elapsed_seconds=20,
        wall_time_seconds=100,
    )
    assert event is not None and event.event_type == BranchEventType.RECOVERY
    assert event.best_score == 0.6


def test_event_selector_keeps_candidate_nearest_realized_fraction():
    selector = EventSelector(
        SelectionConfig(
            preferred_event=BranchEventType.PROMISING,
            target_fraction=0.75,
            max_fraction_error=0.4,
        )
    )
    candidates = []
    for turn, elapsed, scores in (
        (8, 80, (0.2,)),
        (27, 270, (0.2, 0.4)),
        (31, 310, (0.2, 0.4, 0.6)),
    ):
        event = selector.observe_log(
            _submission_log(*scores),
            turn_index=turn,
            max_steps=75,
            elapsed_seconds=elapsed,
            wall_time_seconds=750,
        )
        assert event is not None
        candidates.append(event)

    selected = selector.select(candidates, total_turns=40, total_seconds=400)
    assert selected is not None
    assert selected.turn_index == 31
    assert selected.remaining_steps == 9
    assert selected.remaining_seconds == 90
    assert selected.trajectory_fraction == pytest.approx(0.775)
    assert selected.fraction_error == pytest.approx(0.025)

    strict = EventSelector(
        SelectionConfig(target_fraction=0.75, max_fraction_error=0.01)
    )
    assert strict.select(candidates, total_turns=40, total_seconds=400) is None


def test_selector_assignment_is_per_rollout_and_reproducible():
    values = [
        assign_event_type(
            assignment="hashed",
            promising_ratio=0.5,
            seed=7,
            group_index=3,
            sample_index=index,
        )
        for index in range(32)
    ]
    repeated = [
        assign_event_type(
            assignment="hashed",
            promising_ratio=0.5,
            seed=7,
            group_index=3,
            sample_index=index,
        )
        for index in range(32)
    ]
    assert values == repeated
    assert set(values) == {BranchEventType.PROMISING, BranchEventType.RECOVERY}
    assert (
        assign_event_type(
            assignment="alternating",
            promising_ratio=0.5,
            seed=7,
            group_index=3,
            sample_index=0,
        )
        == BranchEventType.PROMISING
    )
    assert (
        assign_event_type(
            assignment="alternating",
            promising_ratio=0.5,
            seed=7,
            group_index=3,
            sample_index=1,
        )
        == BranchEventType.RECOVERY
    )


def test_event_selector_rejects_invalid_scores_and_late_turns():
    selector = EventSelector(SelectionConfig(min_remaining_fraction=0.3))
    assert (
        selector.observe_log(
            _submission_log(92.0),
            turn_index=3,
            max_steps=10,
            elapsed_seconds=10,
            wall_time_seconds=100,
        )
        is None
    )
    assert (
        selector.observe_log(
            _submission_log(0.4),
            turn_index=9,
            max_steps=10,
            elapsed_seconds=95,
            wall_time_seconds=100,
        )
        is None
    )


class _FakeRawSandbox:
    def __init__(self):
        self.calls = []

    def snapshot_directory(self, path, timeout, ttl):
        self.calls.append(("directory", path, timeout, ttl))
        return SimpleNamespace(object_id="im-directory")

    def snapshot_filesystem(self, timeout, ttl):
        self.calls.append(("filesystem", timeout, ttl))
        return SimpleNamespace(object_id="im-filesystem")


class _FakeSandbox:
    def __init__(self):
        self.sb = _FakeRawSandbox()

    def exec(self, command, check=False):
        if command.startswith("du "):
            return 0, "12\n", ""
        if command.startswith("find "):
            return 0, "7\n", ""
        return 0, "", ""


def test_snapshot_adapter_records_id_latency_and_workspace_size():
    sandbox = _FakeSandbox()
    result = snapshot_sandbox(
        sandbox,
        kind=SnapshotKind.DIRECTORY,
        path="/app",
        ttl_seconds=60,
    )
    assert result.snapshot_id == "im-directory"
    assert result.estimated_bytes == 12 * 1024
    assert result.estimated_files == 7
    assert sandbox.sb.calls == [("directory", "/app", 55, 60)]
    assert sandbox_compute_cost(3600) == pytest.approx(0.663696)


def test_buffer_persists_latest_state_and_recovers_leases(tmp_path):
    store = ManifestStore(tmp_path / "manifests.jsonl")
    buffer = RetroBuffer(max_items=2, store=store)
    first, second, third = _manifest("im-1"), _manifest("im-2"), _manifest("im-3")
    assert buffer.add(first) == []
    assert buffer.add(second) == []
    evicted = buffer.add(third)
    assert [item.snapshot_id for item in evicted] == ["im-1"]
    leased = buffer.lease("worker")
    assert leased is not None

    recovered = RetroBuffer(max_items=2, store=store)
    assert recovered.available() == 2
    assert all(item.status == SnapshotStatus.AVAILABLE for item in recovered.manifests())


def test_tentative_manifest_requires_activation_and_newest_pool_order(tmp_path):
    tentative = _manifest("im-tentative", status=SnapshotStatus.TENTATIVE)
    assert not tentative.is_eligible()
    tentative.activate()
    assert tentative.is_eligible()

    store = ManifestStore(tmp_path / "manifests.jsonl")
    buffer = RetroBuffer(max_items=4, store=store)
    old = _manifest("im-old", source_update=3, event_type="promising")
    new = _manifest("im-new", source_update=5, event_type="promising")
    recovery = _manifest("im-recovery", source_update=6, event_type="recovery")
    buffer.add(old)
    buffer.add(new)
    buffer.add(recovery)

    leased = buffer.lease(
        "worker",
        current_update=6,
        min_policy_age=0,
        max_policy_age=4,
        event_type="promising",
        order="newest",
    )
    assert leased is not None and leased.snapshot_id == "im-new"


def test_branchability_metrics_require_eight_and_summarize():
    outcomes = tuple(
        BranchOutcome(
            reward=reward,
            best_reward=reward,
            output_tokens=100,
            sandbox_seconds=10,
            judge_calls=1,
            weight_versions=("v1",),
        )
        for reward in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0)
    )
    group = BranchSurveyGroup(
        snapshot_id="im-1",
        selector="recovery",
        inherited_score=0.3,
        inherited_best=0.6,
        outcomes=outcomes,
    )
    metrics = group.metrics()
    assert metrics["nondegenerate"]
    assert metrics["solve_at_8"]
    assert metrics["single_behavior_version"]
    summary = summarize_groups([group])
    assert summary["groups"] == 1
    assert summary["nondegenerate_rate"] == 1.0

    with pytest.raises(ValueError, match="8 siblings"):
        BranchSurveyGroup(
            snapshot_id="im-2",
            selector="x",
            inherited_score=0,
            inherited_best=0,
            outcomes=outcomes[:7],
        )


def test_chain_checkpoint_restores_exact_fully_masked_prefix():
    model_mod = _import_with_stubs("agentic_rl.retro.model")
    chain_mod = _import_with_stubs("agentic_rl.model")
    chain = chain_mod.Chain()
    chain.tokens = [11, 12, 13]
    chain.loss_mask = [0, 1, 1]
    chain.logprobs = [0.0, -0.2, -0.3]
    chain.versions = ["v7"]
    chain.prompt_len = 1
    chain.seen_msgs = 2
    chain.msg_hashes = ["system", "user"]

    agent = SimpleNamespace(
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "observation"},
        ],
        n_calls=1,
        cost=0.0,
        extra_template_vars={"task": "x"},
    )
    source_model = SimpleNamespace(chains=[chain], cur=chain)
    checkpoint = model_mod.capture_checkpoint(agent, source_model)

    restored_model = SimpleNamespace(
        chains=[],
        tokenizer=SimpleNamespace(decode=lambda tokens, skip_special_tokens=False: str(tokens)),
    )
    model_mod.restore_recording_model(restored_model, checkpoint)
    restored = restored_model.chains[0]
    assert restored.tokens == [11, 12, 13]
    assert restored.prompt_len == 3
    assert restored.loss_mask == [0, 0, 0]
    assert restored.logprobs == [0.0, 0.0, 0.0]
    assert restored.versions == []
    assert restored.seen_msgs == 2
