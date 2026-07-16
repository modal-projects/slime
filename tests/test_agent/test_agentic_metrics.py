"""The custom rollout-log hook must surface the tail/straggler + health metrics
that made the ~3h-straggler slowness invisible on the old dashboard.

``agentic_rl.metrics`` only needs numpy at module scope (its slime imports live inside
``log_rollout_data``), so we exercise ``_agentic_metrics`` / ``_async_metrics`` directly
with fake samples. numpy must be real (stubbed numpy would make every stat a MagicMock);
any other missing dep is permissively stubbed so the import resolves on a CPU env.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


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


try:
    import numpy  # noqa: F401 - real numpy is required for meaningful stats
    metrics = _import_with_stubs("agentic_rl.metrics")
except Exception as exc:  # pragma: no cover - unsatisfiable import env
    pytest.skip(f"agentic_rl.metrics unimportable: {exc}", allow_module_level=True)


def _s(**agentic):
    return SimpleNamespace(metadata={"agentic": agentic}, weight_versions=agentic.get("_wv", []))


def test_tail_and_health_metrics():
    samples = [
        _s(turns=10, elapsed_sec=100.0, output_tokens=500, response_tokens=1000, gen_time=5.0, exec_timeouts=0, is_solved=True, exit_status="Submitted"),
        _s(turns=50, elapsed_sec=200.0, output_tokens=800, response_tokens=2000, gen_time=8.0, exec_timeouts=0, is_solved=False, exit_status="step_limit"),
        # the straggler: 3h, hit exec timeouts, died via the exception path (still measured)
        _s(turns=75, elapsed_sec=11000.0, output_tokens=300, response_tokens=20000, gen_time=10.0, exec_timeouts=3, is_solved=False, exit_status="episode_exception"),
    ]
    out = metrics._agentic_metrics(samples, SimpleNamespace(agentic_episode_timeout=1800))

    assert out["agentic/elapsed_sec/max"] == 11000.0
    assert out["agentic/elapsed_sec/p50"] == 200.0
    assert out["agentic/straggler_ratio"] == pytest.approx(11000.0 / 200.0)  # max / median
    assert out["agentic/budget_hit_frac"] == pytest.approx(1 / 3)  # only the 11000s sample >= 0.95*1800
    assert out["agentic/exec_timeouts/mean"] == pytest.approx(1.0)  # (0+0+3)/3
    assert out["agentic/exec_timeout_frac"] == pytest.approx(1 / 3)  # 1 of 3 episodes wedged
    assert out["agentic/trained_token_frac"] == pytest.approx(1600 / 23000)
    assert out["agentic/turns/p90"] == pytest.approx(70.0)  # np.percentile([10,50,75], 90)
    # exit-status mix still emitted
    assert out["agentic/exit_frac/episode_exception"] == pytest.approx(1 / 3)


def test_no_elapsed_degrades_gracefully():
    """Old samples without elapsed_sec must not crash or emit straggler keys."""
    out = metrics._agentic_metrics([_s(turns=5)], SimpleNamespace())
    assert "agentic/elapsed_sec/max" not in out
    assert "agentic/straggler_ratio" not in out
    assert out["agentic/turns/p90"] == 5.0


def test_timing_phase_max_and_p90():
    """Per-phase timing must surface max (+ p90 for agent/verifier), not just mean --
    a hung verifier that burns the full timeout while the agent finished in minutes
    (e.g. caikit__caikit-657: agent 180s, verifier/RM 1801s) is invisible in the mean.
    ``generate`` (LLM time) now lives under timing/ and feeds the decode rate."""
    samples = [
        _s(turns=10, output_tokens=1000, timing={"boot": 1.0, "agent": 100.0, "generate": 50.0, "verifier": 30.0, "episode": 140.0}),
        _s(turns=61, output_tokens=2000, timing={"boot": 2.0, "agent": 180.0, "generate": 100.0, "verifier": 1800.0, "episode": 2010.0}),
    ]
    out = metrics._agentic_metrics(samples, SimpleNamespace())
    assert out["agentic/timing/agent/mean"] == pytest.approx(140.0)
    assert out["agentic/timing/agent/max"] == pytest.approx(180.0)
    assert out["agentic/timing/generate/mean"] == pytest.approx(75.0)  # LLM time, now under timing/
    assert out["agentic/timing/generate/max"] == pytest.approx(100.0)
    assert out["agentic/timing/verifier/max"] == pytest.approx(1800.0)  # the hung grade
    assert out["agentic/timing/boot/max"] == pytest.approx(2.0)
    assert out["agentic/timing/episode/max"] == pytest.approx(2010.0)
    assert out["agentic/timing/verifier/p90"] == pytest.approx(1623.0)  # np.percentile([30,1800],90)
    assert out["agentic/timing/agent/p90"] == pytest.approx(172.0)
    # decode rate reads timing.generate now: mean(1000/50, 2000/100) = mean(20, 20) = 20
    assert out["agentic/decode_tok_per_s/mean"] == pytest.approx(20.0)
    assert "agentic/gen_time/mean" not in out  # gen_time is no longer a flat metric


def test_async_sample_age_and_version_span():
    now = 1000.0
    samples = [
        SimpleNamespace(metadata={"agentic": {"gen_timestamp": now - 300}}, weight_versions=["1", "2"]),
        SimpleNamespace(metadata={"agentic": {"gen_timestamp": now - 100}}, weight_versions=["2"]),
    ]
    out = metrics._async_metrics(samples, now)
    assert out["async/sample_age_sec/max"] == 300.0
    assert out["async/sample_age_sec/mean"] == pytest.approx(200.0)
    assert out["async/version_span/max"] == 2.0
    assert out["async/version_lag/max"] == 1.0  # batch freshest=2, stalest min=1


def _outcome_sample(final=None, best=None, reward=0.0):
    outcome = {} if final is None else {"final": final, "best_submitted": best}
    return SimpleNamespace(metadata={"agentic": {"outcome": outcome}} if final is not None else {}, reward=reward)


def test_final_and_best_traj_reward_helpers():
    """The per-sample readers behind rollout/average_last_reward (O0 reference:
    unshaped final grade) and rollout/best_reward_per_traj (O1 reference:
    max(final, best submitted))."""
    s = _outcome_sample(final=0.2, best=0.9)
    assert metrics._final_reward(s) == 0.2
    assert metrics._best_traj_reward(s) == 0.9

    # the final grade already beats every submission -> best-of-traj == final
    s = _outcome_sample(final=0.95, best=0.4)
    assert metrics._best_traj_reward(s) == 0.95

    # no submissions in the episode -> degrade to the final grade
    s = _outcome_sample(final=0.42, best=None)
    assert metrics._best_traj_reward(s) == 0.42

    # no outcome breakdown at all (masked null / non-harbor): fall back to reward
    s = _outcome_sample(reward=0.7)
    assert metrics._final_reward(s) == 0.7
    assert metrics._best_traj_reward(s) == 0.7
    assert metrics._best_traj_reward(_outcome_sample(reward=None)) == 0.0  # null sample
