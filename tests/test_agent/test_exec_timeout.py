"""A hung in-sandbox command must not stall the rollout, and no single command may
outlast the agent's wall-time budget.

Regression for the ~3h straggler on ``w_qwen3_6_swe_rebench_v2_noncolocate_3n``
(run 3kodg5qc, rollout_0): one bash call (a benign ``python3`` file-edit) issued
50th in an episode never returned. ``Sandbox.exec`` passed ``timeout=`` to Modal --
which only kills the SERVER-side process -- then read ``p.stdout``/``p.wait()`` with
no client deadline, so a wedged gRPC stream blocked ~3h until Modal tore it down.
That one sample == the whole sync rollout step (trainer idle ~95%).

The fix bounds the whole round-trip client-side (rc 124 on a wedge) and, while an
agent leg is active, caps each command at the episode's remaining wall-time via
``Sandbox.deadline`` (armed by ``RolloutEnv.run_agent_leg``).

Importing ``agentic_rl.sandbox`` pulls in ``modal`` / ``minisweagent``; on a CPU env
we stub exactly the missing modules (never installed ones), so this runs in CI and
locally and skips cleanly if the import still can't be satisfied.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _import_with_stubs(modname: str):
    """Import ``modname``, permissively stubbing each *missing* dependency and retrying.
    Only modules that fail to import get stubbed, so a full env uses the real ones."""
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
    sandbox_mod = _import_with_stubs("agentic_rl.sandbox")
    Sandbox = sandbox_mod.Sandbox
except Exception as exc:  # pragma: no cover - unsatisfiable import env
    pytest.skip(f"agentic_rl.sandbox unimportable: {exc}", allow_module_level=True)


class _FakeStream:
    def __init__(self, data: bytes = b"", block: bool = False):
        self._data, self._block = data, block

    def read(self) -> bytes:
        if self._block:
            threading.Event().wait()  # block forever; runs in a daemon thread
        return self._data


class _FakeProc:
    def __init__(self, rc: int = 0, out: bytes = b"", err: bytes = b"", block: bool = False):
        self.stdout = _FakeStream(out, block)
        self.stderr = _FakeStream(err)
        self._rc = rc

    def wait(self) -> int:
        return self._rc


class _FakeModalSB:
    """Stands in for ``modal.Sandbox`` -- records the ``timeout=`` exec() was given."""

    def __init__(self, proc: _FakeProc):
        self._proc, self.calls = proc, []

    def exec(self, *args, timeout=None, text=None):
        self.calls.append(SimpleNamespace(args=args, timeout=timeout))
        return self._proc


def _sandbox(proc: _FakeProc, *, exec_timeout=120, deadline=None) -> "Sandbox":
    """A Sandbox wired to a fake Modal handle, bypassing the real boot in __init__."""
    sb = Sandbox.__new__(Sandbox)
    sb.sb = _FakeModalSB(proc)
    sb.cwd = "/"
    sb.exec_timeout = exec_timeout
    sb.exec_time = 0.0
    sb.exec_count = 0
    sb.exec_timeouts = 0
    sb.deadline = deadline
    return sb


def test_wedged_stream_returns_124_fast(monkeypatch):
    """A command whose output stream never returns is bounded, not a 3h hang."""
    monkeypatch.setattr(sandbox_mod, "_EXEC_GRACE_SEC", 0.3)
    sb = _sandbox(_FakeProc(block=True), exec_timeout=0.3)
    t0 = time.monotonic()
    rc, out, err = sb.exec("python3 -c 'while True: pass'")
    assert rc == 124
    assert "timed out" in err
    assert time.monotonic() - t0 < 5  # bounded by budget+grace, not the stream
    assert (sb.exec_count, sb.exec_timeouts) == (1, 1)  # counted for the canary metric


def test_check_true_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_EXEC_GRACE_SEC", 0.3)
    sb = _sandbox(_FakeProc(block=True), exec_timeout=0.3)
    with pytest.raises(TimeoutError):
        sb.exec("hang", check=True)


def test_caps_timeout_at_remaining_budget():
    """(a) With a leg deadline ~5s out, the command's timeout is capped to it, not 120."""
    sb = _sandbox(_FakeProc(rc=0, out=b"ok"), exec_timeout=120, deadline=time.monotonic() + 5)
    rc, out, err = sb.exec("echo ok")
    assert (rc, out) == (0, "ok")
    assert 0 < sb.sb.calls[-1].timeout <= 5  # no command outlasts the remaining budget


def test_refuses_when_budget_exhausted():
    """(a) Past the deadline: don't even dispatch -- return a timeout observation."""
    sb = _sandbox(_FakeProc(rc=0, out=b"ok"), deadline=time.monotonic() - 1)
    rc, out, err = sb.exec("echo late")
    assert rc == 124
    assert "exhausted" in err
    assert sb.sb.calls == []  # never reached the sandbox
    assert (sb.exec_count, sb.exec_timeouts) == (1, 0)  # attempted, but not a wedge


def test_normal_passthrough_without_deadline():
    """No deadline (boot/prep/verify): output + rc flow through, uncapped at exec_timeout."""
    sb = _sandbox(_FakeProc(rc=3, out=b"hello", err=b"warn"))
    rc, out, err = sb.exec("echo hello")
    assert (rc, out, err) == (3, "hello", "warn")
    assert sb.sb.calls[-1].timeout == 120
    assert (sb.exec_count, sb.exec_timeouts) == (1, 0)


def test_run_agent_leg_arms_and_clears_deadline(monkeypatch):
    """(a) end-to-end: the agent leg arms ``sandbox.deadline`` ~now+wall_time and clears it."""
    base_mod = _import_with_stubs("agentic_rl.environment.base")
    sb = _sandbox(_FakeProc(rc=0, out=b"ok"))
    observed: dict = {}

    class _FakeAgent:
        def __init__(self, model, sandbox, **kw):
            self._sb = sandbox
            observed["wall"] = kw.get("wall_time_limit_seconds")

        def run(self, task):
            observed["deadline_during"] = self._sb.deadline
            return {"exit_status": "ok"}

    fake_pkg = types.ModuleType("minisweagent.agents")
    fake_pkg.__path__ = []
    fake_default = types.ModuleType("minisweagent.agents.default")
    fake_default.DefaultAgent = _FakeAgent
    monkeypatch.setitem(sys.modules, "minisweagent.agents", fake_pkg)
    monkeypatch.setitem(sys.modules, "minisweagent.agents.default", fake_default)

    t0 = time.monotonic()
    info = base_mod.RolloutEnv.run_agent_leg(object(), sb, "task", max_steps=5, wall_time_sec=900)

    assert info == {"exit_status": "ok"}
    assert observed["wall"] == 900
    assert observed["deadline_during"] is not None
    assert t0 + 800 < observed["deadline_during"] < t0 + 1000  # armed during the leg
    assert sb.deadline is None  # cleared after
