"""Regression tests for persisting the agent exit code + log tail.

A zero-turn rollout aborts as ``adapter_session_empty`` on the graceful success
path (the agent process launched but made no adapter calls). Previously the dump
could not say *why* turns=0 — ``_detached_run`` only logged the exit code/tail to
tail-only Modal logs that age out. These tests pin the wiring that now carries
``agent_exit_code`` (+ a failure-only ``agent_tail``) into the abort sample's
metadata so the dump self-explains (e.g. exit=137 -> OOM-killed).

- ``test_detached_run_returns_exit_and_tail`` (unit): fake sandbox, asserts
  ``_detached_run`` returns the parsed exit code with the log tail on a nonzero
  exit and an empty tail on a clean exit.
- ``test_empty_session_carries_agent_diag`` (unit): ``_merge_samples`` empty
  path copies ``agent_exit_code``/``agent_tail`` from ``reward_result.extra``
  onto the aborted sample's metadata, and stays clean when they're absent.
- ``test_abort_result_merges_extra`` (unit): ``_abort_result`` merges an extra
  dict without clobbering ``abort_reason``.
"""
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from async_rl_research.agent.base import EXIT_BUDGET_EXCEEDED, AgentRunResult, AgentRuntime
from async_rl_research.environment.base import RewardResult
from async_rl_research.generate import _abort_result, _merge_samples
from slime.utils.types import Sample

NUM_GPUS = 0


def test_agent_run_result_defaults():
    assert AgentRunResult(0).tail == ""
    assert AgentRunResult(137, "boom").exit_code == 137
    assert EXIT_BUDGET_EXCEEDED == -2


# --------------------------------------------------------------------------
# _detached_run: the new exit-code + failure-tail capture
# --------------------------------------------------------------------------
class _FakeRuntime(AgentRuntime):
    name = "fake"
    adapter_cls = object  # non-None is all __init_subclass__ requires

    async def run_agent(self, *a, **k):  # unused; abstract method must exist
        raise NotImplementedError


class _FakeSandbox:
    """Minimal sandbox: the done-marker poll yields ``exit_code``; the tail
    command yields ``tail_text``; everything else (rm/chmod/setsid) is a no-op."""

    def __init__(self, exit_code, tail_text):
        self._exit = exit_code
        self._tail = tail_text

    async def write_file(self, path, body):
        return None

    async def exec(self, cmd, check=False, timeout=None):
        if "cat" in cmd and "_done" in cmd:  # poll: `test -f .. && cat ..`
            return (0, str(self._exit), "")
        if "tail -c 4000" in cmd:
            return (0, self._tail, "")
        return (0, "", "")  # rm / chmod / setsid launch


def _run_detached(exit_code, tail_text):
    rt = _FakeRuntime()
    sb = _FakeSandbox(exit_code, tail_text)
    return asyncio.run(
        rt._detached_run(
            sb,
            workdir="/app",
            command="true",
            time_budget_sec=5,
            poll_interval_sec=0.01,  # don't sleep 5s in a test
        )
    )


def test_detached_run_returns_exit_and_tail():
    # nonzero exit -> tail captured and returned (137 == 128 + SIGKILL == OOM)
    res = _run_detached(137, "fatal: Out of memory\nKilled")
    assert isinstance(res, AgentRunResult)
    assert res.exit_code == 137
    assert "Out of memory" in res.tail

    # clean exit -> no tail read (empty), per "tail only on failure"
    ok = _run_detached(0, "should-not-be-read")
    assert ok.exit_code == 0
    assert ok.tail == ""


# --------------------------------------------------------------------------
# _merge_samples empty path carries the diag onto the abort
# --------------------------------------------------------------------------
def _merge_empty(extra):
    # On the empty path _merge_samples returns before touching `state`, so a
    # dummy is safe.
    return _merge_samples(
        sample=Sample(index=0, prompt="x"),
        state=None,
        segments=[],
        reward_result=RewardResult(reward=0.0, is_solved=False, extra=extra),
        elapsed_sec=1.0,
        instance_id="gravitational__teleport-deadbeef",
    )


def test_empty_session_carries_agent_diag():
    out = _merge_empty({"agent_exit_code": 137, "agent_tail": "Killed (OOM)", "harbor_steps_total": 1})
    assert len(out) == 1
    md = out[0].metadata
    assert md["abort_reason"] == "adapter_session_empty"
    assert md["agent_exit_code"] == 137
    assert md["agent_tail"] == "Killed (OOM)"
    # only the agent diag is copied, not unrelated extra keys
    assert "harbor_steps_total" not in md


def test_empty_session_without_diag_is_clean():
    # e.g. budget exhausted before any leg ran -> extra has no agent_* keys
    out = _merge_empty({})
    md = out[0].metadata
    assert md["abort_reason"] == "adapter_session_empty"
    assert "agent_exit_code" not in md
    assert "agent_tail" not in md


def test_venv_setup_hardens_pydantic_core_import():
    # Provisioning verifies the agent's deep import `import minisweagent.agents.default`
    # and force-reinstalls (--reinstall --no-cache) to repair a partial wheel.
    # `-P` is load-bearing: the check runs with cwd = the image WORKDIR (e.g.
    # /testbed), so without it a task repo named like an agent dep (the pydantic
    # SWE-gym tasks ship /testbed/pydantic/) shadows the venv and crashes the
    # import -> a deterministic `exception:RuntimeError`. Verified fixed on
    # pydantic-6104/6043/8511 (rc 1 -> 0). See profiles/provisioning_repro_pydantic.py.
    from async_rl_research.agent.mini_swe_agent import _VENV_CHECK, _VENV_SETUP

    assert "minisweagent.agents.default" in _VENV_SETUP
    assert "--reinstall" in _VENV_SETUP and "--no-cache" in _VENV_SETUP
    assert "minisweagent.agents.default" in _VENV_CHECK
    # the import checks MUST use `-P` (keep cwd/workdir off sys.path)
    assert "-P -c 'import minisweagent.agents.default'" in _VENV_CHECK
    assert _VENV_SETUP.count("-P -c 'import minisweagent.agents.default'") == 2
    # the repair script must be valid bash (it's assembled as a Python string)
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash:
        r = subprocess.run([bash, "-n"], input=_VENV_SETUP, text=True, capture_output=True)
        assert r.returncode == 0, f"_VENV_SETUP is not valid bash:\n{r.stderr}"


def test_abort_result_merges_extra():
    out = _abort_result(Sample(index=1, prompt="x"), "adapter_session_empty", extra={"agent_exit_code": 1})
    md = out[0].metadata
    assert md["abort_reason"] == "adapter_session_empty"
    assert md["agent_exit_code"] == 1
    # extra is optional: other abort reasons still work
    out2 = _abort_result(Sample(index=2, prompt="x"), "boot_timeout:600s")
    assert out2[0].metadata["abort_reason"] == "boot_timeout:600s"
    assert "agent_exit_code" not in out2[0].metadata
