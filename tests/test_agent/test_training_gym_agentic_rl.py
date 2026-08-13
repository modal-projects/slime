from __future__ import annotations

import threading
import time
import sys
import types
from types import SimpleNamespace

from agentic_rl import grade, metrics, sandbox

# The worker unit tests exercise queue/staleness logic only. Avoid importing
# SGLang's GPU server stack in a CPU test environment.
_sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")
_sglang_rollout.GenerateState = object


async def _unused_generate_and_rm_group(*_args, **_kwargs):
    raise AssertionError("generation should not run in this unit test")


_sglang_rollout.generate_and_rm_group = _unused_generate_and_rm_group
sys.modules["slime.rollout.sglang_rollout"] = _sglang_rollout

from slime.rollout import fully_async_rollout
from slime.utils.types import Sample


class _Stream:
    def __init__(self, value: bytes = b"", *, blocks: bool = False):
        self.value = value
        self.blocks = blocks

    def read(self) -> bytes:
        if self.blocks:
            threading.Event().wait()
        return self.value


class _Process:
    def __init__(self, *, output: bytes = b"", error: bytes = b"", blocks: bool = False):
        self.stdout = _Stream(output, blocks=blocks)
        self.stderr = _Stream(error)

    def wait(self) -> int:
        return 0


class _SandboxHandle:
    def __init__(self, process: _Process):
        self.process = process
        self.timeouts: list[int] = []

    def exec(self, *_args, timeout=None, **_kwargs):
        self.timeouts.append(timeout)
        return self.process


def _sandbox(process: _Process, *, timeout: int = 1, deadline=None):
    instance = sandbox.Sandbox.__new__(sandbox.Sandbox)
    instance.sb = _SandboxHandle(process)
    instance.cwd = "/"
    instance.exec_timeout = timeout
    instance.exec_time = 0.0
    instance.exec_timeouts = 0
    instance.deadline = deadline
    return instance


def test_sandbox_bounds_wedged_stream(monkeypatch):
    monkeypatch.setattr(sandbox, "_EXEC_GRACE_SEC", 0.05)
    instance = _sandbox(_Process(blocks=True), timeout=0.05)

    started = time.monotonic()
    returncode, output = instance.exec("python -c 'while True: pass'")

    assert returncode == 124
    assert "timed out" in output
    assert instance.exec_timeouts == 1
    assert time.monotonic() - started < 2


def test_sandbox_caps_command_at_episode_deadline():
    instance = _sandbox(
        _Process(output=b"ok"),
        timeout=120,
        deadline=time.monotonic() + 5,
    )

    returncode, output = instance.exec("echo ok")

    assert (returncode, output) == (0, "ok")
    assert 0 < instance.sb.timeouts[-1] <= 5


def test_grade_parses_pytest_summary():
    output = "\n".join(
        [
            "PASSED tests/test_api.py::test_fixed",
            "FAILED tests/test_api.py::test_other - AssertionError",
        ]
    )

    assert grade._passed_tests(output) == {"tests/test_api.py::test_fixed"}


def test_agentic_metrics_include_async_health():
    samples = [
        SimpleNamespace(
            metadata={
                "agentic": {
                    "turns": 2,
                    "exec_timeouts": 0,
                    "episode_time": 10.0,
                    "gen_timestamp": 900.0,
                    "solved": 1.0,
                    "exit_status": "Submitted",
                }
            },
            weight_versions=["1", "2"],
        ),
        SimpleNamespace(
            metadata={
                "agentic": {
                    "turns": 4,
                    "exec_timeouts": 2,
                    "episode_time": 20.0,
                    "gen_timestamp": 950.0,
                    "solved": 0.0,
                    "exit_status": "LimitsExceeded",
                }
            },
            weight_versions=["2"],
        ),
    ]

    agent = metrics._agentic_metrics(samples)
    async_health = metrics._async_metrics(samples, now=1000.0)

    assert agent["agentic/turns/mean"] == 3.0
    assert agent["agentic/exec_timeouts/mean"] == 1.0
    assert agent["agentic/solved_frac"] == 0.5
    assert async_health["async/version_span/max"] == 2.0
    assert async_health["async/version_lag/max"] == 1.0
    assert async_health["async/sample_age_sec/max"] == 100.0


def test_fully_async_pool_is_bounded_and_aborts_regenerate(monkeypatch):
    monkeypatch.setattr(
        fully_async_rollout,
        "GenerateState",
        lambda _args: SimpleNamespace(sampling_params={}),
    )

    class Buffer:
        def __init__(self):
            self.requeued = []

        def add_samples(self, groups):
            self.requeued.extend(groups)

    args = SimpleNamespace(rollout_max_staleness=2, rollout_batch_size=4)
    data_buffer = Buffer()
    worker = fully_async_rollout.AsyncRolloutWorker(
        args,
        data_buffer,
        concurrency=64,
    )
    assert worker.pool_limit == 8

    group = [Sample(status=Sample.Status.ABORTED), Sample(status=Sample.Status.COMPLETED)]
    worker.inflight_gids.add(3)
    done = SimpleNamespace(result=lambda: group)
    worker._make_done_cb(3)(done)

    assert 3 not in worker.inflight_gids
    assert data_buffer.requeued == [group]
    assert all(sample.status == Sample.Status.PENDING for sample in group)
