from unittest.mock import MagicMock, Mock

import pytest

from agentic_rl.core.sandbox import Sandbox
from agentic_rl.core.timing import PhaseTimer
from agentic_rl.envs.base import EpisodeLimits
from agentic_rl.envs.harbor.env import HarborEnv


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr("time.sleep", Mock())


@pytest.mark.parametrize("operation", ["read_file", "write_file"])
def test_file_transport_retry(operation):
    sandbox = object.__new__(Sandbox)
    sandbox.sb = Mock()
    sandbox.exec_timeout = 1
    rpc = getattr(sandbox.sb.filesystem, "read_text" if operation == "read_file" else "write_text")
    rpc.side_effect = [AttributeError("'Connection' object has no attribute '_transport'"), "ok"]
    args = ("/problem",) if operation == "read_file" else ("/problem", "instruction")
    getattr(sandbox, operation)(*args)
    assert rpc.call_count == 2


@pytest.mark.parametrize(
    ("error", "attempts"),
    [(ConnectionError("closed"), 3), (AttributeError("unrelated"), 1), (FileNotFoundError("missing"), 1)],
)
def test_rpc_retry_exhaustion_and_permanent_errors(error, attempts):
    operation = Mock(side_effect=error)
    with pytest.raises(type(error)):
        Sandbox._rpc_retry(operation)
    assert operation.call_count == attempts


@pytest.fixture
def setup_env(tmp_path):
    env = HarborEnv()
    env._pre_agent_setup = Mock()
    sandboxes = [MagicMock(spec=Sandbox) for _ in range(3)]
    for sandbox in sandboxes:
        sandbox.__enter__.return_value = sandbox
    env._sandbox = Mock(side_effect=sandboxes)
    md = {"instance_id": "task", "task_dir": str(tmp_path), "workdir": "/repo"}
    step = {"instruction": "fix it"}
    return env, sandboxes, md, step


@pytest.mark.parametrize("operation", ["exec", "write_file"])
def test_setup_timeout_replaces_sandbox(setup_env, operation):
    env, sandboxes, md, step = setup_env
    getattr(sandboxes[0], operation).side_effect = TimeoutError("unresponsive")
    sandbox, workdir = env._prepare_sandbox(md, step, 1800, EpisodeLimits(), PhaseTimer())
    assert sandbox is sandboxes[1]
    assert workdir == "/repo"
    sandboxes[0].terminate.assert_called_once()
    sandboxes[1].terminate.assert_not_called()
    sandboxes[1].write_file.assert_called_once_with("/repo/PROBLEM_STATEMENT.md", "fix it")
    assert env._sandbox.call_count == 2
    env._pre_agent_setup.assert_called_once()


@pytest.mark.parametrize(("error", "attempts"), [(TimeoutError("unresponsive"), 3), (ValueError("bad setup"), 1)])
def test_setup_retry_exhaustion_and_cleanup(setup_env, error, attempts):
    env, sandboxes, md, step = setup_env
    for sandbox in sandboxes:
        sandbox.write_file.side_effect = error
    with pytest.raises(type(error)):
        env._prepare_sandbox(md, step, 1800, EpisodeLimits(), PhaseTimer())
    assert env._sandbox.call_count == attempts
    for sandbox in sandboxes[:attempts]:
        sandbox.terminate.assert_called_once()


def test_agent_failure_does_not_replay_episode(setup_env):
    env, sandboxes, md, step = setup_env
    md.update(steps=[step], reward_strategy=None)
    run_leg = Mock(side_effect=ConnectionError("closed"))
    with pytest.raises(ConnectionError):
        env._episode(md, run_leg=run_leg, agent_budget_sec=1800, limits=EpisodeLimits())
    run_leg.assert_called_once()
    env._sandbox.assert_called_once()
    sandboxes[0].__exit__.assert_called_once()


def test_multiple_steps_keep_sandbox_and_prepare_once(setup_env):
    env, sandboxes, md, _ = setup_env
    steps = [
        {"instruction": text, "name": text, "tests_path": "tests", "verifier": {}, "min_reward": None}
        for text in ("first", "second")
    ]
    md.update(steps=steps, reward_strategy=None, verifier={})
    sandbox = sandboxes[0]
    sandbox.exec_count = 1
    sandbox.exec_time = 0.1
    sandbox.exec_timeouts = 0
    sandbox.exec_durations = [0.1]
    env._verify = Mock(return_value={"reward": 1.0})
    run_leg = Mock()
    result = env._episode(md, run_leg=run_leg, agent_budget_sec=1800, limits=EpisodeLimits())
    assert result.is_solved
    env._sandbox.assert_called_once()
    assert [call.args[0] for call in run_leg.call_args_list] == [sandbox, sandbox]
    assert [call.args[1] for call in sandbox.write_file.call_args_list] == ["first", "second"]
    assert env._pre_agent_setup.call_count == 2
