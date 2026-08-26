"""Contract #10: old string-loaded entry-point paths must keep resolving.

Configs in this repo, the guide repo, and W&B-recorded commands reference the
pre-restructure module paths. The deprecation shims must resolve to the SAME
objects as the new canonical paths for one deprecation window.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_with_stubs(modname: str):
    """Import ``modname``, stubbing each missing third-party dep (see test_behavior_lag)."""

    for _ in range(60):
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            missing = exc.name
            if not missing or missing in sys.modules or missing.startswith("agentic_rl"):
                raise
            stub = types.ModuleType(missing)
            stub.__path__ = []
            stub.__getattr__ = lambda _name: MagicMock()  # noqa: B023
            sys.modules[missing] = stub
    return importlib.import_module(modname)


@pytest.mark.parametrize(
    ("old_module", "attr", "new_module"),
    [
        ("agentic_rl.generate", "generate", "agentic_rl.core.generate"),
        ("agentic_rl.metrics", "log_rollout_data", "agentic_rl.obs.metrics"),
        ("agentic_rl.model", "Chain", "agentic_rl.core.model"),
        ("agentic_rl.sandbox", "Sandbox", "agentic_rl.core.sandbox"),
        ("agentic_rl.prompts", "BASH_TOOL", "agentic_rl.core.prompts"),
        ("agentic_rl.environment.base", "load_env", "agentic_rl.envs.base"),
        ("agentic_rl.environment.harbor", "HarborEnv", "agentic_rl.envs.harbor.env"),
        ("agentic_rl.environment.frontiercs", "FrontierCsEnv", "agentic_rl.envs.frontier_cs.env"),
        ("agentic_rl.environment.rewards", "SHAPERS", "agentic_rl.rewards.rewards"),
        (
            "agentic_rl.environment.submissions",
            "parse_submissions_log",
            "agentic_rl.envs.frontier_cs.submissions",
        ),
        (
            "agentic_rl.environment.convert2slime.harbor",
            "main",
            "agentic_rl.envs.harbor.convert",
        ),
        (
            "agentic_rl.retro.launch_config",
            "build_launch_configs",
            "agentic_rl.launch.launch_config",
        ),
        ("agentic_rl.turn_reward", "compute_turn_rewards", "agentic_rl.rewards.turn_reward"),
    ],
)
def test_old_path_resolves_to_the_same_object(old_module, attr, new_module):
    old = _import_with_stubs(old_module)
    new = _import_with_stubs(new_module)
    assert getattr(old, attr) is getattr(new, attr)


def test_env_registry_specs_resolve_through_new_paths():
    from agentic_rl.envs.base import ENVS

    for spec in ENVS.values():
        module_path, _, class_name = spec.partition(":")
        module = _import_with_stubs(module_path)
        assert hasattr(module, class_name), spec


def test_modal_launcher_shim_parses_and_reexports():
    # modal isn't installed in CI; assert the shim's shape statically.
    import ast

    shim = (_REPO_ROOT / "agentic_rl" / "retro" / "modal_train.py").read_text()
    tree = ast.parse(shim)
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "agentic_rl.launch.modal_train"
        for alias in node.names
    }
    assert {"app", "train", "download_data", "post_process_data"} <= imported


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
