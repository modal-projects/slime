"""Deprecated shim (restructure 2026-08-26): moved to agentic_rl.envs.legacy.openthoughts_agent_convert."""

from agentic_rl.envs.legacy.openthoughts_agent_convert import *  # noqa: F401,F403
if __name__ == "__main__":
    import runpy

    runpy.run_module("agentic_rl.envs.legacy.openthoughts_agent_convert", run_name="__main__")
