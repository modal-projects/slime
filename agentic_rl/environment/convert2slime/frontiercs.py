"""Deprecated shim (restructure 2026-08-26): moved to agentic_rl.envs.frontier_cs.convert."""

from agentic_rl.envs.frontier_cs.convert import *  # noqa: F401,F403
if __name__ == "__main__":
    import runpy

    runpy.run_module("agentic_rl.envs.frontier_cs.convert", run_name="__main__")
