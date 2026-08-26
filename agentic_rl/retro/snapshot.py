"""Deprecated shim (restructure 2026-08-26): moved to agentic_rl.retro.backends.modal_snapshot."""

from agentic_rl.retro.backends.modal_snapshot import *  # noqa: F401,F403
from agentic_rl.retro.backends.modal_snapshot import (  # noqa: F401
    delete_snapshot,
    restore_directory,
    sandbox_compute_cost,
    snapshot_sandbox,
    workspace_stats,
)
