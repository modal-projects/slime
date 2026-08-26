"""Core model-seam layer: the de-forked fully-async rollout machinery.

See RUNBOOK.md §7 — this package holds mechanism shared by every arm
(vanilla and retro), string-loaded via slime hooks. Nothing in here may
import from agentic_rl.retro (retro composes core, never the reverse).
"""
