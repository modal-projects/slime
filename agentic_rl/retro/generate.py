"""Explicit custom-generate hook for snapshot capture and retro branches."""

from __future__ import annotations

from agentic_rl.generate import generate as generate_agentic

from .group import RETRO_ENV_SPEC


async def generate(args, sample, sampling_params, evaluation: bool = False):
    sample.metadata = {
        **(sample.metadata or {}),
        "task_type": RETRO_ENV_SPEC,
    }
    return await generate_agentic(args, sample, sampling_params, evaluation=evaluation)
