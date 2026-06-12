"""RolloutEnv: the contract between ``generate.py`` and one task family.

An *env* packages everything specific to one task/dataset family (SWE-Gym,
harbor, ...): how to validate a dataset row, how to boot + prepare the task
sandbox, how to drive the agent across the task's step(s), and how to grade
the result into a reward. Everything else -- adapter/HTTP lifecycle, session
management, trajectory merge, abort/timeout isolation -- is the generic
recipe in ``generate.py`` and never changes per task family.

This mirrors ``agent/base.py``'s AgentRuntime exactly: one ``generate()``
orchestrates ``runtime x env``. The runtime knows *which agent* runs and how
to launch it; the env knows *what task* it runs on and what reward it earned.

Schema-pair convention
----------------------
``env/<name>.py`` and ``env/convert2slime/<name>.py`` are a pair: the
converter is the ONLY writer of the ``metadata`` dict for task type
``<name>`` and the env's ``normalize_metadata`` is the only reader. When a
field is added, both edits land in sibling files. Rows select their env via
``metadata.task_type`` (absent -> ``swe_gym``, the historical schema).

Writing a new env
-----------------
Subclass, declare ``name``, implement ``normalize_metadata`` + ``rollout``,
register in ``ENVS``. ``rollout`` owns the sandbox lifecycle end-to-end
(boot, prep, agent run(s), grading) so each family can sequence them freely:
SWE grades a captured diff in a separate CLEAN sandbox after the work sandbox
closes; harbor verifies in-place inside the still-open agent sandbox (and
loops over steps for multi-step tasks).

Envs are instantiated once per rollout worker (cached by ``load_env``) and
must be stateless across samples -- ``rollout`` receives everything per-call.
"""

from __future__ import annotations

import gzip
import importlib
import io
import logging
import shlex
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# Task-layer artifact shared by all envs: the problem statement lands in the
# workdir (agents/runners read it from disk via MSWE_PROBLEM_FILE etc.) and is
# always excluded from any captured diff.
PROBLEM_FILE = "PROBLEM_STATEMENT.md"


class EnvMetadataError(ValueError):
    """A dataset row is unusable for this env; str(err) becomes the abort reason."""


@dataclass(frozen=True)
class RewardResult:
    """What an env's ``rollout`` hands back to the trajectory merge.

    ``reward`` is the scalar training signal; ``is_solved`` the boolean used
    for run-level solve-rate logging; ``extra`` is env-specific diagnostics
    merged into the trajectory metadata (SWE: ``applied_cleanly``; harbor:
    the raw rewards dict + per-step results).
    """

    reward: float
    is_solved: bool
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutEnv(ABC):
    """One task family's integration: row schema + sandbox + grading.

    ``generate.py`` only ever touches: ``name``, ``normalize_metadata``,
    ``rollout``.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        # Fail at import time, not mid-rollout (same rule as AgentRuntime).
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) is None:
            raise TypeError(f"{cls.__name__} must define class attribute 'name' (see RolloutEnv)")

    @abstractmethod
    def normalize_metadata(self, sample) -> dict[str, Any]:
        """Normalize one dataset row (a slime ``Sample``) into the env's md dict.

        Must include ``instance_id`` (used for session ids + logging) and
        ``workdir``/``agent_config`` if the agent runtime is expected to read
        them. Raises ``EnvMetadataError`` (str = abort reason) for rows this
        env cannot run.
        """

    @abstractmethod
    async def rollout(
        self,
        md: dict[str, Any],
        *,
        runtime,
        session_id: str,
        adapter_url: str,
        agent_time_budget_sec: int,
        eval_timeout_sec: int,
    ) -> RewardResult:
        """Run the full task episode: boot, prep, agent run(s), grading.

        ``runtime`` is the active ``agent.base.AgentRuntime``; call
        ``runtime.run_agent(sb, md=, session_id=, adapter_url=,
        time_budget_sec=)`` for each agent leg (all legs share the one
        adapter session, so a multi-step episode is still one trajectory).
        ``agent_time_budget_sec`` bounds TOTAL agent wallclock across legs;
        ``eval_timeout_sec`` caps each grading command. The caller wraps this
        whole coroutine in a wall-clock guard -- exceeding
        budget + eval + slack aborts the sample.
        """

    # ------------------------------------------------------------------
    # Shared sandbox helpers
    # ------------------------------------------------------------------
    @staticmethod
    async def write_problem_file(sb, workdir: str, text: str | None) -> None:
        await sb.write_file(f"{workdir}/{PROBLEM_FILE}", text or "")

    @staticmethod
    async def upload_dir(sb, host_dir: str | Path, sandbox_dir: str) -> None:
        """Copy a host directory's CONTENTS into ``sandbox_dir`` (created fresh).

        Ships one gzipped tar through ``write_file`` instead of N round-trips;
        task tests/solution dirs are small, so in-memory packing is fine.
        """
        host_dir = Path(host_dir)
        buf = io.BytesIO()
        # mtime=0 so re-uploads of identical content are byte-identical.
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                tar.add(host_dir, arcname=".")
        archive = f"/tmp/.upload_{abs(hash(str(host_dir))) % 10**8}.tgz"
        await sb.write_file(archive, buf.getvalue())
        q = shlex.quote
        await sb.exec(
            f"rm -rf {q(sandbox_dir)} && mkdir -p {q(sandbox_dir)} && tar -xzf {q(archive)} -C {q(sandbox_dir)}",
            check=True,
            timeout=120,
        )


def coerce_prompt(prompt) -> str:
    """Best-effort extraction of plain text from a slime prompt field."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


# ---------------------------------------------------------------------------
# Registry + loader (mirrors agent.base.RUNTIMES / load_runtime)
# ---------------------------------------------------------------------------
DEFAULT_ENV = "swe_gym"

# task_type -> "module:Class". Values are strings so importing base.py never
# imports any env module (env modules pull in provider backends).
ENVS: dict[str, str] = {
    "swe_gym": "async_rl_research.environment.swe_gym:SweGymEnv",
    "harbor": "async_rl_research.environment.harbor:HarborEnv",
}

_ENV_CACHE: dict[str, RolloutEnv] = {}


def load_env(spec: str | None = None) -> RolloutEnv:
    """Resolve ``spec`` (a row's ``metadata.task_type``) to a cached env instance.

    Accepted forms: a registry short name ("harbor"), or "pkg.module:Class"
    for out-of-tree envs. Absent -> ``DEFAULT_ENV`` (the historical SWE rows
    carry no ``task_type``).
    """
    spec = spec or DEFAULT_ENV
    cached = _ENV_CACHE.get(spec)
    if cached is not None:
        return cached
    target = ENVS.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"unknown task_type {spec!r}; known: {sorted(ENVS)} (or pass 'pkg.module:Class')")
    module_path, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module_path), attr, None)
    if not (isinstance(cls, type) and issubclass(cls, RolloutEnv)):
        raise TypeError(f"task_type {spec!r} resolved to {cls!r}, which is not a RolloutEnv subclass")
    env = cls()
    _ENV_CACHE[spec] = env
    return env
