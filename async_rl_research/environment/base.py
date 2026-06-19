"""RolloutEnv: the contract between ``generate.py`` and one task family.

An *env* packages everything task-family-specific (SWE-Gym, harbor, ...): row
validation, sandbox boot/prep, driving the agent across step(s), grading into a
reward. Mirrors ``agent/base.py``'s AgentRuntime: one ``generate()``
orchestrates ``runtime x env``.

Schema-pair convention: ``environment/<name>.py`` and ``environment/convert2slime/<name>.py``
are paired -- the converter is the only writer of the ``metadata`` dict and
``normalize_metadata`` the only reader. Rows select their env via
``metadata.task_type`` (absent -> ``harbor``).

Writing a new env: subclass, declare ``name``, implement ``normalize_metadata``
+ ``rollout``, register in ``ENVS``. ``rollout`` owns the whole sandbox
lifecycle. Envs are cached once per worker and must be stateless across samples.
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


# Problem statement written to the workdir; always excluded from captured diffs.
PROBLEM_FILE = "PROBLEM_STATEMENT.md"


class EnvMetadataError(ValueError):
    """A dataset row is unusable for this env; str(err) becomes the abort reason."""


@dataclass(frozen=True)
class RewardResult:
    """What an env's ``rollout`` hands back to the trajectory merge: a scalar
    ``reward``, an ``is_solved`` flag for solve-rate logging, and ``extra``
    env-specific diagnostics merged into trajectory metadata.
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
        # Fail at import time, not mid-rollout.
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) is None:
            raise TypeError(f"{cls.__name__} must define class attribute 'name' (see RolloutEnv)")

    @abstractmethod
    def normalize_metadata(self, sample) -> dict[str, Any]:
        """Normalize one dataset row (slime ``Sample``) into the env's md dict.

        Must include ``instance_id``, plus ``workdir``/``agent_config`` if the
        runtime reads them. Raises ``EnvMetadataError`` for unrunnable rows.
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

        Call ``runtime.run_agent`` per agent leg; all legs share one adapter
        session so a multi-step episode stays one trajectory.
        ``agent_time_budget_sec`` bounds TOTAL agent wallclock across legs;
        ``eval_timeout_sec`` caps each grading command.
        """

    def effective_budgets(
        self, md: dict[str, Any], *, agent_time_budget_sec: int, eval_timeout_sec: int
    ) -> dict[str, int]:
        """Wall-clock budgets actually enforced this rollout (for the dump/dashboard)."""
        from ..modal_sandbox import ModalSandbox

        return {
            "boot_sec": ModalSandbox._boot_timeout_from_env(),
            "agent_sec": agent_time_budget_sec,
            "eval_sec": eval_timeout_sec,
        }

    # ------------------------------------------------------------------
    # Shared sandbox helpers
    # ------------------------------------------------------------------
    @staticmethod
    async def write_problem_file(sb, workdir: str, text: str | None) -> None:
        await sb.write_file(f"{workdir}/{PROBLEM_FILE}", text or "")

    @staticmethod
    async def upload_dir(sb, host_dir: str | Path, sandbox_dir: str) -> None:
        """Copy a host dir's CONTENTS into a fresh ``sandbox_dir`` via one
        gzipped tar (task tests/solution dirs are small)."""
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
DEFAULT_ENV = "harbor"

# task_type -> "module:Class" (strings so importing base.py imports no env module).
ENVS: dict[str, str] = {
    "harbor": "async_rl_research.environment.harbor:HarborEnv",
}

_ENV_CACHE: dict[str, RolloutEnv] = {}


def load_env(spec: str | None = None) -> RolloutEnv:
    """Resolve ``spec`` (a row's ``metadata.task_type``) to a cached env.

    Accepts a registry short name ("harbor") or "pkg.module:Class"; absent ->
    ``DEFAULT_ENV``.
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
