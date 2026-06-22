"""RolloutEnv: the contract between ``generate.py`` and one task family.

An env packages everything task-family-specific (SWE-rebench, harbor, ...): row
validation, sandbox boot, driving the in-process mini-swe agent leg(s) against
that sandbox, and grading into a reward. The recorder (``model.RecordingModel``)
and the bash sandbox (``sandbox.Sandbox``) are the shared tools an env composes;
``generate.py`` creates the recorder, picks the env by ``metadata.task_type``,
and runs ``env.rollout`` in a worker thread.

Everything here is synchronous (one episode = one thread): the mini-swe loop and
Modal RPCs block, and slime fans many episodes out concurrently across a thread
pool. Write a new env by subclassing, declaring ``name``, implementing
``normalize_metadata`` + ``rollout``, and registering it in ``ENVS``.
"""

from __future__ import annotations

import gzip
import importlib
import io
import shlex
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .. import prompts

PROBLEM_FILE = "PROBLEM_STATEMENT.md"


class EnvMetadataError(ValueError):
    """A dataset row is unusable for this env; str(err) becomes the abort reason."""


@dataclass
class EpisodeLimits:
    max_steps: int = 30
    episode_timeout: int = 1800
    exec_timeout: int = 120
    grade_timeout: int = 1800
    eval_timeout: int = 600


@dataclass(frozen=True)
class RewardResult:
    """What an env hands back: a scalar ``reward``, an ``is_solved`` flag for
    solve-rate logging, and ``extra`` diagnostics merged into sample metadata."""

    reward: float
    is_solved: bool
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutEnv(ABC):
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) is None:
            raise TypeError(f"{cls.__name__} must define class attribute 'name'")

    @abstractmethod
    def normalize_metadata(self, sample) -> dict[str, Any]:
        """Normalize one slime ``Sample`` into this env's md dict (must include
        ``instance_id``). Raise ``EnvMetadataError`` for unrunnable rows."""

    @abstractmethod
    def rollout(self, md: dict[str, Any], *, model, limits: EpisodeLimits) -> RewardResult:
        """Boot the sandbox, run the agent leg(s) driving ``model``, grade, and
        return the episode reward. Runs synchronously in a worker thread."""

    # shared helpers --------------------------------------------------------
    @staticmethod
    def run_agent_leg(model, sandbox, task: str, *, max_steps: int, wall_time_sec: int) -> dict:
        """Run one stock mini-swe ``DefaultAgent`` leg in-process against the
        sandbox; ``model`` records tokens. Returns the agent's exit info."""
        from minisweagent.agents.default import DefaultAgent

        agent = DefaultAgent(
            model,
            sandbox,
            system_template=prompts.SYSTEM_TEMPLATE,
            instance_template=prompts.INSTANCE_TEMPLATE,
            step_limit=max_steps,
            cost_limit=0.0,
            wall_time_limit_seconds=wall_time_sec,
        )
        return agent.run(task=task) or {}

    @staticmethod
    def write_problem_file(sb, workdir: str, text: str | None) -> None:
        sb.write_file(f"{workdir}/{PROBLEM_FILE}", text or "")

    @staticmethod
    def upload_dir(sb, host_dir: str | Path, sandbox_dir: str) -> None:
        """Copy a host dir's contents into a fresh ``sandbox_dir`` via one gzipped
        tar (task tests/solution dirs are small)."""
        host_dir = Path(host_dir)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
            tar.add(host_dir, arcname=".")
        archive = f"/tmp/.upload_{abs(hash(str(host_dir))) % 10**8}.tgz"
        sb.write_file(archive, buf.getvalue())
        q = shlex.quote
        sb.exec(f"rm -rf {q(sandbox_dir)} && mkdir -p {q(sandbox_dir)} && tar -xzf {q(archive)} -C {q(sandbox_dir)}", check=True, timeout=120)


def coerce_prompt(prompt) -> str:
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


# registry --------------------------------------------------------------------
DEFAULT_ENV = "harbor"

ENVS: dict[str, str] = {
    "harbor": "agentic_rl.environment.harbor:HarborEnv",
    "frontier_cs": "agentic_rl.environment.frontiercs:FrontierCsEnv",
    "swerebench": "agentic_rl.environment.swerebench:SweRebenchEnv",
}

_ENV_CACHE: dict[str, RolloutEnv] = {}


def load_env(spec: str | None = None) -> RolloutEnv:
    """Resolve ``spec`` (a row's ``metadata.task_type``) to a cached env. Accepts
    a registry short name or ``pkg.module:Class``; absent -> ``DEFAULT_ENV``."""
    spec = spec or DEFAULT_ENV
    if spec in _ENV_CACHE:
        return _ENV_CACHE[spec]
    target = ENVS.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"unknown task_type {spec!r}; known: {sorted(ENVS)} (or 'pkg.module:Class')")
    module_path, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module_path), attr, None)
    if not (isinstance(cls, type) and issubclass(cls, RolloutEnv)):
        raise TypeError(f"task_type {spec!r} resolved to {cls!r}, not a RolloutEnv subclass")
    env = _ENV_CACHE[spec] = cls()
    return env
