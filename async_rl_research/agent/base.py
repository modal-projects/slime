"""AgentRuntime: the contract between ``generate.py`` and one agent framework.

A *runtime* packages everything specific to one in-sandbox agent framework
(mini-swe-agent, opencode, ...): its slime adapter, provisioning, and launch.
Subclass, declare the class attributes, implement ``run_agent`` by composing
``_ensure_provisioned`` + ``_detached_run``, and register in ``RUNTIMES`` below.

On-policy rule: the adapter applies the request body OVER its per-session
sampling defaults, so a runtime must strip the agent's own temperature/top_p
(a client-sent temperature silently turns rollouts greedy). Runtimes are
instantiated once per worker and must be stateless across samples.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import shlex
import time
from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

from slime.agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


# run_agent return value when the wallclock budget elapsed before the done
# marker appeared (otherwise run_agent returns the agent's exit code).
EXIT_BUDGET_EXCEEDED = -2


class AgentRunResult(NamedTuple):
    """Outcome of one agent leg: the process ``exit_code`` (or
    ``EXIT_BUDGET_EXCEEDED``) plus, on a nonzero exit, the last few KB of the
    agent's stdout/stderr (``tail``; empty on a clean exit).

    Persisted into sample metadata so a zero-turn ``adapter_session_empty``
    self-explains in the dump (e.g. exit=137 -> OOM-killed) instead of relying
    on tail-only Modal logs that age out once the run finishes.
    """

    exit_code: int
    tail: str = ""


class AgentRuntime(ABC):
    """One agent framework's integration: wire adapter + provision + launch.

    Required attributes (validated at class-definition time): ``name`` (registry
    key / log prefix) and ``adapter_cls`` (slime adapter for the wire protocol).
    Optional: ``model_name`` (advertised to the agent), ``scratch_prefix``
    (launch scratch prefix under workdir), ``diff_exclude`` (extra scratch to
    drop from the diff; launch scratch is excluded automatically).
    """

    name: ClassVar[str]
    adapter_cls: ClassVar[type]
    model_name: ClassVar[str] = "slime-actor"
    scratch_prefix: ClassVar[str] = ".agent"
    diff_exclude: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        # Fail at import time, not mid-rollout, on missing declarations.
        super().__init_subclass__(**kwargs)
        missing = [a for a in ("name", "adapter_cls") if getattr(cls, a, None) is None]
        if missing:
            raise TypeError(f"{cls.__name__} must define class attribute(s) {missing!r} (see AgentRuntime)")

    @property
    def diff_exclude_all(self) -> tuple[str, ...]:
        """Everything to drop from the captured diff: launch scratch + extras."""
        return (*self._launch_scratch_files(), *self.diff_exclude)

    @abstractmethod
    async def run_agent(
        self,
        sb: Sandbox,
        *,
        md: dict,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
    ) -> AgentRunResult:
        """Provision + launch the agent in the booted, task-prepped sandbox.

        The agent must call ``adapter_url`` with ``session_id`` as its bearer so
        the adapter groups its turns. May be called multiple times per sample in
        the SAME sandbox (multi-step). Returns an ``AgentRunResult`` (exit code,
        or ``EXIT_BUDGET_EXCEEDED``, plus a failure-only log tail).
        """

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------
    def _launch_scratch_files(self) -> tuple[str, str, str]:
        """(launcher script, done marker, log) names under workdir."""
        p = self.scratch_prefix
        return (f"{p}_run.sh", f"{p}_done", f"{p}_log")

    async def _detached_run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        command: str,
        env: dict[str, str] | None = None,
        time_budget_sec: int,
        poll_interval_sec: float = 5.0,
        log_tag: str = "",
    ) -> AgentRunResult:
        """Launch ``command`` detached in ``workdir`` (``setsid ... &``) and poll
        a done-marker, so a foreground exec stream reset can't re-launch the
        whole agent. ``command`` is spliced in verbatim (caller quotes paths).
        Returns an ``AgentRunResult``: the exit code (or ``EXIT_BUDGET_EXCEEDED``
        if ``time_budget_sec`` elapses first) plus, on a nonzero exit, the last
        4 KB of the agent log (surfaced host-side AND returned for the dump).
        """
        q = shlex.quote
        launch, done, log = self._launch_scratch_files()
        exports = "".join(f"export {k}={q(str(v))}\n" for k, v in (env or {}).items())
        launcher_body = (
            "#!/bin/bash\n"
            f"cd {q(workdir)}\n"
            f"{exports}"
            f"{command} > {q(log)} 2>&1\n"
            f"echo $? > {q(done)}\n"
        )
        await sb.write_file(f"{workdir}/{launch}", launcher_body)
        # rm the done marker BEFORE launching: a stale marker from a prior leg
        # would satisfy the first poll while the new agent still runs.
        await sb.exec(f"cd {q(workdir)} && rm -f {q(done)} && chmod +x {q(launch)}", check=False, timeout=30)
        await sb.exec(
            f"cd {q(workdir)} && setsid bash {q(launch)} < /dev/null > /dev/null 2>&1 &",
            check=False,
            timeout=30,
        )

        done_path = f"{workdir}/{done}"
        deadline = time.time() + time_budget_sec
        exit_code = EXIT_BUDGET_EXCEEDED
        while time.time() < deadline:
            await asyncio.sleep(poll_interval_sec)
            ec, out, _ = await sb.exec(f"test -f {q(done_path)} && cat {q(done_path)}", check=False, timeout=15)
            if ec == 0:
                try:
                    exit_code = int((out or "").strip() or "-1")
                except ValueError:
                    exit_code = -1
                break
        tail = ""
        if exit_code != 0:
            _, raw_tail, _ = await sb.exec(f"tail -c 4000 {q(f'{workdir}/{log}')} 2>/dev/null", check=False, timeout=15)
            tail = (raw_tail or "").strip()
            if tail:
                logger.warning("[%s] %s exit=%s %s tail:\n%s", self.name, log_tag, exit_code, log, tail)
        logger.info("[%s] %s exit=%s elapsed<=%ds", self.name, log_tag, exit_code, time_budget_sec)
        return AgentRunResult(exit_code, tail)

    async def _ensure_provisioned(
        self,
        sb: Sandbox,
        *,
        spec: str,
        marker_path: str,
        setup_script: str,
        check_cmd: str | None = None,
        timeout: int = 900,
    ) -> bool:
        """Idempotent toolchain install keyed on a spec marker.

        Skips when ``marker_path`` already holds exactly ``spec`` (and
        ``check_cmd`` exits 0), so a pre-baked image short-circuits while a
        changed pin rebuilds. The marker is written LAST so a half-finished
        install is never mistaken for complete. Returns True if it ran.
        """
        q = shlex.quote
        probe = f"cat {q(marker_path)} 2>/dev/null"
        if check_cmd:
            probe = f"({check_cmd}) >/dev/null 2>&1 && " + probe
        _, out, _ = await sb.exec(probe, check=False, timeout=60)
        if (out or "").strip() == spec:
            return False

        logger.info("[%s] provisioning spec=%s in sandbox %s", self.name, spec, sb.sandbox_id[:8])
        await sb.exec(setup_script, check=True, timeout=timeout)
        if check_cmd:
            await sb.exec(check_cmd, check=True, timeout=60)
        await sb.exec(f"printf '%s' {q(spec)} > {q(marker_path)}", check=True, timeout=30)
        return True


# ---------------------------------------------------------------------------
# Registry + loader
# ---------------------------------------------------------------------------
DEFAULT_RUNTIME = "mini-swe"

# Short name -> "module:Class" (strings so importing base.py imports no runtime).
RUNTIMES: dict[str, str] = {
    "mini-swe": "async_rl_research.agent.mini_swe_agent:MiniSweAgentRuntime",
}


def load_runtime(spec: str | None = None) -> AgentRuntime:
    """Resolve ``spec`` to an AgentRuntime instance, validating eagerly.

    Accepts a registry short name ("mini-swe"), "pkg.module:ClassName", or a
    module path exposing ``RUNTIME`` (legacy driver form).
    """
    spec = spec or DEFAULT_RUNTIME
    target = RUNTIMES.get(spec, spec)
    if ":" in target:
        module_path, _, attr = target.partition(":")
    else:
        module_path, attr = target, "RUNTIME"
    module = importlib.import_module(module_path)
    cls = getattr(module, attr, None)
    if cls is None:
        raise ValueError(
            f"agent runtime {spec!r}: module {module_path!r} does not expose {attr!r}; "
            f"known short names: {sorted(RUNTIMES)}"
        )
    if not (isinstance(cls, type) and issubclass(cls, AgentRuntime)):
        raise TypeError(f"agent runtime {spec!r} resolved to {cls!r}, which is not an AgentRuntime subclass")
    return cls()
