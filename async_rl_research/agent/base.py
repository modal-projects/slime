"""AgentRuntime: the contract between ``generate.py`` and one agent framework.

A *runtime* packages everything specific to one in-sandbox agent framework
(mini-swe-agent, opencode, pi, ...): which slime adapter speaks its wire
protocol, how to provision the agent inside the work sandbox, and how to
launch it. Everything else -- adapter/HTTP lifecycle, task workspace prep,
grading, trajectory merge -- is the generic recipe in ``generate.py`` plus
the active task env (``env/base.py``) and never changes per agent.

The shared launch machinery lives HERE (not in a separate module) on purpose:
``_detached_run`` creates scratch files under ``workdir`` (launcher script,
done marker, log), and the entity that creates scratch must be the entity
that excludes it from the captured diff. ``diff_exclude_all`` = the base's
launch scratch + the subclass's ``diff_exclude``, so "forgot to exclude the
launcher" is structurally impossible.

Writing a new runtime
---------------------
Subclass, declare the class attributes, implement ``run_agent`` by composing
the two helpers. Sketch of an opencode-style runtime::

    class OpenCodeRuntime(AgentRuntime):
        name = "opencode"
        adapter_cls = OpenAIAdapter        # or AnthropicAdapter
        diff_exclude = ("opencode.json",)  # extra scratch beyond launch files

        async def run_agent(self, sb, *, md, session_id, adapter_url, time_budget_sec):
            await self._ensure_provisioned(sb, spec=..., marker_path=..., setup_script=...)
            await sb.write_file(f"{md['workdir']}/opencode.json", _config(adapter_url))
            return await self._detached_run(
                sb, workdir=md["workdir"],
                command="opencode run ...",
                env={"OPENAI_API_KEY": session_id},
                time_budget_sec=time_budget_sec,
            )

Then register it in ``RUNTIMES`` below.

On-policy rule every runtime must respect: the adapter applies the request
body OVER its per-session sampling defaults, so the agent must NOT send its
own temperature/top_p (a client-sent temperature silently turns rollouts
greedy -> zero-variance GRPO groups). Strip sampling knobs at the agent's
config layer (see mini_swe_agent's runner for the pattern).

Runtimes are instantiated once per rollout worker (held by
``generate._State``) and must be stateless across samples -- ``run_agent``
receives everything per-call.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import shlex
import time
from abc import ABC, abstractmethod
from typing import ClassVar

from slime.agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


# run_agent return convention: the agent's exit code, or this when the
# wallclock budget elapsed before the done marker appeared.
EXIT_BUDGET_EXCEEDED = -2


class AgentRuntime(ABC):
    """One agent framework's integration: wire adapter + provision + launch.

    Required class attributes (validated at class-definition time):

        name          registry key / log prefix, e.g. "mini-swe"
        adapter_cls   slime adapter class for the agent's wire protocol
                      (OpenAIAdapter / AnthropicAdapter). generate._State
                      constructs it as adapter_cls(tokenizer=, sglang_url=,
                      tool_parser=, reasoning_parser=).

    Optional class attributes:

        model_name      model name advertised to the agent ("slime-actor");
                        the adapter ignores it (routes to the served actor) --
                        clients only need it to pick the right API dialect.
        scratch_prefix  prefix for the launch scratch files _detached_run
                        writes under workdir (".agent" -> .agent_run.sh /
                        .agent_done / .agent_log).
        diff_exclude    EXTRA scratch files the runtime writes under workdir
                        (runner scripts, configs, prompt-convention artifacts
                        like mini-swe's "patch.txt"). Launch scratch is
                        excluded automatically -- do not repeat it here.

    ``generate.py`` only ever touches: ``adapter_cls``, ``run_agent``,
    ``diff_exclude_all``, ``name``.
    """

    name: ClassVar[str]
    adapter_cls: ClassVar[type]
    model_name: ClassVar[str] = "slime-actor"
    scratch_prefix: ClassVar[str] = ".agent"
    diff_exclude: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        # Fail at import time, not mid-rollout: a runtime missing its
        # declarations should never make it into a training run.
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
    ) -> int:
        """Provision + launch the agent in the already-booted, task-prepped sandbox.

        The workspace is ready (the active env applied its setup and wrote
        ``PROBLEM_FILE``) -- implementations only set up their own agent. The
        agent must target ``adapter_url`` for model calls and send
        ``session_id`` as its auth/bearer so the adapter groups its turns.
        ``md`` is the env-normalized dataset row (``RolloutEnv
        .normalize_metadata``); may be called multiple times per sample in the
        SAME sandbox (multi-step episodes). Returns the agent's exit code or
        ``EXIT_BUDGET_EXCEEDED``.
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
    ) -> int:
        """Launch ``command`` detached in ``workdir`` and poll a done-marker.

        Writes a launcher script (cd + ``env`` exports + command, stdout/err
        to the log file, exit code to the done marker) and starts it in its
        own session (``setsid ... &``) so the exec RPC returns immediately
        rather than streaming a multi-minute foreground command. This is NOT
        Modal's ``Sandbox.detach()`` -- the sandbox object stays live.

        We avoid a long-lived foreground exec because a worker stream reset
        mid-run would be classified transient and re-launch the whole agent
        (see ModalSandbox._is_transient / _retry); the poll RPCs are short and
        idempotent, so a dropped poll just retries (and each one counts as
        sandbox activity if an idle_timeout is configured). On nonzero exit
        the log tail is surfaced host-side so an in-sandbox crash is never
        just an opaque exit code.

        ``command`` is spliced into the script verbatim -- the caller quotes
        its own paths (shlex.quote). Returns the command's exit code, or
        ``EXIT_BUDGET_EXCEEDED`` if ``time_budget_sec`` elapses first.
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
        # rm the done marker BEFORE launching: a multi-leg episode (e.g. the
        # harbor env's multi-step tasks) relaunches in the same sandbox, and a
        # stale marker from the previous leg would satisfy the first poll
        # while the new agent is still running.
        await sb.exec(f"cd {q(workdir)} && rm -f {q(done)} && chmod +x {q(launch)}", check=False, timeout=30)
        # Detach so the exec RPC returns immediately; the marker file is the signal.
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
        if exit_code != 0:
            _, tail, _ = await sb.exec(f"tail -c 4000 {q(f'{workdir}/{log}')} 2>/dev/null", check=False, timeout=15)
            if (tail or "").strip():
                logger.warning("[%s] %s exit=%s %s tail:\n%s", self.name, log_tag, exit_code, log, tail.strip())
        logger.info("[%s] %s exit=%s elapsed<=%ds", self.name, log_tag, exit_code, time_budget_sec)
        return exit_code

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

        If ``marker_path`` already holds exactly ``spec`` (and ``check_cmd``,
        when given, exits 0), the install is skipped -- this is what lets a
        pre-baked derived image (or a previous boot) short-circuit, while a
        changed pin rebuilds stale pre-baked toolchains instead of silently
        running the old agent. Otherwise ``setup_script`` runs (bash, checked),
        ``check_cmd`` re-verifies the result, and the marker is written LAST so
        a half-finished install is never mistaken for a complete one.

        Returns True if provisioning ran, False on a marker hit. Setup
        typically needs outbound network, which the default
        ``block_network=False`` allows.
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

# Short name -> "module:Class". Values are strings (not classes) so importing
# base.py never imports any runtime module.
RUNTIMES: dict[str, str] = {
    "mini-swe": "async_rl_research.agent.mini_swe_agent:MiniSweAgentRuntime",
}


def load_runtime(spec: str | None = None) -> AgentRuntime:
    """Resolve ``spec`` to an AgentRuntime instance, validating eagerly.

    Accepted forms:
      * registry short name              "mini-swe"
      * explicit class                   "pkg.module:ClassName"
      * module path exposing RUNTIME     "pkg.module"   (legacy driver form;
        what existing ASYNC_RL_AGENT_DRIVER configs pass)
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
