"""mini-swe-agent runtime (the default AgentRuntime).

Runs stock mini-swe-agent (v2) headless inside the sandbox in an isolated
uv-venv, with every model call dialing back to the slime adapter over litellm's
OpenAI-compatible API. Token capture + loss masking happen host-side; the
runner never sees token ids. The adapter MUST use the served model's sglang
tool-call parser (v2 drives bash via native tool-calls), e.g.
``--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3``.

The in-sandbox programs live in sibling files so they stay lint/format-able:
``mini_swe_runner.py`` (the headless runner) and ``config/venv_setup.sh`` (uv
provisioning); both are read at import and written into the sandbox at rollout.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from ..environment.base import PROBLEM_FILE

# Renders tool-call arguments as a dict so Qwen3.6's qwen3_coder chat template
# doesn't crash on turn 2+ (safe for hermes-style Qwen3 too).
from .adapters import QwenOpenAIAdapter
from .base import AgentRunResult, AgentRuntime

_HERE = Path(__file__).parent

MSWE_STEP_LIMIT = int(os.environ.get("MSWE_STEP_LIMIT", "50"))
# Consecutive no-tool-call model turns before the runner ends the episode: a
# stuck model that never reaches the context wall would otherwise format-error
# its way to MSWE_STEP_LIMIT. See _StopAwareModel in mini_swe_runner.py.
MSWE_MAX_EMPTY_TURNS = int(os.environ.get("MSWE_MAX_EMPTY_TURNS", "3"))
# Which YAML config (prompts) the runner loads. Override ladder: MSWE_CONFIG
# env (global) > metadata.agent_config (per-row) > universal config below.
MSWE_CONFIG = os.environ.get("MSWE_CONFIG", "")
# Read at import: the scaffold must be identical for every rollout in a run.
UNIVERSAL_CONFIG_YAML = (_HERE / "config" / "universal.yaml").read_text(encoding="utf-8")
# Exact-pinned: prompts + wire protocol are part of the RL task distribution,
# and mini_swe_runner.py is written against the v2 API.
MSWE_PIP_SPEC = os.environ.get("MSWE_PIP_SPEC", "mini-swe-agent==2.3.1")
# Prepended to PATH for the agent's bash commands: LocalEnvironment runs via
# /bin/sh so `conda activate testbed` never fires; this is how its python wins.
MSWE_PATH_PREPEND = os.environ.get("MSWE_PATH_PREPEND", "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin")
# Isolated venv so the testbed conda env is never used or clobbered. Provisioned
# at boot with uv; can be pre-baked into a derived image.
MSWE_AGENT_VENV = os.environ.get("MSWE_AGENT_VENV", "/opt/mswe-agent")
MSWE_AGENT_PYTHON_VERSION = os.environ.get("MSWE_AGENT_PYTHON_VERSION", "3.11")

_VENV_PY = f"{MSWE_AGENT_VENV}/bin/python"

_RUNNER = ".mswe_runner.py"
_CONFIG_FILE = ".mswe_config.yaml"


# Headless in-sandbox runner (mini-swe-agent v2, exact-pinned). NO sampling
# knobs reach the request body -- the adapter applies them OVER its per-session
# defaults, so a client-sent temperature would silently turn rollouts greedy.
MINI_RUNNER_PY = (_HERE / "mini_swe_runner.py").read_text(encoding="utf-8")


# uv-venv provisioning script. The host exports the resolved config in front of
# config/venv_setup.sh so operator env overrides (MSWE_AGENT_VENV, ...) still
# propagate into the sandbox; the script's own defaults keep it runnable
# standalone. See the script header + profiles/PROVISIONING_VENV_VS_VOLUME.md.
_VENV_SETUP = (
    f"export MSWE_AGENT_VENV={shlex.quote(MSWE_AGENT_VENV)}\n"
    f"export MSWE_AGENT_PYTHON_VERSION={shlex.quote(MSWE_AGENT_PYTHON_VERSION)}\n"
    f"export MSWE_PIP_SPEC={shlex.quote(MSWE_PIP_SPEC)}\n"
) + (_HERE / "config" / "venv_setup.sh").read_text(encoding="utf-8")

# MSWEA_SILENT_STARTUP suppresses the import-time banner that would otherwise
# corrupt the provisioning probe's marker comparison. Import the agent's real
# entrypoint (not just the top package) so the probe also rejects a pre-baked
# venv whose pydantic_core native module is missing -> re-provision instead of
# launching a doomed agent. `-P` keeps the image WORKDIR (e.g. /testbed) off
# sys.path so a repo named like an agent dep (pydantic tasks) can't shadow the
# venv and fail the probe; see config/venv_setup.sh and the runner launch.
_VENV_CHECK = f"MSWEA_SILENT_STARTUP=1 {shlex.quote(_VENV_PY)} -P -c 'import minisweagent.agents.default'"


class MiniSweAgentRuntime(AgentRuntime):
    name = "mini-swe"
    adapter_cls = QwenOpenAIAdapter
    # model_name inherits AgentRuntime's "slime-actor" default.
    scratch_prefix = ".mswe"
    # "patch.txt": submission artifact the builtin swebench prompt tells the
    # agent to create, which `git add -N .` would otherwise sweep into the diff.
    diff_exclude = (_RUNNER, _CONFIG_FILE, "patch.txt")

    async def run_agent(
        self,
        sb: Sandbox,
        *,
        md: dict,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
    ) -> AgentRunResult:
        """Provision mini-swe-agent in ``sb``, run it on the task, poll to done."""
        workdir = md["workdir"]
        await sb.write_file(f"{workdir}/{_RUNNER}", MINI_RUNNER_PY)
        await sb.write_file(f"{workdir}/{_CONFIG_FILE}", UNIVERSAL_CONFIG_YAML)
        await self._ensure_provisioned(
            sb,
            spec=MSWE_PIP_SPEC,
            marker_path=f"{MSWE_AGENT_VENV}/.mswe_spec",
            setup_script=_VENV_SETUP,
            check_cmd=_VENV_CHECK,
        )

        base = f"{adapter_url}/v1"
        env = {
            "OPENAI_API_BASE": base,
            "OPENAI_BASE_URL": base,
            "OPENAI_API_KEY": session_id,
            "MSWE_MODEL": self.model_name,
            "MSWE_WORKDIR": workdir,
            "MSWE_PROBLEM_FILE": f"{workdir}/{PROBLEM_FILE}",
            # Override ladder: global env > per-row builtin > universal config.
            "MSWE_CONFIG": MSWE_CONFIG or md.get("agent_config") or "",
            "MSWE_CONFIG_FILE": f"{workdir}/{_CONFIG_FILE}",
            "MSWE_STEP_LIMIT": str(MSWE_STEP_LIMIT),
            "MSWE_MAX_EMPTY_TURNS": str(MSWE_MAX_EMPTY_TURNS),
            "MSWE_PATH_PREPEND": MSWE_PATH_PREPEND,
            "MSWEA_SILENT_STARTUP": "1",
        }
        # Run with the ISOLATED venv interpreter; -P keeps the workdir off
        # sys.path so a repo sharing a name with an agent dep can't shadow it.
        return await self._detached_run(
            sb,
            workdir=workdir,
            command=f"{shlex.quote(_VENV_PY)} -P {shlex.quote(_RUNNER)}",
            env=env,
            time_budget_sec=time_budget_sec,
            log_tag=f"session={session_id}",
        )


# Module export for dotted-module-path loading (see load_runtime).
RUNTIME = MiniSweAgentRuntime
