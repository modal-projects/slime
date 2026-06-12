"""mini-swe-agent runtime (the default AgentRuntime).

The contract + shared machinery (detached launch/poll, idempotent
provisioning) live in ``agent/base.py``; the generic rollout recipe lives in
``async_rl_research.generate`` and the per-task-family envs in
``async_rl_research.environment``. By the time ``run_agent`` is called the workspace
is already task-prepped (the active env applied its setup and wrote
``PROBLEM_FILE``). Here we own only what is unique to mini-swe-agent:

    * which adapter speaks this agent's wire protocol (mini-swe-agent talks
      to litellm's OpenAI-compatible API, so we intercept with OpenAIAdapter)
    * the in-sandbox **headless runner** (``MINI_RUNNER_PY``) -- stock
      mini-swe-agent (LitellmModel + LocalEnvironment + DefaultAgent) wired so
      every model call dials back to the slime adapter
    * the isolated uv-venv provisioning spec (a standalone py3.11 +
      ``mini-swe-agent`` that never touches the image's testbed conda env)
    * the launch env/command wiring (litellm env vars, ``-P`` safe path)

Token capture + loss masking happen entirely host-side in the adapter; the
runner is "dumb" and never sees token ids. mini-swe-agent runs UNMODIFIED at
its public OpenAI boundary.

Design A wire flow per turn::

    in-sandbox: litellm.completion(messages, tools=[BASH_TOOL])
                -> POST {adapter_url}/v1/chat/completions  Bearer <session_id>
    host adapter: render messages -> input_ids -> SGLang /generate
                  (return_logprob) -> record TurnRecord -> OpenAI JSON back
    in-sandbox: run bash tool-call locally -> append observation -> loop

mini-swe-agent v2 drives bash through NATIVE tool-calls, so the adapter MUST
be given the served model's sglang tool-call parser or every response looks
tool-less and the agent format-errors in a loop. Set the matching parsers on
the launcher, e.g. ``--sglang-tool-call-parser qwen25`` (Qwen3 emits
hermes-style ``<tool_call>`` JSON) and ``--sglang-reasoning-parser qwen3``.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from slime.agent.adapters import OpenAIAdapter
from slime.agent.sandbox import Sandbox

from .base import AgentRuntime

# Task-layer constant: the active env writes the problem statement here
# before run_agent is called; the runner reads it via MSWE_PROBLEM_FILE.
from ..environment.base import PROBLEM_FILE


# --- mini-swe-agent-specific knobs ------------------------------------------
MSWE_STEP_LIMIT = int(os.environ.get("MSWE_STEP_LIMIT", "50"))
# Which YAML config (prompts!) the runner loads. The DEFAULT is the repo-owned
# universal config below, uploaded into the sandbox -- ONE prompt scaffold for
# all task families; the task-specific deliverable lives in the instruction
# text the env writes (see config/universal.yaml's scope rule). Override
# ladder (both name a BUILTIN config relative to the package's config dir,
# e.g. "mini.yaml" or "benchmarks/swebench.yaml"):
#   MSWE_CONFIG env (global, experiments) > metadata.agent_config (per-row)
#   > the uploaded universal config.
MSWE_CONFIG = os.environ.get("MSWE_CONFIG", "")
# Read at import: the prompt scaffold ships with this module and must be
# identical for every rollout in a run.
UNIVERSAL_CONFIG_YAML = (Path(__file__).parent / "config" / "universal.yaml").read_text(encoding="utf-8")
# Exact-pinned: the scaffold's prompts + wire protocol are part of the RL task
# distribution (a PyPI drift mid-experiment silently changes the environment),
# and MINI_RUNNER_PY below is written against the v2 API.
MSWE_PIP_SPEC = os.environ.get("MSWE_PIP_SPEC", "mini-swe-agent==2.3.1")
# Prepended to PATH for the agent's bash commands (runner keeps only the dirs
# that exist in the image). LocalEnvironment runs commands via /bin/sh, so the
# swebench images' bashrc-based `conda activate testbed` never fires; putting
# the testbed env's bin first is how its python/pytest win.
MSWE_PATH_PREPEND = os.environ.get(
    "MSWE_PATH_PREPEND", "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin"
)
# The agent runs in an isolated venv so the testbed conda env (often pinned to
# an old python) is never used or clobbered. Provisioned at boot with uv; can be
# pre-baked into a derived image (presence of the venv interpreter + matching
# spec marker skips it).
MSWE_AGENT_VENV = os.environ.get("MSWE_AGENT_VENV", "/opt/mswe-agent")
MSWE_AGENT_PYTHON_VERSION = os.environ.get("MSWE_AGENT_PYTHON_VERSION", "3.11")

_VENV_PY = f"{MSWE_AGENT_VENV}/bin/python"

# Runtime scratch under workdir beyond the base's launch files (which keep
# their historical .mswe_* names via scratch_prefix below).
_RUNNER = ".mswe_runner.py"
_CONFIG_FILE = ".mswe_config.yaml"


# ---------------------------------------------------------------------------
# Headless in-sandbox runner.
#
# Written against mini-swe-agent v2 (exact-pinned via MSWE_PIP_SPEC). The v2
# specifics this depends on: prompts come from a YAML config in the same
# schema as the packaged ones (we upload the repo-owned universal config;
# builtins remain reachable via the MSWE_CONFIG override ladder), bash is
# driven through NATIVE tool-calls, and cost tracking hard-fails on models
# litellm cannot price (hence cost_tracking="ignore_errors"). The wiring that
# must hold regardless of version: litellm points at the slime adapter
# (OPENAI_API_BASE/OPENAI_API_KEY), and NO sampling knobs reach the request
# body -- the adapter applies the body OVER its per-session defaults, so a
# client-sent temperature would silently turn rollouts greedy (zero-variance
# GRPO groups).
# ---------------------------------------------------------------------------
MINI_RUNNER_PY = r'''"""Headless mini-swe-agent (v2) runner -- runs INSIDE the sandbox (design A)."""
import os
import sys
import traceback
from pathlib import Path

WORKDIR = os.environ["MSWE_WORKDIR"]
MODEL = os.environ.get("MSWE_MODEL", "slime-actor")
STEP_LIMIT = int(os.environ.get("MSWE_STEP_LIMIT", "50"))
PATH_PREPEND = os.environ.get("MSWE_PATH_PREPEND", "")
with open(os.environ["MSWE_PROBLEM_FILE"], encoding="utf-8") as fh:
    TASK = fh.read()

try:
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    # Prompt config: the host uploads the repo-owned UNIVERSAL config next to
    # this runner (MSWE_CONFIG_FILE) -- the default for every task family.
    # MSWE_CONFIG, when set (global env override or a row's agent_config),
    # names a BUILTIN packaged config instead. Read the builtin path directly
    # -- get_config_from_spec() would also try cwd-relative candidates, which
    # a repo file could shadow.
    cfg_path = Path(os.environ["MSWE_CONFIG_FILE"])
    builtin = os.environ.get("MSWE_CONFIG", "")
    if builtin:
        candidate = builtin_config_dir / builtin
        if candidate.is_file():
            cfg_path = candidate
        else:
            print("[runner] builtin config %s not found; using the universal config" % candidate)
    cfg = yaml.safe_load(cfg_path.read_text())
    agent_cfg = dict(cfg.get("agent") or {})
    model_cfg = dict(cfg.get("model") or {})
    env_cfg = dict(cfg.get("environment") or {})

    # api_base / api_key come from OPENAI_API_BASE / OPENAI_API_KEY in the env
    # (litellm's openai provider reads them). The bundled config pins
    # temperature=0.0 for benchmarking -- strip all sampling knobs so the
    # adapter's per-session defaults (training's temperature) stay in force.
    model_kwargs = dict(model_cfg.get("model_kwargs") or {})
    model_kwargs.pop("temperature", None)
    model_kwargs.pop("top_p", None)
    model_cfg.update(
        model_name="openai/" + MODEL,
        model_kwargs=model_kwargs,
        # "openai/slime-actor" has no litellm price entry; the default mode
        # would raise on the first successful completion.
        cost_tracking="ignore_errors",
    )
    agent_cfg.update(step_limit=STEP_LIMIT, cost_limit=0.0)  # 0 disables the cost check

    # The agent's bash commands run via /bin/sh (dash on these images), so the
    # config's BASH_ENV-based conda activation never fires; prepend the testbed
    # env's bin dirs onto PATH instead. config.env wins over os.environ.
    env_overrides = dict(env_cfg.get("env") or {})
    prepend = [p for p in PATH_PREPEND.split(":") if p and os.path.isdir(p)]
    if prepend:
        env_overrides["PATH"] = ":".join(prepend) + ":" + os.environ.get("PATH", "")

    model = LitellmModel(**model_cfg)
    env = LocalEnvironment(cwd=WORKDIR, env=env_overrides, timeout=int(env_cfg.get("timeout") or 60))
    agent = DefaultAgent(model, env, **agent_cfg)
    info = agent.run(TASK)
    print("[runner] exit_status=%s" % info.get("exit_status"))
    sys.exit(0)
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


# Provision mini-swe-agent into an isolated py3.11 uv venv. uv resolves and
# installs a standalone interpreter, so we don't depend on the image shipping
# py3.11 (prefer a baked uv; fall back to the astral installer). The base
# writes the spec marker only after this script AND the import check succeed.
_VENV_SETUP = (
    'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\n'
    "if ! command -v uv >/dev/null 2>&1; then\n"
    "  if command -v curl >/dev/null 2>&1; then\n"
    "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
    '    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\n'
    "  else\n"
    "    # slim images (e.g. harbor's python:*-slim) ship neither curl nor\n"
    "    # wget, but they do ship pip; uv's PyPI wheel lands on PATH.\n"
    "    python3 -m pip install --quiet uv\n"
    "  fi\n"
    "fi\n"
    f"rm -rf {shlex.quote(MSWE_AGENT_VENV)}\n"
    f"uv venv --python {shlex.quote(MSWE_AGENT_PYTHON_VERSION)} {shlex.quote(MSWE_AGENT_VENV)}\n"
    f"uv pip install --python {shlex.quote(_VENV_PY)} {shlex.quote(MSWE_PIP_SPEC)}\n"
)

# MSWEA_SILENT_STARTUP suppresses the import-time banner that would otherwise
# corrupt the provisioning probe's marker comparison (and litter the run log).
_VENV_CHECK = f"MSWEA_SILENT_STARTUP=1 {shlex.quote(_VENV_PY)} -c 'import minisweagent'"


class MiniSweAgentRuntime(AgentRuntime):
    name = "mini-swe"
    adapter_cls = OpenAIAdapter
    # Advertised to litellm as "openai/<model_name>". The adapter ignores the
    # name (it routes to the SGLang-served actor); litellm only needs the
    # provider prefix so it speaks the OpenAI dialect at our adapter_url.
    model_name = "slime-actor"
    # Keep the historical scratch names (.mswe_run.sh / .mswe_done / .mswe_log).
    scratch_prefix = ".mswe"
    # "patch.txt" is the submission artifact the builtin swebench prompt
    # instructs the agent to create -- the universal config doesn't, but an
    # MSWE_CONFIG/agent_config override back to the builtin would, and
    # `git add -N .` would sweep it into the diff. (Launch scratch + the
    # task-layer PROBLEM_FILE are excluded by the base / by the swe_gym env's
    # git_diff itself.)
    diff_exclude = (_RUNNER, _CONFIG_FILE, "patch.txt")

    async def run_agent(
        self,
        sb: Sandbox,
        *,
        md: dict,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
    ) -> int:
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
            # litellm's openai provider reads these for base URL + bearer auth.
            "OPENAI_API_BASE": base,
            "OPENAI_BASE_URL": base,
            "OPENAI_API_KEY": session_id,
            "MSWE_MODEL": self.model_name,
            "MSWE_WORKDIR": workdir,
            "MSWE_PROBLEM_FILE": f"{workdir}/{PROBLEM_FILE}",
            # Override ladder: global env > per-row builtin override > the
            # uploaded universal config ("" -> the runner uses MSWE_CONFIG_FILE).
            "MSWE_CONFIG": MSWE_CONFIG or md.get("agent_config") or "",
            "MSWE_CONFIG_FILE": f"{workdir}/{_CONFIG_FILE}",
            "MSWE_STEP_LIMIT": str(MSWE_STEP_LIMIT),
            "MSWE_PATH_PREPEND": MSWE_PATH_PREPEND,
            # keep the v2 import-time banner out of the runner log.
            "MSWEA_SILENT_STARTUP": "1",
        }
        # Run the agent with the ISOLATED venv interpreter. The runner still
        # cd's into workdir + uses LocalEnvironment, so the agent's bash
        # tool-calls (tests, git) run against the repo -- with the testbed
        # env's bin on PATH (MSWE_PATH_PREPEND) -- only the agent process
        # itself is isolated. -P (safe path, py3.11+) keeps the script dir
        # (= workdir) off sys.path: a repo that shares a name with an agent
        # dep (e.g. the pydantic instances) would otherwise shadow the venv's
        # copy and crash the runner at import time, before any model call.
        return await self._detached_run(
            sb,
            workdir=workdir,
            command=f"{shlex.quote(_VENV_PY)} -P {shlex.quote(_RUNNER)}",
            env=env,
            time_budget_sec=time_budget_sec,
            log_tag=f"session={session_id}",
        )


# Module export for dotted-module-path loading (legacy ASYNC_RL_AGENT_DRIVER
# configs pass "async_rl_research.agent.mini_swe_agent"; see base.load_runtime).
RUNTIME = MiniSweAgentRuntime
