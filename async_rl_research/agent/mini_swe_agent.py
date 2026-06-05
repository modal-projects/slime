"""mini-swe-agent driver.

This module is the agent-specific half of the rollout. The generic recipe
(adapter/HTTP lifecycle, trajectory merge, abort isolation, dataset
normalization, sandbox boot / git_diff / evaluate) lives in
``async_rl_research.generate`` and ``async_rl_research.sandbox``. Here we
own only what is unique to mini-swe-agent:

    * ADAPTER_CLS / MODEL_NAME -- which slime adapter speaks this agent's wire
      protocol (mini-swe-agent talks to litellm's OpenAI-compatible API, so we
      intercept with OpenAIAdapter).
    * the in-sandbox **headless runner** (``MINI_RUNNER_PY``) -- stock
      mini-swe-agent (LitellmModel + LocalEnvironment + DefaultAgent) wired so
      every model call dials back to the slime adapter.
    * ``run_agent`` -- provision the package, upload the runner + task, launch
      it detached, and poll a done-marker (sandbox gateways reset long-lived
      HTTP/2 connections, so we cannot hold a multi-minute foreground exec).

Token capture + loss masking happen entirely host-side in the adapter; this
runner is "dumb" and never sees token ids. mini-swe-agent runs UNMODIFIED at
its public OpenAI boundary -- the only requirement on the sandbox image is
python + the ``mini-swe-agent`` package (prefer baking it in; the best-effort
pip install below is a fallback for dev).

Design A wire flow per turn::

    in-sandbox: litellm.completion(messages, tools=[BASH_TOOL])
                -> POST {adapter_url}/v1/chat/completions  Bearer <session_id>
    host adapter: render messages -> input_ids -> SGLang /generate
                  (return_logprob) -> record TurnRecord -> OpenAI JSON back
    in-sandbox: run bash tool-call locally -> append observation -> loop

The served model must support tool-call bash; set the matching SGLang parsers
on the launcher, e.g. ``--sglang-tool-call-parser qwen3_coder`` and
``--sglang-reasoning-parser qwen3``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time

from slime.agent.adapters import OpenAIAdapter
from slime.agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


# --- driver declaration (read by async_rl_research.generate._State) ---------
ADAPTER_CLS = OpenAIAdapter
# Advertised to litellm as "openai/<MODEL_NAME>". The adapter ignores the name
# (it routes to the SGLang-served actor); litellm only needs the provider
# prefix so it speaks the OpenAI dialect at our adapter_url.
MODEL_NAME = "slime-actor"


# --- mini-swe-agent-specific knobs ------------------------------------------
MSWE_STEP_LIMIT = int(os.environ.get("MSWE_STEP_LIMIT", "50"))
# Prefer baking `pip install mini-swe-agent==<pin>` into the sandbox image. If
# MSWE_PIP_INSTALL=1, run_agent will best-effort install it at boot (needs the
# sandbox to have outbound PyPI access).
MSWE_PIP_INSTALL = os.environ.get("MSWE_PIP_INSTALL", "0") == "1"
MSWE_PIP_SPEC = os.environ.get("MSWE_PIP_SPEC", "mini-swe-agent")

# Sandbox paths (kept under workdir; excluded from the captured diff).
_RUNNER = ".mswe_runner.py"
_LAUNCH = ".mswe_run.sh"
_DONE = ".mswe_done"
_LOG = ".mswe_log"
_PROBLEM = "PROBLEM_STATEMENT.md"


# ---------------------------------------------------------------------------
# Headless in-sandbox runner.
#
# NOTE: PIN + VERIFY the imports / kwargs against the mini-swe-agent version
# baked into the image -- class names and config kwargs have shifted across
# releases. The wiring that must hold regardless: litellm points at the slime
# adapter (OPENAI_API_BASE/OPENAI_API_KEY), and we DO NOT set temperature here
# (the adapter's per-session sampling defaults must win, keeping RL on-policy).
# ---------------------------------------------------------------------------
MINI_RUNNER_PY = r'''"""Headless mini-swe-agent runner -- runs INSIDE the sandbox (design A)."""
import os
import sys
import traceback

WORKDIR = os.environ["MSWE_WORKDIR"]
MODEL = os.environ.get("MSWE_MODEL", "slime-actor")
STEP_LIMIT = int(os.environ.get("MSWE_STEP_LIMIT", "50"))
with open(os.environ["MSWE_PROBLEM_FILE"], encoding="utf-8") as fh:
    TASK = fh.read()

try:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    # api_base / api_key come from OPENAI_API_BASE / OPENAI_API_KEY in the env
    # (litellm's openai provider reads them). No temperature/top_p here.
    model = LitellmModel(model_name="openai/" + MODEL)
    env = LocalEnvironment(cwd=WORKDIR)
    # cost_limit=0 disables cost tracking (meaningless against a local actor).
    agent = DefaultAgent(model, env, step_limit=STEP_LIMIT, cost_limit=0.0)
    agent.run(TASK)
    sys.exit(0)
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


# ---------------------------------------------------------------------------
# run_agent: provision + launch + poll (the only entrypoint generate.py calls)
# ---------------------------------------------------------------------------
async def run_agent(
    sb: Sandbox,
    *,
    md: dict,
    session_id: str,
    adapter_url: str,
    time_budget_sec: int,
) -> int:
    """Provision mini-swe-agent in ``sb``, run it on the task, poll to done.

    Returns the runner's exit code, or ``-2`` if the wallclock budget elapses
    first. The agent dials back to ``adapter_url`` for every model call and
    authenticates with ``session_id`` so the adapter groups its turns.
    """
    workdir = md["workdir"]
    await _prepare_workspace(sb, workdir, md)
    await _ensure_installed(sb)
    return await _launch_and_poll(
        sb,
        workdir=workdir,
        session_id=session_id,
        adapter_url=adapter_url,
        time_budget_sec=time_budget_sec,
    )


async def _prepare_workspace(sb: Sandbox, workdir: str, md: dict) -> None:
    # git operations inside the sandbox need the repo marked safe; the diff is
    # captured by sandbox.git_diff later.
    await sb.exec("git config --system --add safe.directory '*'", check=False, timeout=60)
    if md.get("pre_commands"):
        await _apply_pre_commands(sb, workdir, md["pre_commands"])
    await sb.write_file(f"{workdir}/{_PROBLEM}", md.get("problem_statement") or "")
    await sb.write_file(f"{workdir}/{_RUNNER}", MINI_RUNNER_PY)


async def _apply_pre_commands(sb: Sandbox, workdir: str, pre) -> None:
    # Keep the work sandbox baseline aligned with eval (sweb-style pre_commands
    # are typically `git checkout <base_sha> -f`); skipping them makes the
    # model's diff context mismatch the eval base -> apply failures.
    body = pre.replace("\\n", "\n") if isinstance(pre, str) else "\n".join(c for c in (pre or []) if c)
    await sb.write_file(f"{workdir}/.mswe_pre.sh", "set -e\n" + body)
    await sb.exec(f"cd {shlex.quote(workdir)} && bash .mswe_pre.sh", check=False, timeout=600)


async def _ensure_installed(sb: Sandbox) -> None:
    ec, out, _ = await sb.exec(
        'python -c "import minisweagent" 2>/dev/null && echo MSWE_OK', check=False, timeout=60
    )
    if "MSWE_OK" in (out or ""):
        return
    if not MSWE_PIP_INSTALL:
        raise RuntimeError(
            "mini-swe-agent is not installed in the sandbox image. Bake "
            f"`pip install {MSWE_PIP_SPEC}` into the image, or set "
            "MSWE_PIP_INSTALL=1 to install at boot (needs outbound PyPI)."
        )
    logger.info("[mini_swe_agent] installing %s in sandbox %s", MSWE_PIP_SPEC, sb.sandbox_id[:8])
    await sb.exec(f"pip install --no-input {shlex.quote(MSWE_PIP_SPEC)}", check=True, timeout=600)


async def _launch_and_poll(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    adapter_url: str,
    time_budget_sec: int,
) -> int:
    """Launch the runner detached + poll a done-marker file.

    Sandbox gateways reset HTTP/2 around ~6.5 min, so we cannot keep a
    long-lived foreground exec. The launcher writes the exit code into a marker
    file; we poll it every 5s via short RPCs (which also keeps the sandbox
    alive against idle GC).
    """
    q = shlex.quote
    base = q(f"{adapter_url}/v1")
    launcher_body = (
        "#!/bin/bash\n"
        f"cd {q(workdir)}\n"
        # litellm's openai provider reads these for base URL + bearer auth.
        f"export OPENAI_API_BASE={base}\n"
        f"export OPENAI_BASE_URL={base}\n"
        f"export OPENAI_API_KEY={q(session_id)}\n"
        f"export MSWE_MODEL={q(MODEL_NAME)}\n"
        f"export MSWE_WORKDIR={q(workdir)}\n"
        f"export MSWE_PROBLEM_FILE={q(f'{workdir}/{_PROBLEM}')}\n"
        f"export MSWE_STEP_LIMIT={q(str(MSWE_STEP_LIMIT))}\n"
        f"python {q(_RUNNER)} > {q(_LOG)} 2>&1\n"
        f"echo $? > {q(_DONE)}\n"
    )
    await sb.write_file(f"{workdir}/{_LAUNCH}", launcher_body)
    await sb.exec(f"cd {q(workdir)} && chmod +x {q(_LAUNCH)}", check=False, timeout=30)
    # Detach so the exec RPC returns immediately; the marker file is the signal.
    await sb.exec(
        f"cd {q(workdir)} && setsid bash {q(_LAUNCH)} < /dev/null > /dev/null 2>&1 &",
        check=False,
        timeout=30,
    )

    done_path = f"{workdir}/{_DONE}"
    deadline = time.time() + time_budget_sec
    exit_code = -2  # convention: -2 = budget exceeded
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {q(done_path)} && cat {q(done_path)}", check=False, timeout=15)
        if ec == 0:
            try:
                exit_code = int((out or "").strip() or "-1")
            except ValueError:
                exit_code = -1
            break
    logger.info("[mini_swe_agent] session=%s exit=%s elapsed<=%ds", session_id, exit_code, time_budget_sec)
    return exit_code


# sandbox.git_diff should exclude these scratch files from the captured diff:
DIFF_EXCLUDE = (_PROBLEM, _RUNNER, _LAUNCH, _DONE, _LOG, ".mswe_pre.sh")
