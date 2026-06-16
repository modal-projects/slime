"""mini-swe-agent runtime (the default AgentRuntime).

Runs stock mini-swe-agent (v2) headless inside the sandbox in an isolated
uv-venv, with every model call dialing back to the slime adapter over litellm's
OpenAI-compatible API. Token capture + loss masking happen host-side; the
runner never sees token ids. The adapter MUST use the served model's sglang
tool-call parser (v2 drives bash via native tool-calls), e.g.
``--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3``.
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

MSWE_STEP_LIMIT = int(os.environ.get("MSWE_STEP_LIMIT", "50"))
# Which YAML config (prompts) the runner loads. Override ladder: MSWE_CONFIG
# env (global) > metadata.agent_config (per-row) > universal config below.
MSWE_CONFIG = os.environ.get("MSWE_CONFIG", "")
# Read at import: the scaffold must be identical for every rollout in a run.
UNIVERSAL_CONFIG_YAML = (Path(__file__).parent / "config" / "universal.yaml").read_text(encoding="utf-8")
# Exact-pinned: prompts + wire protocol are part of the RL task distribution,
# and MINI_RUNNER_PY below is written against the v2 API.
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
# knobs reach the request body -- the adapter applies it OVER its per-session
# defaults, so a client-sent temperature would silently turn rollouts greedy.
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

    # Default to the uploaded universal config; MSWE_CONFIG (if set) names a
    # BUILTIN packaged config. Read the builtin path directly -- the spec helper
    # would also try cwd-relative candidates a repo file could shadow.
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

    # Strip all sampling knobs (the config pins temperature=0.0 for
    # benchmarking) so the adapter's per-session defaults stay in force.
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
    agent_cfg.update(step_limit=STEP_LIMIT, cost_limit=0.0)

    # Prepend the testbed env's bin dirs onto PATH (config.env wins over
    # os.environ); conda activation never fires under /bin/sh.
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


# Provision mini-swe-agent into an isolated py3.11 uv venv. uv is our package
# manager; we NEVER fall back to the image's pip, because many task images
# (SWE-bench-Pro) ship a poisoned PIP_INDEX_URL pointing at a dead build-time
# mirror -> `pip install uv` hits "Connection refused .../uv/". Instead, mirror
# harbor's bootstrap: ensure curl via the OS package manager, install uv from
# the pinned astral script (falling back to pip ONLY with an explicit PyPI
# index), and retry network steps to absorb transient GitHub release resets.
# Measured across all 731 SWE-bench-Pro images at conc 50: 99.7% success (vs the
# original pip-fallback's ~52% on no-curl images). See
# profiles/PROVISIONING_VENV_VS_VOLUME.md.
_VENV_SETUP = (
    "set -e\n"
    'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\n'
    'retry() { for i in 1 2 3; do bash -c "$1" && return 0; sleep $((i*4)); done; return 1; }\n'
    "if ! command -v uv >/dev/null 2>&1; then\n"
    # Ensure curl via whatever package manager the image ships (best-effort:
    # if none works we still try the pip path below).
    '  command -v curl >/dev/null 2>&1 || retry "apt-get update && apt-get install -y curl'
    ' || apk add --no-cache curl || yum install -y curl || dnf install -y curl" || true\n'
    # uv via the pinned astral script (bypasses the image's pip entirely);
    # pip fallback forces a clean PyPI index so a poisoned image config can't win,
    # then retries with --break-system-packages so PEP 668 ("externally-managed")
    # images don't hard-fail. Try plain first: older pip (<23) lacks the flag but
    # also doesn't enforce PEP 668, so the plain attempt already succeeds there.
    '  retry "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh"'
    ' || retry "python3 -m pip install --index-url https://pypi.org/simple --root-user-action=ignore uv'
    ' || python3 -m pip install --index-url https://pypi.org/simple --root-user-action=ignore --break-system-packages uv"\n'
    '  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\n'
    "fi\n"
    f"rm -rf {shlex.quote(MSWE_AGENT_VENV)}\n"
    f'retry "uv venv --python {MSWE_AGENT_PYTHON_VERSION} {shlex.quote(MSWE_AGENT_VENV)}"\n'
    f'retry "uv pip install --python {shlex.quote(_VENV_PY)} {shlex.quote(MSWE_PIP_SPEC)}"\n'
    # pydantic_core's compiled native module intermittently fails to land in the
    # venv (partial wheel from a corrupt uv cache / index contention at high
    # conc) -> the agent dies at `import minisweagent.agents.default` with zero
    # turns (adapter_session_empty; ~46% of gravitational/teleport on one eval).
    # Verify the agent's REAL import and force a clean reinstall (--reinstall
    # --no-cache) to repair the partial wheel; a still-broken venv then fails
    # LOUDLY here (set -e -> CalledProcessError) instead of as a silent empty.
    f"for i in 1 2 3; do MSWEA_SILENT_STARTUP=1 {shlex.quote(_VENV_PY)} -c 'import minisweagent.agents.default' 2>/dev/null && break;"
    f' retry "uv pip install --python {shlex.quote(_VENV_PY)} --reinstall --no-cache {shlex.quote(MSWE_PIP_SPEC)}" || true; done\n'
    f"MSWEA_SILENT_STARTUP=1 {shlex.quote(_VENV_PY)} -c 'import minisweagent.agents.default'\n"
)

# MSWEA_SILENT_STARTUP suppresses the import-time banner that would otherwise
# corrupt the provisioning probe's marker comparison. Import the agent's real
# entrypoint (not just the top package) so the probe also rejects a pre-baked
# venv whose pydantic_core native module is missing -> re-provision instead of
# launching a doomed agent.
_VENV_CHECK = f"MSWEA_SILENT_STARTUP=1 {shlex.quote(_VENV_PY)} -c 'import minisweagent.agents.default'"


class MiniSweAgentRuntime(AgentRuntime):
    name = "mini-swe"
    adapter_cls = QwenOpenAIAdapter
    model_name = "slime-actor"
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
