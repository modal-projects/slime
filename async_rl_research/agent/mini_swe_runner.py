"""Headless mini-swe-agent (v2) runner -- runs INSIDE the sandbox (design A).

Written into the agent sandbox by ``mini_swe_agent.MiniSweAgentRuntime.run_agent``
and executed with the isolated venv interpreter (``python -P``). NOT importable
on the host: it reads ``MSWE_*`` env vars and the problem file at startup and
talks to the slime adapter over litellm's OpenAI-compatible API. NO sampling
knobs reach the request body -- the adapter applies them OVER its per-session
defaults, so a client-sent temperature would silently turn rollouts greedy.
"""
import os
import sys
import traceback
from pathlib import Path

WORKDIR = os.environ["MSWE_WORKDIR"]
MODEL = os.environ.get("MSWE_MODEL", "slime-actor")
STEP_LIMIT = int(os.environ.get("MSWE_STEP_LIMIT", "50"))
MAX_EMPTY_TURNS = int(os.environ.get("MSWE_MAX_EMPTY_TURNS", "3"))
PATH_PREPEND = os.environ.get("MSWE_PATH_PREPEND", "")
with open(os.environ["MSWE_PROBLEM_FILE"], encoding="utf-8") as fh:
    TASK = fh.read()

try:
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.exceptions import FormatError, LimitsExceeded
    from minisweagent.models.litellm_model import LitellmModel

    class _StopAwareModel(LitellmModel):
        """End the episode instead of looping when the model can't progress.

        A no-tool-call response surfaces as FormatError from super().query()
        (LitellmModel stashes the raw response, incl. finish_reason, on it). We
        stop on finish_reason='length' (adapter signalled context/output budget
        exhausted) or after MAX_EMPTY_TURNS consecutive no-tool-call turns,
        raising LimitsExceeded -- mini-swe's own graceful 'exit' path, the same
        one step_limit uses. Without this mini-swe retries the format error every
        turn and burns the whole context to step_limit (49 dead turns seen on
        eval); finish_reason is otherwise never inspected.
        """

        _empty = 0

        def query(self, messages, **kwargs):
            try:
                msg = super().query(messages, **kwargs)
            except FormatError as e:
                resp = (e.messages[0].get("extra") or {}).get("response") or {}
                fr = ((resp.get("choices") or [{}])[0] or {}).get("finish_reason")
                self._empty += 1
                if fr == "length" or self._empty >= MAX_EMPTY_TURNS:
                    status = "ContextLengthExceeded" if fr == "length" else "NoProgress"
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": f"ending session: finish_reason={fr}, no-tool-call streak={self._empty}",
                            "extra": {"exit_status": status, "submission": ""},
                        }
                    )
                raise
            self._empty = 0
            return msg

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

    model = _StopAwareModel(**model_cfg)
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
