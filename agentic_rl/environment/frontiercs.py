"""FrontierCsEnv: Frontier-CS competitive-programming tasks as harbor rollouts.

Harbor tasks (per-task Dockerfile + ``tests/evaluate.py``) whose grading is a C++
submission scored by a go-judge. On top of stock ``HarborEnv``:

  * a verifier server (Node + go-judge) booted once per worker, holding the
    problem testdata, that BOTH the verifier's ``evaluate.py`` and the agent's
    iterative ``submit.sh`` POST to (see ``verifier_server/``);
  * the agent gets ``JUDGE_URL``/``PROBLEM_ID`` in its sandbox env so ``submit.sh``
    can self-grade mid-episode (the kept iterative-submit loop);
  * each task's ``environment/`` runtime files (statement.txt, submit.sh, ...)
    are staged into the workdir before the agent.

Rows select this env via ``metadata.task_type == "frontier_cs"``.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Any

from .base import EnvMetadataError, EpisodeLimits, RewardResult
from .harbor import HarborEnv
from .verifier_server.autostart import ensure_started

logger = logging.getLogger("agentic_rl")

_SKIP_ENV_FILES = frozenset({"Dockerfile", "docker-compose.yaml", "docker-compose.yml", ".dockerignore"})


class FrontierCsEnv(HarborEnv):
    name = "frontier_cs"

    def normalize_metadata(self, sample) -> dict[str, Any]:
        md = super().normalize_metadata(sample)
        if not (md.get("verifier") or {}).get("env", {}).get("PROBLEM_ID"):
            raise EnvMetadataError("missing_problem_id")
        return md

    def rollout(self, md: dict[str, Any], *, model, limits: EpisodeLimits) -> RewardResult:
        # Boot the verifier server once per worker; exports FRONTIER_CS_JUDGE_URL,
        # which harbor's verifier.env ${FRONTIER_CS_JUDGE_URL} templating resolves
        # for evaluate.py. Inject the resolved URL + PROBLEM_ID into the agent's
        # sandbox env so the iterative submit.sh loop reaches the same judge.
        url = ensure_started()
        pid = str((md.get("verifier") or {}).get("env", {}).get("PROBLEM_ID") or "")
        md = {
            **md,
            "agent_extra_env": {
                "JUDGE_URL": url,
                "PROBLEM_ID": pid,
                "SUBMIT_MAX_POLL_TIME": os.environ.get("FRONTIER_CS_SUBMIT_MAX_POLL_TIME", "600"),
            },
        }
        return super().rollout(md, model=model, limits=limits)

    def _pre_agent_setup(self, sb, task_dir: Path, md: dict[str, Any]) -> None:
        env_dir = task_dir / "environment"
        if not env_dir.is_dir():
            return
        workdir = md["workdir"]
        for f in sorted(env_dir.iterdir()):
            if f.is_file() and f.name not in _SKIP_ENV_FILES:
                sb.write_file(f"{workdir}/{f.name}", f)
        sb.exec(f"chmod +x {shlex.quote(workdir)}/submit.sh 2>/dev/null || true", check=False, timeout=30)
