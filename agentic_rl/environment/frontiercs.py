"""FrontierCsEnv: Frontier-CS competitive-programming tasks as harbor rollouts.

Harbor tasks (per-task Dockerfile + ``tests/evaluate.py``) whose grading is a C++
submission scored by a go-judge. On top of stock ``HarborEnv``:

  * a verifier server (Node + go-judge) booted once per worker, holding the
    problem testdata, that BOTH the verifier's ``evaluate.py`` and the agent's
    iterative ``submit.sh`` POST to (see ``verifier_server/``);
  * the agent gets ``JUDGE_URL``/``PROBLEM_ID`` in its sandbox env so ``submit.sh``
    can self-grade mid-episode (the kept iterative-submit loop);
  * the workspace (/app) is kept minimal: only ``statement.txt`` + ``submit.sh``/
    ``submit.py`` are staged. The full statement and ``AGENT.md`` guidance are folded
    into the prompt (see ``_compose_prompt``) so the agent reads them in-context
    instead of cat-ing files; ``config.yaml`` (judge-internal) is dropped entirely.

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
from .submissions import parse_submissions_log
from .verifier_server.autostart import ensure_started

logger = logging.getLogger("agentic_rl")

# Only these task files are staged into /app. The statement + AGENT.md guidance are
# folded into the prompt (see _compose_prompt); config.yaml is judge-internal.
_STAGE_ENV_FILES = frozenset({"statement.txt", "submit.sh", "submit.py"})
# Where the task's submit.py appends one record per iterative judge submission.
SUBMISSIONS_LOG = "/logs/agent/submissions.jsonl"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _strip_md_fence(text: str) -> str:
    """Drop a wrapping ```...``` fence so the statement doesn't nest a code block."""
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


class FrontierCsEnv(HarborEnv):
    name = "frontier_cs"
    # The statement is folded into the prompt, so skip harbor's PROBLEM_STATEMENT.md.
    writes_problem_statement_file = False

    def normalize_metadata(self, sample) -> dict[str, Any]:
        md = super().normalize_metadata(sample)
        if not (md.get("verifier") or {}).get("env", {}).get("PROBLEM_ID"):
            raise EnvMetadataError("missing_problem_id")
        md["problem_statement"] = self._compose_prompt(md["problem_statement"], Path(md["task_dir"]) / "environment")
        return md

    @staticmethod
    def _compose_prompt(instruction: str, env_dir: Path) -> str:
        """Fold statement.txt + AGENT.md into the prompt so /app needs neither on disk
        for the agent to read them: the agent gets the full problem + workspace guidance
        in-context instead of spending turn 0 cat-ing files. statement.txt is still
        staged for reference, but the instruction's "go read these files" pointers are
        rewritten to point at the folded sections so the agent doesn't cat them."""
        statement = _strip_md_fence(_read_text(env_dir / "statement.txt"))
        guidance = _read_text(env_dir / "AGENT.md")
        # Redirect the instruction's stale "read it from disk" pointers to the folded
        # sections below (only when we actually fold that content).
        if statement:
            instruction = instruction.replace(
                "- Read statement.txt for the full problem description",
                "- The full problem description is in the <problem_statement> section below",
            ).replace(
                "Begin by reading the full problem statement in statement.txt.",
                "The full problem statement is in the <problem_statement> section below.",
            )
        if guidance:
            instruction = instruction.replace(
                "Read AGENT.md in this directory for compilation, testing, and workflow guidance.",
                "Compilation, testing, and workflow guidance is in the <workspace_guidance> section below.",
            )
        parts = [instruction]
        if statement:
            parts.append(f"<problem_statement>\n{statement}\n</problem_statement>")
        if guidance:
            parts.append(f"<workspace_guidance>\n{guidance}\n</workspace_guidance>")
        return "\n\n".join(parts)

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
        for name in sorted(_STAGE_ENV_FILES):
            f = env_dir / name
            if f.is_file():
                sb.write_file(f"{workdir}/{name}", f)
        sb.exec(f"chmod +x {shlex.quote(workdir)}/submit.sh 2>/dev/null || true", check=False, timeout=30)

    def _collect_artifacts(self, sb, workdir: str) -> dict[str, Any]:
        """Pull back the agent's iterative-submission trace. The staged ``submit.py``
        logs every mid-episode judge attempt to ``SUBMISSIONS_LOG``; the verifier
        grades only the final solution, so these intermediate scores are otherwise
        lost at sandbox teardown. Collapse them into an ordered per-attempt trace +
        summary on ``sample.metadata`` (each ``ordinal`` aligns with the agent's
        Nth submit tool-call, for later per-turn reward shaping)."""
        raw = sb.read_file(SUBMISSIONS_LOG)
        if not raw.strip():
            return {}
        parsed = parse_submissions_log(raw)
        if not parsed["submissions"]:
            return {}
        return {"submissions": parsed["submissions"], "submission_summary": parsed["summary"]}
