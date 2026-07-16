"""FrontierCsEnv: Frontier-CS competitive-programming tasks as harbor rollouts.

Harbor tasks (per-task Dockerfile + ``tests/evaluate.py``) whose grading is a C++
submission scored by a go-judge. On top of stock ``HarborEnv``:

  * a verifier server (Node + go-judge) booted once per worker, holding the
    problem testdata, that BOTH the final grading and the agent's iterative
    ``submit.sh`` POST to (see ``verifier_server/``);
  * the agent gets ``JUDGE_URL``/``PROBLEM_ID``/``AGENT_ID`` in its sandbox env so
    ``submit.sh`` can self-grade mid-episode (the kept iterative-submit loop) and
    the judge attributes every submission to this episode;
  * REWARD-RELEVANT SCORES ARE SERVER-SIDE: the episode's submission record is
    read back from the judge (``/agent/:id/submissions``) rather than the
    sandbox's agent-writable ``submissions.jsonl`` (kept only to enrich entries
    + as a forgery tripwire), and the final ``solution.cpp`` is graded by the
    WORKER submitting it to the judge directly (``_verify`` override) rather
    than trusting the sandbox-written ``reward.json``. Both fall back to the
    legacy in-sandbox path if the judge/endpoint is unavailable.
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
import uuid
from pathlib import Path
from typing import Any

from .base import EnvMetadataError, EpisodeLimits, RewardResult
from .harbor import HarborEnv
from .submissions import merge_server_submissions, parse_submissions_log
from .verifier_server import client as judge_client
from .verifier_server.autostart import ensure_started

logger = logging.getLogger("agentic_rl")

# Only these task files are staged into /app. The statement + AGENT.md guidance are
# folded into the prompt (see _compose_prompt); config.yaml is judge-internal.
_STAGE_ENV_FILES = frozenset({"statement.txt", "submit.sh", "submit.py"})
# Where the task's submit.py appends one record per iterative judge submission.
SUBMISSIONS_LOG = "/logs/agent/submissions.jsonl"
# Grade the final solution.cpp from the WORKER (trusted) instead of running the
# in-sandbox tests/evaluate.py and trusting its reward.json. Default on; set 0
# to revert to the legacy in-sandbox verifier.
SERVER_VERIFY_ENV = "FRONTIER_CS_SERVER_VERIFY"
_SOLVED_EPS = 1e-6


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
        # sandbox env so the iterative submit.sh loop reaches the same judge, plus
        # a fresh per-episode AGENT_ID: submit.py sends it with every POST, letting
        # the judge attribute submissions to this episode so _collect_artifacts can
        # read the episode's scores back SERVER-SIDE instead of trusting the
        # sandbox's submissions.jsonl.
        url = ensure_started()
        verifier = md.get("verifier") or {}
        pid = str(verifier.get("env", {}).get("PROBLEM_ID") or "")
        agent_id = uuid.uuid4().hex
        md = {
            **md,
            "judge_url": url,
            "judge_agent_id": agent_id,
            # AGENT_ID into the verifier env too, so the worker-side _verify can
            # attribute the final grade as "<episode>:final" on the judge.
            "verifier": {**verifier, "env": {**verifier.get("env", {}), "AGENT_ID": agent_id}},
            "agent_extra_env": {
                "JUDGE_URL": url,
                "PROBLEM_ID": pid,
                "AGENT_ID": agent_id,
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

    def _collect_artifacts(self, sb, workdir: str, md: dict[str, Any]) -> dict[str, Any]:
        """Pull the episode's iterative-submission record, SERVER-SIDE first.

        The judge attributes every mid-episode submission to this episode's
        ``AGENT_ID`` (see ``rollout``); read the scores back from the judge's own
        records (``/agent/:id/submissions``) so the reward-bearing values never
        transit the agent-writable sandbox. The sandbox's ``SUBMISSIONS_LOG`` is
        still parsed, but only to enrich matched entries (uuid/code size) and to
        feed the forgery-tripwire counters (``n_log_only``/``n_score_mismatch``
        in the summary). If the judge endpoint is unavailable (pre-deployed old
        judge), fall back to the legacy log-derived trace, tagged
        ``source: "agent_log"`` so the fallback rate is visible in W&B."""
        raw = sb.read_file(SUBMISSIONS_LOG)
        parsed = parse_submissions_log(raw) if raw.strip() else {"submissions": [], "summary": {}}

        url, agent_id = md.get("judge_url"), md.get("judge_agent_id")
        if url and agent_id:
            try:
                server_rows = judge_client.fetch_agent_submissions(url, agent_id)
            except judge_client.JudgeClientError as e:
                logger.warning("[frontier_cs] %s: server-side submission fetch failed (%s); "
                               "falling back to sandbox log", md.get("instance_id"), e)
            else:
                if not server_rows and not parsed["submissions"]:
                    return {}
                merged = merge_server_submissions(server_rows, parsed)
                return {"submissions": merged["submissions"], "submission_summary": merged["summary"]}

        if not parsed["submissions"]:
            return {}
        return {"submissions": parsed["submissions"], "submission_summary": {**parsed["summary"], "source": "agent_log"}}

    def _verify(self, sb, *, tests_dir: Path, workdir: str, verifier: dict[str, Any], eval_timeout_sec: int | None, instance_id: str) -> dict[str, Any] | None:
        """Grade the final solution.cpp from the WORKER: read the file off the
        sandbox and submit it to the judge from this (trusted) process, mirroring
        the in-sandbox ``tests/evaluate.py`` exactly (same normalization:
        ``clamp01(score/100)``, same is_solved rule) — but the grade never
        transits the sandbox, so a booby-trapped test environment or forged
        ``reward.json`` cannot influence the reward. Legacy in-sandbox path via
        ``FRONTIER_CS_SERVER_VERIFY=0`` or as fallback on unexpected errors."""
        if os.environ.get(SERVER_VERIFY_ENV, "1").strip().lower() in ("0", "false", "no"):
            return super()._verify(sb, tests_dir=tests_dir, workdir=workdir, verifier=verifier,
                                   eval_timeout_sec=eval_timeout_sec, instance_id=instance_id)
        url = os.environ.get("FRONTIER_CS_JUDGE_URL", "").strip()
        pid = str((verifier.get("env") or {}).get("PROBLEM_ID") or "")
        agent_id = (verifier.get("env") or {}).get("AGENT_ID") or ""
        if not url or not pid:
            logger.warning("[frontier_cs] %s: server verify missing judge url/pid; using in-sandbox verifier", instance_id)
            return super()._verify(sb, tests_dir=tests_dir, workdir=workdir, verifier=verifier,
                                   eval_timeout_sec=eval_timeout_sec, instance_id=instance_id)

        code = sb.read_file(f"{workdir}/solution.cpp")
        if not code.strip():
            return {"reward": 0.0, "score_raw": 0.0, "cases_passed": 0, "cases_total": 0,
                    "is_solved": False, "detail": "no solution.cpp", "graded_from": "server"}
        timeout = int(eval_timeout_sec or verifier.get("timeout_sec") or 600)
        try:
            result = judge_client.grade_solution(
                url, pid, code,
                agent_id=f"{agent_id}:final" if agent_id else "",
                poll_timeout_sec=timeout,
            )
        except judge_client.JudgeClientError as e:
            # Judge unreachable/timed out: the in-sandbox evaluate.py talks to the
            # same judge, so don't fall back (it would just burn the timeout again).
            logger.warning("[frontier_cs] %s: server-side grading failed: %s", instance_id, e)
            return {"reward": 0.0, "score_raw": 0.0, "cases_passed": 0, "cases_total": 0,
                    "is_solved": False, "detail": f"server grading failed: {e}", "graded_from": "server"}
        except Exception:  # noqa: BLE001 — unexpected bug in the new path: revert to legacy
            logger.exception("[frontier_cs] %s: server-side grading errored; using in-sandbox verifier", instance_id)
            return super()._verify(sb, tests_dir=tests_dir, workdir=workdir, verifier=verifier,
                                   eval_timeout_sec=eval_timeout_sec, instance_id=instance_id)

        if result.get("status") == "error":
            detail = str(result.get("message") or result.get("error") or "judge error")
            return {"reward": 0.0, "score_raw": 0.0, "cases_passed": 0, "cases_total": 0,
                    "is_solved": False, "detail": detail, "graded_from": "server"}
        score_raw = float(result.get("score") or 0.0)  # judge-native 0..100
        cases = result.get("cases") or []
        cases_total = len(cases) if isinstance(cases, list) else 0
        cases_passed = sum(1 for c in cases if isinstance(c, dict) and (c.get("scoreRatio") or 0) >= 1.0)
        is_solved = bool(result.get("passed")) or score_raw >= 100.0 - _SOLVED_EPS
        return {"reward": max(0.0, min(1.0, score_raw / 100.0)), "score_raw": score_raw,
                "cases_passed": cases_passed, "cases_total": cases_total,
                "is_solved": is_solved, "graded_from": "server"}
