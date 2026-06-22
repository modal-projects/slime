"""SweRebenchEnv: SWE-rebench-style coding tasks (agentic_rl's original path).

Single-shot episode: boot the task image, run one mini-swe leg, capture the git
diff, then grade in a FRESH sandbox (never the agent's dirtied one — prevents
reward hacking via leftover state) by applying the agent patch + held-out test
patch and running the tests. Reward 1.0 iff all FAIL_TO_PASS and PASS_TO_PASS
pass. pytest only (filter the dataset to ``parse_log_pytest``); add a parser for
other frameworks. Rows select this env via ``metadata.task_type == "swerebench"``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..sandbox import Sandbox
from .base import EnvMetadataError, EpisodeLimits, RewardResult, RolloutEnv, coerce_prompt

logger = logging.getLogger("agentic_rl")

# upstream eval apply flags
_APPLY = "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn"


class SweRebenchEnv(RolloutEnv):
    name = "swerebench"

    def normalize_metadata(self, sample) -> dict[str, Any]:
        m = sample.metadata or {}
        image = m.get("image_name") or m.get("docker_image")
        repo = m.get("repo")
        if not image:
            raise EnvMetadataError("missing_image_name")
        if not repo:
            raise EnvMetadataError("missing_repo")
        if not m.get("install_config", {}).get("test_cmd"):
            raise EnvMetadataError("missing_test_cmd")
        return {
            "instance_id": m.get("instance_id") or sample.label or repo,
            "image_name": image,
            "repo": repo,
            "workdir": "/" + repo.split("/")[1],
            "problem_statement": m.get("problem_statement") or coerce_prompt(sample.prompt),
            "install_config": m.get("install_config"),
            "test_patch": m.get("test_patch") or "",
            "FAIL_TO_PASS": list(m.get("FAIL_TO_PASS") or []),
            "PASS_TO_PASS": list(m.get("PASS_TO_PASS") or []),
            "cpus": m.get("cpus"),
            "memory_mb": m.get("memory_mb"),
        }

    def rollout(self, md: dict[str, Any], *, model, limits: EpisodeLimits) -> RewardResult:
        workdir = md["workdir"]
        patch, exit_status, boot_time, grade_time = "", "none", 0.0, 0.0
        sb = None
        t0 = time.perf_counter()
        try:
            sb = Sandbox(
                md["image_name"],
                cwd=workdir,
                # lifetime covers episode only; grading runs in a fresh sandbox.
                lifetime=limits.episode_timeout + 300,
                exec_timeout=limits.exec_timeout,
                cpu=md.get("cpus"),
                memory_mb=md.get("memory_mb"),
            )
            boot_time = sb.boot_time
            info = self.run_agent_leg(
                model, sb, md["problem_statement"], max_steps=limits.max_steps, wall_time_sec=limits.episode_timeout
            )
            exit_status = info.get("exit_status", "?")
            _, patch, _ = sb.exec("git add -A && git diff --cached HEAD", cwd=workdir, timeout=120)
        except Exception:  # noqa: BLE001 - episode still trains on turns it produced (reward 0)
            logger.exception("[swerebench] episode failed (instance=%s)", md["instance_id"])
        finally:
            if sb is not None:
                sb.terminate()

        reward, is_solved = 0.0, False
        if patch and not model.aborted:
            g0 = time.perf_counter()
            try:
                reward = self._grade(md, patch, timeout=limits.grade_timeout)
                is_solved = reward >= 1.0
            except Exception:  # noqa: BLE001
                logger.exception("[swerebench] grading failed (instance=%s)", md["instance_id"])
            grade_time = time.perf_counter() - g0

        return RewardResult(
            reward=reward,
            is_solved=is_solved,
            extra={
                "exit_status": exit_status,
                "timing": {"boot": round(boot_time, 1), "grade": round(grade_time, 1), "episode": round(time.perf_counter() - t0, 1)},
            },
        )

    def _grade(self, md: dict[str, Any], model_patch: str, *, timeout: int) -> float:
        cfg = md["install_config"]
        workdir = md["workdir"]
        test_cmds = cfg["test_cmd"] if isinstance(cfg["test_cmd"], list) else [cfg["test_cmd"]]
        sb = Sandbox(md["image_name"], cwd=workdir, lifetime=timeout, cpu=md.get("cpus"), memory_mb=md.get("memory_mb"))
        try:
            sb.write_file("/tmp/model.patch", model_patch)
            sb.write_file("/tmp/test.patch", md["test_patch"])
            script = "\n".join(
                ["set -e", "git reset --hard HEAD", f"{_APPLY} /tmp/model.patch", f"{_APPLY} /tmp/test.patch", *test_cmds]
            )
            _, out, err = sb.exec(script, cwd=workdir, timeout=timeout)
        finally:
            sb.terminate()

        passed = _passed_tests(out + "\n" + err)
        if not md["FAIL_TO_PASS"]:
            return 0.0  # no failing test to fix -> not a valid task
        resolved = all(t in passed for t in md["FAIL_TO_PASS"]) and all(t in passed for t in md["PASS_TO_PASS"])
        return 1.0 if resolved else 0.0


def _passed_tests(log: str) -> set[str]:
    """Test ids that PASSED in pytest output (mirrors SWE-rebench parse_log_pytest)."""
    passed = set()
    for line in log.splitlines():
        if line.startswith("PASSED"):
            parts = line.split()
            if len(parts) >= 2:
                passed.add(parts[1])
    return passed
