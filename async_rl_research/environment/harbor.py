"""Harbor env: run harbor-format tasks (USACO, ...) as RL episodes on Modal.

Schema pair of ``env/convert2slime/harbor.py`` (see ``base.py``); the converter
bakes everything into ``metadata`` so this never reads ``task.toml`` at rollout.

Episode (harbor "shared" verifier semantics): boot sandbox, then per step write
the instruction, run the agent leg against the shared session, verify IN-PLACE
(upload tests/, run test.sh, parse /logs/verifier/reward.{json,txt}), and gate
on min_reward. Per-step rewards aggregate (mean | final) to a scalar. Tests are
uploaded only AFTER the agent leg so the agent can't read them.

Rollout needs ``ASYNC_RL_TASK_ROOT`` (dir relative ``task_path``s resolve
against). Oracle check (no model):

    python -m async_rl_research.environment.harbor out/usaco.jsonl --task-root out --limit 3
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from ..modal_sandbox import DockerfileImage, ModalSandbox
from .base import EnvMetadataError, RewardResult, RolloutEnv, coerce_prompt

logger = logging.getLogger(__name__)

TASK_ROOT_ENV = "ASYNC_RL_TASK_ROOT"

# Per-task task.toml timeouts override the env defaults only when this is set;
# default (off) keeps the env vars (boot/agent/eval) the single source of truth.
TASK_TIMEOUT_OVERRIDE = os.environ.get("ASYNC_RL_TASK_TIMEOUT_OVERRIDE", "0").strip().lower() in ("1", "true", "yes")

# ${VAR} / ${VAR:-default} templates in verifier env values, resolved against
# the HEAD's os.environ (unresolvable -> skipped with a warning).
_ENV_TEMPLATE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")


def _effective(env_val: int, task_val: Any) -> int:
    """Task.toml timeout wins over the env default only under TASK_TIMEOUT_OVERRIDE."""
    return int(task_val) if (TASK_TIMEOUT_OVERRIDE and task_val) else int(env_val)


def _resolve_env_templates(env: dict[str, str] | None) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in (env or {}).items():
        m = _ENV_TEMPLATE.fullmatch(str(value))
        if not m:
            resolved[key] = str(value)
            continue
        name, default = m.group(1), m.group(2)
        actual = os.environ.get(name, default)
        if actual is None:
            logger.warning("[harbor] env %s=${%s} unresolvable on this host; skipping", key, name)
            continue
        resolved[key] = actual
    return resolved


def _scalar_reward(rewards: dict[str, Any] | None) -> float:
    """Harbor 1D convention: the 'reward' key; a single-entry dict counts too."""
    if not rewards:
        return 0.0
    if "reward" in rewards:
        return float(rewards["reward"])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    logger.warning("[harbor] multi-key rewards %s without 'reward' key; scalar=0", sorted(rewards))
    return 0.0


def _meets_min_reward(rewards: dict[str, Any] | None, min_reward: float | dict[str, float] | None) -> bool:
    """Harbor's step gate: missing rewards/keys are treated as -inf."""
    if min_reward is None:
        return True
    if isinstance(min_reward, dict):
        return all(rewards is not None and key in rewards and float(rewards[key]) >= float(v) for key, v in min_reward.items())
    return rewards is not None and "reward" in rewards and float(rewards["reward"]) >= float(min_reward)


class HarborEnv(RolloutEnv):
    name = "harbor"
    # No agent_config default: harbor instruction.md files carry their own
    # deliverable contract. Override per-row via metadata.agent_config.

    # Row schema (written by env/convert2slime/harbor.py -- keep in sync)
    def normalize_metadata(self, sample) -> dict[str, Any]:
        m = sample.metadata or {}
        task_path = m.get("task_path")
        if not task_path:
            raise EnvMetadataError("missing_task_path")
        task_dir = self._resolve_task_dir(task_path)

        docker_image = m.get("docker_image")
        dockerfile = m.get("dockerfile")
        if not docker_image and not dockerfile:
            raise EnvMetadataError("missing_docker_image_or_dockerfile")
        if dockerfile and not (task_dir / dockerfile).is_file():
            raise EnvMetadataError(f"dockerfile_missing:{task_dir / dockerfile}")

        steps = m.get("steps") or None
        if steps:
            for step in steps:
                if not (task_dir / step["tests_path"] / "test.sh").is_file():
                    raise EnvMetadataError(f"tests_missing:{step['tests_path']}")
        elif not (task_dir / "tests" / "test.sh").is_file():
            raise EnvMetadataError("tests_missing:tests")

        return {
            "instance_id": m.get("instance_id") or sample.label or task_dir.name,
            "task_dir": str(task_dir),
            "docker_image": docker_image,
            "dockerfile": dockerfile,
            "workdir": m.get("workdir"),  # None -> detected from the sandbox
            "problem_statement": m.get("problem_statement") or coerce_prompt(sample.prompt),
            "agent_timeout_sec": m.get("agent_timeout_sec"),
            "build_timeout_sec": m.get("build_timeout_sec"),
            "verifier": m.get("verifier") or {},
            "steps": steps,
            "reward_strategy": m.get("reward_strategy"),
            "cpus": m.get("cpus"),
            "memory_mb": m.get("memory_mb"),
            "agent_config": m.get("agent_config"),
        }

    def effective_budgets(self, md: dict[str, Any], *, agent_time_budget_sec: int, eval_timeout_sec: int) -> dict[str, int]:
        return {
            "boot_sec": _effective(ModalSandbox._boot_timeout_from_env(), md.get("build_timeout_sec")),
            "agent_sec": _effective(agent_time_budget_sec, md.get("agent_timeout_sec")),
            "eval_sec": _effective(eval_timeout_sec, (md.get("verifier") or {}).get("timeout_sec")),
        }

    @staticmethod
    def _resolve_task_dir(task_path: str) -> Path:
        p = Path(task_path)
        if not p.is_absolute():
            root = os.environ.get(TASK_ROOT_ENV)
            if not root:
                raise EnvMetadataError(f"{TASK_ROOT_ENV}_unset")
            p = Path(root) / p
        if not p.is_dir():
            raise EnvMetadataError(f"task_dir_missing:{p}")
        return p

    # Episode
    def _image(self, md: dict[str, Any]) -> str | DockerfileImage:
        if md["docker_image"]:
            return md["docker_image"]
        path = Path(md["task_dir"]) / md["dockerfile"]
        return DockerfileImage(path=str(path), context_dir=str(path.parent))

    def _sandbox(self, md: dict[str, Any]) -> ModalSandbox:
        kwargs: dict[str, Any] = {}
        if md["cpus"]:
            kwargs["cpu"] = float(md["cpus"])
        if md["memory_mb"]:
            kwargs["memory_mb"] = int(md["memory_mb"])
        if md["workdir"]:
            kwargs["workdir"] = md["workdir"]
        if TASK_TIMEOUT_OVERRIDE and md.get("build_timeout_sec"):
            kwargs["boot_timeout"] = int(md["build_timeout_sec"])
        return ModalSandbox(self._image(md), **kwargs)

    def _step_specs(self, md: dict[str, Any]) -> list[dict[str, Any]]:
        """Uniform step list; a single-step task becomes one pseudo-step."""
        if md["steps"]:
            return md["steps"]
        return [
            {
                "name": None,
                "instruction": md["problem_statement"],
                "tests_path": "tests",
                "verifier": md["verifier"],
                "min_reward": None,
                "agent_timeout_sec": None,
            }
        ]

    async def rollout(
        self,
        md: dict[str, Any],
        *,
        runtime,
        session_id: str,
        adapter_url: str,
        agent_time_budget_sec: int,
        eval_timeout_sec: int,
    ) -> RewardResult:
        async def agent_leg(sb, leg_md: dict[str, Any], budget_sec: int):
            return await runtime.run_agent(
                sb,
                md=leg_md,
                session_id=session_id,
                adapter_url=adapter_url,
                time_budget_sec=budget_sec,
            )

        return await self._episode(
            md,
            run_leg=agent_leg,
            agent_time_budget_sec=agent_time_budget_sec,
            eval_timeout_sec=eval_timeout_sec,
        )

    async def _episode(
        self,
        md: dict[str, Any],
        *,
        run_leg,
        agent_time_budget_sec: int,
        eval_timeout_sec: int,
    ) -> RewardResult:
        """Shared by the RL rollout and the oracle check: only the leg differs."""
        task_dir = Path(md["task_dir"])
        steps = self._step_specs(md)
        step_results: list[dict[str, Any]] = []
        # Last agent leg's exit code + failure tail (stays None for the oracle
        # leg, which runs solve.sh rather than the agent). Surfaced in extra so
        # a zero-turn adapter_session_empty self-explains in the rollout dump.
        last_agent = None

        async with self._sandbox(md) as sb:
            workdir = md["workdir"] or await self._detect_workdir(sb)
            q = shlex.quote
            # Test scripts assume /logs/{agent,verifier,artifacts} exist.
            await sb.exec(
                f"mkdir -p {q(workdir)} /logs/agent /logs/verifier /logs/artifacts",
                check=True,
                timeout=60,
            )

            # Start the agent clock only once the sandbox is booted and prepped:
            # a cold per-instance image pull can take many minutes, and charging
            # it against the agent budget would exhaust the window before any step
            # runs (-> zero agent turns -> adapter_session_empty). Provisioning on
            # the first leg likewise gets its own clock inside _detached_run.
            deadline = time.monotonic() + _effective(agent_time_budget_sec, md.get("agent_timeout_sec"))

            for step in steps:
                remaining = int(deadline - time.monotonic())
                if remaining <= 0:
                    logger.warning("[harbor] %s: agent budget exhausted before step %r", md["instance_id"], step["name"])
                    break
                budget = remaining
                if TASK_TIMEOUT_OVERRIDE and step.get("agent_timeout_sec"):
                    budget = min(budget, int(step["agent_timeout_sec"]))

                await self.write_problem_file(sb, workdir, step["instruction"])
                leg_md = {**md, "workdir": workdir}
                leg_result = await run_leg(sb, leg_md, budget)
                if leg_result is not None:
                    last_agent = leg_result

                rewards = await self._verify(
                    sb,
                    tests_dir=task_dir / step["tests_path"],
                    workdir=workdir,
                    verifier={**md["verifier"], **(step.get("verifier") or {})},
                    eval_timeout_sec=eval_timeout_sec,
                    instance_id=md["instance_id"],
                )
                step_results.append({"name": step["name"], "rewards": rewards, "reward": _scalar_reward(rewards)})
                if not _meets_min_reward(rewards, step.get("min_reward")):
                    logger.info("[harbor] %s: step %r below min_reward; aborting remaining steps", md["instance_id"], step["name"])
                    break

        reward = self._aggregate(steps, step_results, md["reward_strategy"])
        extra: dict[str, Any] = {
            "harbor_step_results": step_results,
            "harbor_steps_completed": len(step_results),
            "harbor_steps_total": len(steps),
        }
        if last_agent is not None:
            extra["agent_exit_code"] = last_agent.exit_code
            extra["agent_tail"] = last_agent.tail
        return RewardResult(
            reward=reward,
            # epsilon: weighted pytest fractions sum to 0.999... for a fully-
            # passing task (seen on openthoughts-tblite bash-log-processor-fix)
            is_solved=reward >= 1.0 - 1e-6,
            extra=extra,
        )

    @staticmethod
    def _aggregate(steps: list[dict], results: list[dict], strategy: str | None) -> float:
        """Scalar episode reward from per-step scalars.

        'mean' divides by ALL declared steps (gated steps count 0; stricter than
        harbor's job-level mean, but the conservative signal is what RL wants).
        'final' is the last declared step's reward (0 if never reached).
        """
        if not results:
            return 0.0
        if len(steps) == 1:
            return results[0]["reward"]
        if (strategy or "mean") == "final":
            return results[-1]["reward"] if len(results) == len(steps) else 0.0
        return sum(r["reward"] for r in results) / len(steps)

    @staticmethod
    async def _detect_workdir(sb) -> str:
        # Prebuilt docker_image rows may not know their WORKDIR; ask the sandbox.
        ec, out, _ = await sb.exec("pwd", check=False, timeout=30)
        detected = (out or "").strip().splitlines()[-1] if ec == 0 and (out or "").strip() else ""
        return detected or "/app"

    # In-place verification (harbor's shared-environment Verifier semantics)
    async def _verify(
        self,
        sb,
        *,
        tests_dir: Path,
        workdir: str,
        verifier: dict[str, Any],
        eval_timeout_sec: int,
        instance_id: str,
    ) -> dict[str, Any] | None:
        """Upload tests, run test.sh, parse the reward files. None = no verdict."""
        timeout = _effective(eval_timeout_sec, verifier.get("timeout_sec"))

        await self.upload_dir(sb, tests_dir, "/tests")
        q = shlex.quote
        await sb.exec(
            "chmod +x /tests/test.sh && rm -f /logs/verifier/reward.json /logs/verifier/reward.txt",
            check=False,
            timeout=60,
        )
        env = _resolve_env_templates(verifier.get("env"))
        ec, out, err = await sb.exec(
            f"cd {q(workdir)} && bash /tests/test.sh",
            env=env or None,
            timeout=timeout,
            check=False,
        )
        if os.environ.get("HARBOR_VERIFY_DEBUG"):
            logger.info(
                "[harbor-verify-debug] %s: test.sh exit=%s\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s",
                instance_id,
                ec,
                (out or "")[-3000:],
                (err or "")[-3000:],
            )

        raw_json = await sb.read_file("/logs/verifier/reward.json")
        if raw_json.strip():
            try:
                return dict(json.loads(raw_json))
            except (ValueError, TypeError):
                logger.warning("[harbor] %s: unparseable reward.json: %.200s", instance_id, raw_json)
                return None
        raw_txt = await sb.read_file("/logs/verifier/reward.txt")
        if raw_txt.strip():
            try:
                return {"reward": float(raw_txt.strip())}
            except ValueError:
                logger.warning("[harbor] %s: unparseable reward.txt: %.200s", instance_id, raw_txt)
                return None
        # No reward file: terminal-bench-style tasks end test.sh with a bare
        # pytest run. Grade all-or-nothing on its exit code (0 pass, 1 fail);
        # anything else stays "no verdict" so infra breakage isn't scored.
        if ec in (0, 1):
            logger.info("[harbor] %s: no reward file; graded from test.sh exit=%d", instance_id, ec)
            return {"reward": 1.0 if ec == 0 else 0.0, "graded_from": "exit_code"}
        logger.warning(
            "[harbor] %s: test.sh exit=%s wrote no reward file; stderr tail: %s",
            instance_id,
            ec,
            (err or "")[-400:],
        )
        return None

    # Oracle check (reference solution through the exact rollout path)
    async def oracle_episode(self, md: dict[str, Any], *, solve_timeout_sec: int, eval_timeout_sec: int) -> RewardResult:
        """Replace the agent leg with the task's solution/solve.sh (a counter
        maps each sequential leg to its step)."""
        task_dir = Path(md["task_dir"])
        steps = self._step_specs(md)
        state = {"i": 0}

        async def leg(sb, leg_md: dict[str, Any], budget_sec: int) -> None:
            step = steps[state["i"]]
            state["i"] += 1
            solution_dir = task_dir / "steps" / step["name"] / "solution" if step["name"] else task_dir / "solution"
            if not (solution_dir / "solve.sh").is_file():
                solution_dir = task_dir / "solution"
            if not (solution_dir / "solve.sh").is_file():
                raise FileNotFoundError(f"no solution/solve.sh under {task_dir}")
            await self.upload_dir(sb, solution_dir, "/solution")
            q = shlex.quote
            await sb.exec("chmod +x /solution/solve.sh", check=False, timeout=30)
            ec, out, err = await sb.exec(
                f"cd {q(leg_md['workdir'])} && bash /solution/solve.sh",
                timeout=min(budget_sec, solve_timeout_sec),
                check=False,
            )
            if ec != 0:
                logger.warning("[harbor-oracle] solve.sh exit=%d stderr tail: %s", ec, (err or "")[-400:])
            if os.environ.get("HARBOR_VERIFY_DEBUG"):
                logger.info(
                    "[harbor-oracle-debug] %s: solve.sh exit=%s\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s",
                    md["instance_id"],
                    ec,
                    (out or "")[-3000:],
                    (err or "")[-3000:],
                )

        return await self._episode(
            md,
            run_leg=leg,
            agent_time_budget_sec=solve_timeout_sec * max(1, len(steps)),
            eval_timeout_sec=eval_timeout_sec,
        )


def _oracle_main() -> int:
    import argparse
    import asyncio
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Run harbor reference solutions through the rollout path (reward should be 1.0).")
    parser.add_argument("jsonl", help="converted slime prompt JSONL (env/convert2slime/harbor.py output)")
    parser.add_argument("--task-root", help=f"task root (default: ${TASK_ROOT_ENV} or the JSONL's directory)")
    parser.add_argument("--limit", type=int, default=1, help="how many rows to check (default 1)")
    parser.add_argument("--index", type=int, help="check exactly this row")
    parser.add_argument("--solve-timeout", type=int, default=600)
    parser.add_argument("--eval-timeout", type=int, default=int(os.environ.get("AGENT_EVAL_TIMEOUT_SEC", "600")))
    parser.add_argument(
        "--vm-runtime",
        action="store_true",
        help="boot VM sandboxes (experimental_options={'vm_runtime': True}) instead of gVisor",
    )
    args = parser.parse_args()
    if args.vm_runtime:
        os.environ["SLIME_AGENT_SANDBOX_VM_RUNTIME"] = "1"

    root = args.task_root or os.environ.get(TASK_ROOT_ENV) or str(Path(args.jsonl).resolve().parent)
    os.environ[TASK_ROOT_ENV] = root

    rows = []
    with open(args.jsonl, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    picked = [rows[args.index]] if args.index is not None else rows[: args.limit]

    env = HarborEnv()
    failures = 0
    for row in picked:
        sample = SimpleNamespace(metadata=row.get("metadata"), prompt=row.get("prompt"), label=row.get("label"))
        md = env.normalize_metadata(sample)
        t0 = time.monotonic()
        result = asyncio.run(env.oracle_episode(md, solve_timeout_sec=args.solve_timeout, eval_timeout_sec=args.eval_timeout))
        status = "OK " if result.is_solved else "FAIL"
        print(f"[{status}] {md['instance_id']}: reward={result.reward:.2f} t={time.monotonic() - t0:.0f}s {result.extra}")
        failures += 0 if result.is_solved else 1
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(_oracle_main())
