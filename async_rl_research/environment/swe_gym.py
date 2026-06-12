"""SWE-Gym env: git-diff capture + clean-sandbox grading on Modal.

The schema pair of ``env/convert2slime/swe_gym.py`` (see ``base.py`` for the
convention). Each row carries a prebuilt per-instance registry image plus a
self-contained ``eval_cmd`` (or a ``swepro`` run/parse spec), and grading is
diff-transplant based:

    boot work sandbox -> pre_commands + problem file -> agent runs ->
    capture git diff -> CLOSE work sandbox -> boot CLEAN sandbox ->
    re-apply pre_commands -> apply diff -> run eval_cmd.

No-test-cheating guarantee: the evaluator sandbox never sees the agent's
filesystem -- only the captured diff can affect reward. This is what
``rollout`` owning the whole lifecycle buys: the work sandbox is torn down
BEFORE the evaluator boots, exactly as the pre-refactor flow did.

The provider backend is ``modal_sandbox.ModalSandbox``; boot concurrency and
create-retry live there, so the evaluator sandbox is gated and retried just
like the work sandbox.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Any

from slime.agent.sandbox import Sandbox

from ..modal_sandbox import ModalSandbox
from .base import PROBLEM_FILE, EnvMetadataError, RewardResult, RolloutEnv, coerce_prompt

logger = logging.getLogger(__name__)


# Patch/pre scripts live under /tmp, outside the diff's reach.
_PATCH = "/tmp/__swe_patch__.diff"
_PRE = "/tmp/__swe_pre__.sh"


# Appended to the row's problem statement: with the universal prompt scaffold
# the agent's YAML no longer carries a submission protocol, so the task
# instruction must say what the deliverable is. Reward here is the captured
# working-tree `git diff`, hence: edit in place, don't commit, no patch files
# (the builtin swebench prompt's patch.txt ritual was never what got graded).
_DELIVERABLE_SUFFIX = """

## Deliverable

Fix the issue by editing the repository's source files in place.

- Your work is collected as the uncommitted working-tree changes (`git diff`) of this repository when you finish: leave your edits uncommitted.
- Do NOT commit your changes and do NOT create patch files.
- Do NOT modify tests or configuration files (pyproject.toml, setup.cfg, etc.).
- Delete any reproduction scripts or scratch files you created before finishing.
"""


class SweGymEnv(RolloutEnv):
    name = "swe_gym"
    # No agent_config default: the runtime's universal prompt scaffold is the
    # default; the SWE deliverable contract is _DELIVERABLE_SUFFIX above.
    # Per-row builtin override via metadata.agent_config; global via MSWE_CONFIG.

    def normalize_metadata(self, sample) -> dict[str, Any]:
        m = sample.metadata or {}
        label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
        md = {
            "instance_id": m.get("instance_id") or label or "unknown",
            "image": m.get("image"),
            "workdir": m.get("workdir"),
            "problem_statement": m.get("problem_statement") or coerce_prompt(sample.prompt),
            "swepro": m.get("swepro"),
            "eval_cmd": m.get("eval_cmd"),
            "pre_commands": m.get("pre_commands"),
            "agent_config": m.get("agent_config"),
        }
        if not md["image"] or not md["workdir"]:
            raise EnvMetadataError("missing_image_or_workdir")
        return md

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
        workdir = md["workdir"]
        async with ModalSandbox(md["image"]) as sb:
            await self._prepare_workspace(sb, md)
            await runtime.run_agent(
                sb,
                md=md,
                session_id=session_id,
                adapter_url=adapter_url,
                time_budget_sec=agent_time_budget_sec,
            )
            diff_text = await self._git_diff(sb, workdir, exclude=runtime.diff_exclude_all)

        # Work sandbox is closed; grade the diff in a clean one.
        reward, is_solved, applied = await self._evaluate(md, diff_text, timeout_sec=eval_timeout_sec)
        return RewardResult(
            reward=float(reward),
            is_solved=bool(is_solved),
            extra={"applied_cleanly": bool(applied)},
        )

    # ------------------------------------------------------------------
    # Workspace prep (work sandbox; task-side, agent-agnostic)
    # ------------------------------------------------------------------
    async def _prepare_workspace(self, sb: Sandbox, md: dict[str, Any]) -> None:
        """Bring a freshly booted work sandbox to the task's start state.

        ``pre_commands`` must run in BOTH the work and eval sandboxes (see
        ``_apply_pre_commands``): they are typically ``git checkout
        <base_sha> -f``, and skipping either side makes the model's diff
        context mismatch the eval base -> apply failures.
        """
        # git operations inside the sandbox (pre_commands, the later diff
        # capture) need the repo marked safe for root.
        await sb.exec("git config --system --add safe.directory '*'", check=False, timeout=60)
        if md["pre_commands"]:
            await _apply_pre_commands(sb, md["workdir"], md["pre_commands"])
        await self.write_problem_file(sb, md["workdir"], (md["problem_statement"] or "") + _DELIVERABLE_SUFFIX)

    # ------------------------------------------------------------------
    # Diff capture
    # ------------------------------------------------------------------
    async def _git_diff(self, sb: Sandbox, workdir: str, *, exclude: tuple[str, ...] = ()) -> str:
        """Capture the model's edits as a patch, excluding scratch files.

        ``git add -N .`` stages intent-to-add for new files so they show up in
        the diff. ``exclude`` is the active runtime's ``diff_exclude_all``;
        the task-layer ``PROBLEM_FILE`` is always excluded here.
        """
        excludes = " ".join(f"':(exclude){f}'" for f in (PROBLEM_FILE, *exclude))
        cmd = f"cd {shlex.quote(workdir)} && git add -N . && git diff -- . {excludes}"
        _, out, _ = await sb.exec(cmd, user="root", timeout=120, check=False)
        return out

    # ------------------------------------------------------------------
    # Eval (fresh clean sandbox, apply diff, run dataset tests)
    # ------------------------------------------------------------------
    async def _evaluate(self, md: dict[str, Any], diff_text: str, *, timeout_sec: int) -> tuple[float, bool, bool]:
        """Grade ``diff_text`` in a CLEAN sandbox; returns (reward, solved, applied)."""
        if not (md["swepro"] or md["eval_cmd"]):
            logger.warning("[swe_gym.evaluate] no swepro/eval_cmd; reward=0")
            return 0.0, False, True

        workdir = md["workdir"]
        async with ModalSandbox(md["image"]) as ev:
            if md["pre_commands"]:
                await _apply_pre_commands(ev, workdir, md["pre_commands"])

            applied = await _apply_diff(ev, workdir, diff_text)
            if not applied:
                return 0.0, False, False

            if md["swepro"]:
                reward, solved = await _run_swepro(ev, workdir, md["swepro"], timeout_sec)
                return reward, solved, True
            reward, solved = await _run_eval_cmd(ev, workdir, md["eval_cmd"], timeout_sec)
            return reward, solved, True


async def _apply_pre_commands(sb: Sandbox, workdir: str, pre: list[str] | str) -> None:
    body = pre.replace("\\n", "\n") if isinstance(pre, str) else "\n".join(c for c in (pre or []) if c)
    await sb.write_file(_PRE, "set -e\n" + body)
    await sb.exec(f"cd {shlex.quote(workdir)} && bash {shlex.quote(_PRE)}", check=False, timeout=600)


async def _apply_diff(sb: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await sb.write_file(_PATCH, diff_text)
    wq = shlex.quote(workdir)
    pq = shlex.quote(_PATCH)
    for cmd in (
        f"cd {wq} && git apply --3way --whitespace=nowarn {pq}",
        f"cd {wq} && git apply --whitespace=nowarn {pq}",
        f"cd {wq} && patch -p1 --no-backup-if-mismatch < {pq}",
    ):
        ec, _, _ = await sb.exec(cmd, check=False, timeout=120)
        if ec == 0:
            return True
    return False


async def _run_eval_cmd(sb: Sandbox, workdir: str, cmd: str, timeout: int) -> tuple[float, bool]:
    # What SWE-Gym-Lite emits: a self-contained command whose exit code is the
    # verdict (applies the test_patch, runs pytest on F2P/P2P).
    ec, _, _ = await sb.exec(f"cd {shlex.quote(workdir)} && {cmd}", check=False, timeout=timeout)
    return (1.0 if ec == 0 else 0.0), ec == 0


async def _run_swepro(sb: Sandbox, workdir: str, swepro: dict, timeout: int) -> tuple[float, bool]:
    # Forward-compat pass-through for swepro-style grading (SWE-Gym-Lite data
    # carries none, but a richer dataset can supply a run/parse script pair).
    swepro_dir = "/tmp/swepro_eval"
    await sb.exec(f"mkdir -p {swepro_dir} && chmod 777 {swepro_dir}", check=True, timeout=30)
    for key, dst in (("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")):
        host_path = swepro.get(key)
        if host_path:
            await sb.write_file(f"{swepro_dir}/{dst}", Path(host_path).read_text())
    await sb.exec(f"chmod -R 755 {swepro_dir}", check=False, timeout=30)

    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{swepro_dir}/stdout.log"
    stderr_f = f"{swepro_dir}/stderr.log"
    result_f = f"{swepro_dir}/result.json"
    await sb.exec(
        f"cd {shlex.quote(workdir)} && bash {swepro_dir}/run_script.sh "
        f"{shlex.quote(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        check=False,
        timeout=timeout,
    )
    await sb.exec(
        f"python3 {swepro_dir}/parser.py {stdout_f} {stderr_f} {result_f}",
        check=False,
        timeout=120,
    )
    raw = await sb.read_file(result_f)
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return (1.0 if solved else 0.0), solved
