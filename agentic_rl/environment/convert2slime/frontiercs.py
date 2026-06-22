"""Convert Frontier-CS harbor tasks into slime rollout rows (offline CLI).

Frontier-CS ships as harbor tasks (per-task Dockerfile + ``tests/evaluate.py``
that POSTs a C++ submission to a go-judge). This reuses the harbor converter
(``convert2slime/harbor.py``) and adds the frontier-cs prep so the tasks run under
slime's single-sandbox ``HarborEnv`` instead of the autoresearch docker-compose:

  1. strip ``environment/docker-compose.yaml`` (slime boots one sandbox; the judge
     is a separate verifier server reached over the network);
  2. template ``verifier.env.JUDGE_URL = ${FRONTIER_CS_JUDGE_URL}`` (the rollout
     head resolves it to the booted verifier server);
  3. clean ``[task].name`` to the dir name (tidy instance ids / task_path);
  4. replace ``tests/evaluate.py`` with the canonical slime verifier — grades the
     FINAL ``/app/solution.cpp`` (no best-of) and writes a RAW reward signal
     (``score_raw``, ``cases_passed/total``, ``is_solved``) for central shaping.

Then it sets ``metadata.task_type = "frontier_cs"`` (→ ``FrontierCsEnv``), re-roots
``task_path`` under ``frontier_cs/tasks/<id>``, and splits train/eval.

    python -m agentic_rl.environment.convert2slime.frontiercs \\
        --tasks-dir /path/to/multi-agent-autoresearch/tasks/frontier-cs-algorithm \\
        --out-dir /tmp/pub/frontier_cs --eval-n 38 --seed 0

Writes ``<out-dir>/{frontier_cs_train,frontier_cs_eval}.jsonl`` + ``<out-dir>/tasks/<id>/``.
Publish ``<out-dir>`` to HF at ``path_in_repo="frontier_cs"`` (see convert2slime/README.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from . import harbor as harbor_conv

logger = logging.getLogger(__name__)

TASK_TYPE = "frontier_cs"
FAMILY = "frontier_cs"
_COMPOSE = ("docker-compose.yaml", "docker-compose.yml")
_JUDGE_URL_RE = re.compile(r'JUDGE_URL\s*=\s*"[^"]*"')
_TASK_NAME_RE = re.compile(r'^name\s*=\s*"frontier-cs/[^"]*"', re.MULTILINE)

# Canonical slime verifier, written into each task's tests/evaluate.py. Grades the
# FINAL /app/solution.cpp against the go-judge and emits a RAW signal; the head
# shapes it (rewards.py). No best-of; no docker-compose 'judge' assumptions.
CANONICAL_EVALUATE_PY = '''#!/usr/bin/env python3
"""Slime Frontier-CS verifier: grade the FINAL /app/solution.cpp via the go-judge.

Writes /logs/verifier/reward.json with a RAW signal (score_raw, cases_passed,
cases_total, is_solved) + a default fractional reward; the slime head re-shapes
from the raw fields (ASYNC_RL_REWARD_SHAPE). Grading is on the final file only —
the agent's submit.sh calls are iteration feedback, not the graded artifact.
"""
import json
import os
import time
from pathlib import Path

import requests

JUDGE_URL = os.environ.get("JUDGE_URL", "")
PROBLEM_ID = os.environ.get("PROBLEM_ID", "")
SOLUTION = Path("/app/solution.cpp")
REWARD_JSON = Path("/logs/verifier/reward.json")
REWARD_TXT = Path("/logs/verifier/reward.txt")
POLL_INTERVAL = 2
MAX_POLL_TIME = int(os.environ.get("MAX_POLL_TIME", "600"))


def write(reward, *, score_raw=0.0, cases_passed=0, cases_total=0, is_solved=False, detail=""):
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps({
        "reward": reward, "score_raw": score_raw, "cases_passed": cases_passed,
        "cases_total": cases_total, "is_solved": is_solved, "detail": detail,
    }))
    REWARD_TXT.write_text(str(reward))
    print(f"[verify] reward={reward:.4f} score_raw={score_raw} cases={cases_passed}/{cases_total} "
          f"solved={is_solved} {detail}")


def main():
    if not JUDGE_URL:
        write(0.0, detail="JUDGE_URL unset")
        return
    if not SOLUTION.exists() or not SOLUTION.read_text().strip():
        write(0.0, detail="no solution.cpp")
        return
    code = SOLUTION.read_text()

    # wait for judge readiness
    for _ in range(30):
        try:
            if requests.get(f"{JUDGE_URL}/problems", timeout=5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)

    try:
        r = requests.post(f"{JUDGE_URL}/submit", files={"code": ("solution.cpp", code)},
                          data={"pid": PROBLEM_ID, "lang": "cpp"}, timeout=30)
        r.raise_for_status()
        sid = r.json()["sid"]
    except Exception as e:
        write(0.0, detail=f"submit failed: {e}")
        return

    result, start = None, time.time()
    while time.time() - start < MAX_POLL_TIME:
        try:
            resp = requests.get(f"{JUDGE_URL}/result/{sid}", timeout=10)
            if resp.status_code == 200 and resp.json().get("status") in ("done", "error"):
                result = resp.json()
                break
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)

    if result is None:
        write(0.0, detail="judge timed out")
        return
    if result.get("status") == "error":
        write(0.0, detail=str(result.get("message") or result.get("error") or "judge error"))
        return

    score_raw = float(result.get("score") or 0.0)  # 0..100
    cases = result.get("cases") or []
    cases_total = len(cases) if isinstance(cases, list) else 0
    cases_passed = sum(1 for c in cases if isinstance(c, dict) and (c.get("scoreRatio") or 0) >= 1.0)
    is_solved = bool(result.get("passed")) or score_raw >= 100.0 - 1e-6
    write(max(0.0, min(1.0, score_raw / 100.0)), score_raw=score_raw, cases_passed=cases_passed,
          cases_total=cases_total, is_solved=is_solved)


if __name__ == "__main__":
    main()
'''


def _prep_task(task_dir: Path) -> None:
    """In-place prep of a copied task dir so slime's harbor converter accepts it."""
    env_dir = task_dir / "environment"
    for name in _COMPOSE:
        (env_dir / name).unlink(missing_ok=True)
    toml = task_dir / "task.toml"
    text = toml.read_text(encoding="utf-8")
    text = _JUDGE_URL_RE.sub('JUDGE_URL = "${FRONTIER_CS_JUDGE_URL}"', text)
    text = _TASK_NAME_RE.sub(f'name = "{task_dir.name}"', text, count=1)
    toml.write_text(text, encoding="utf-8")
    (task_dir / "tests" / "evaluate.py").write_text(CANONICAL_EVALUATE_PY, encoding="utf-8")


def convert(task_dirs: list[Path], out_dir: Path, *, eval_n: int, seed: int, limit: int | None) -> tuple[int, int]:
    tasks_out = out_dir / "tasks"
    tasks_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for src in task_dirs:
        if limit is not None and len(rows) >= limit:
            break
        dest = tasks_out / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=harbor_conv._COPY_IGNORE)
        _prep_task(dest)
        try:
            row = harbor_conv.translate_task(dest, dataset=FAMILY)
        except harbor_conv.SkipTask as e:
            skipped += 1
            shutil.rmtree(dest, ignore_errors=True)
            logger.warning("[frontiercs2slime] skip %s: %s", src.name, e)
            continue
        instance_id = row["metadata"]["instance_id"]
        if instance_id != dest.name:  # keep tasks/<id> aligned with the baked instance_id
            aligned = tasks_out / instance_id
            if aligned.exists():
                shutil.rmtree(aligned)
            dest.rename(aligned)
        row["metadata"]["task_type"] = TASK_TYPE
        row["metadata"]["task_path"] = f"{FAMILY}/tasks/{instance_id}"
        rows.append(row)

    if not rows:
        logger.error("[frontiercs2slime] no tasks converted under the given --tasks-dir")
        return 0, skipped

    # Uniform per-dataset repo layout: train.jsonl + eval.jsonl (published under
    # path_in_repo=frontier_cs; see datasets.py / convert2slime/README.md).
    random.Random(seed).shuffle(rows)
    eval_n = max(0, min(eval_n, len(rows)))
    _write_jsonl(out_dir / "eval.jsonl", rows[:eval_n])
    _write_jsonl(out_dir / "train.jsonl", rows[eval_n:])
    return len(rows), skipped


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Convert Frontier-CS harbor tasks to slime JSONL + task dirs.")
    parser.add_argument("--tasks-dir", required=True, help="dir of frontier-cs-algorithm-<n> harbor task dirs")
    parser.add_argument("--out-dir", required=True, help="output dir (JSONL + tasks/); publish to HF frontier_cs/")
    parser.add_argument("--eval-n", type=int, default=38, help="held-out eval tasks (rest are train)")
    parser.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    parser.add_argument("--limit", type=int, default=0, help="cap total tasks (0 = all; for smoke tests)")
    args = parser.parse_args(argv)

    tasks_dir = Path(args.tasks_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    task_dirs = harbor_conv._discover_task_dirs(tasks_dir)
    if not task_dirs:
        raise SystemExit(f"no task dirs (with task.toml) under {tasks_dir}")

    converted, skipped = convert(task_dirs, out_dir, eval_n=args.eval_n, seed=args.seed, limit=args.limit or None)
    if not converted:
        return 1
    n_eval = max(0, min(args.eval_n, converted))
    print(
        f"converted {converted} frontier-cs tasks ({skipped} skipped) -> {out_dir}\n"
        f"  train: {converted - n_eval} -> {out_dir / 'train.jsonl'}\n"
        f"  eval:  {n_eval} -> {out_dir / 'eval.jsonl'}\n"
        "next: also copy the 2.5GB problems/ into out-dir, then publish out-dir to HF\n"
        "  with path_in_repo=frontier_cs (see convert2slime/README.md §B)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
