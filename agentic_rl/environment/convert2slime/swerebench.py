"""Materialize ``nebius/SWE-rebench-V2`` as slime prompt data (harbor format).

SWE-rebench-V2 is a SWE-bench-style dataset (``instance_id`` / ``repo`` /
``base_commit`` / ``patch`` / ``test_patch`` / ``FAIL_TO_PASS`` / ``PASS_TO_PASS``
+ a prebuilt ``image_name`` + an ``install_config`` carrying ``test_cmd`` and a
named ``log_parser``). slime's runtime (``environment/harbor.py``) only ingests
*harbor* task dirs, so each HF row is rendered into a harbor task dir here and
handed to the shared ``harbor.convert`` (same output as every other dataset:
``<key>.jsonl`` + ``tasks/<instance_id>/``).

Per task dir we emit::

    task.toml                # harbor config (docker_image=image_name, timeouts, cpus/mem)
    instruction.md           # problem_statement (agent-facing)
    tests/config.json        # the instance (FAIL_TO_PASS/PASS_TO_PASS/test_patch)
    tests/test.sh            # reset tests -> apply test_patch -> test_cmd -> grade
    tests/grade.py           # stdlib pytest grader -> /logs/verifier/reward.json
    solution/solve.sh        # apply the gold ``patch`` (oracle check only)

The prebuilt ``image_name`` is used DIRECTLY as ``docker_image`` (no Dockerfile
build). Verified image conventions (NOT swebench's): the repo is at
``/<repo-basename>`` (also the image's WORKDIR, so ``workdir`` is left unset and
harbor detects it; scripts never cd), and these are plain system-python images
(repo pip-installed system-wide, pytest on PATH) — not ``/testbed`` + conda.

The eval contract mirrors the proven ``swebench-verified`` harbor template
(reset the test files, ``git apply`` the test patch, run the test command, parse
the log for F2P/P2P resolution). The grader is a SELF-CONTAINED stdlib port of
``swebench.harness.log_parsers.python.parse_log_pytest`` + ``grading`` — we do
NOT pip-install the harness at verify time (no network, lower rollout latency).

Scope: PYTHON ONLY. The public ``SWE-rebench/SWE-bench-fork`` registers only the
Python ``log_parser`` names (``parse_log_pytest*``) as string keys; the
multilingual registry (``parse_log_gotest``, ``parse_log_cargo``, ...) is not
public. ``--language`` defaults to ``python`` and other values are refused until
a matching grader is added (see ``_GRADERS``).

    python -m agentic_rl.environment.convert2slime.swerebench \
        --out-dir data/swe_rebench_v2 --eval-n 200 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from agentic_rl.environment.convert2slime import harbor

logger = logging.getLogger(__name__)

HF_DATASET = "nebius/SWE-rebench-V2"
DATASET_KEY = "swe_rebench_v2"  # /data subdir == task_path prefix == registry key

# The prebuilt images check the repo out here and put toolchains on PATH via ENV.
# Nebius's published images check the repo out at /<repo-basename> (also the
# image's default WORKDIR) and put the toolchain on PATH — NOT swebench's /testbed
# + conda. So we use ``image_name`` directly and let harbor DETECT the workdir
# (``pwd`` -> the repo dir); scripts must not hardcode a cd.

# Languages we can grade today. Each maps to a log_parser family we vendor in
# ``_GRADE_PY``. Extend by adding a grader + relaxing this set.
_SUPPORTED_LANGUAGES = {"python"}
# install_config.log_parser values this converter knows how to grade.
_SUPPORTED_PARSERS = {"parse_log_pytest", "parse_log_pytest_options", "parse_log_pytest_v2"}

# Pull only the b/ path out of each ``diff --git a/X b/Y`` header.
_DIFF_FILE_RE = re.compile(r"^diff --git a/.+? b/(.+)$", re.MULTILINE)


class SkipRow(Exception):
    """Row is out of scope for this converter; str(err) is the logged reason."""


# ---------------------------------------------------------------------------
# Quality filter (meta.llm_metadata)
# ---------------------------------------------------------------------------
def _passes_quality(row: dict[str, Any], min_grade: str | None) -> bool:
    """Keep rows whose LLM quality grade is >= ``min_grade`` (A best) and that
    carry no detected B-issues. No filter when ``min_grade`` is None."""
    if not min_grade:
        return True
    md = (row.get("meta") or {}).get("llm_metadata") or {}
    grade = (md.get("code") or "").strip().upper()
    if not grade or grade > min_grade.strip().upper():  # 'A' < 'B' < 'C' ...
        return False
    issues = md.get("detected_issues") or {}
    return not any(bool(v) for v in issues.values())


# ---------------------------------------------------------------------------
# File renderers
# ---------------------------------------------------------------------------
def _safe_id(instance_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", instance_id) or "task"


def _test_files(test_patch: str) -> list[str]:
    """Files the test patch touches (reset to base + apply, like swebench)."""
    return _DIFF_FILE_RE.findall(test_patch or "")


def _repo_workdir(repo: str) -> str:
    """The repo checkout dir inside the prebuilt swerebench image (= image WORKDIR).

    Nebius's images check the repo out at ``/<repo-basename>`` (the basename of the
    ``org/repo`` slug), which is also the image's default WORKDIR. ``test.sh`` /
    ``solve.sh`` use repo-RELATIVE paths, so the verifier MUST run in this dir.

    We emit it EXPLICITLY into task.toml rather than relying on harbor's runtime
    ``pwd`` detection: harbor builds the sandbox with ``cwd="/"`` when ``workdir`` is
    unset (``harbor.py``), and every ``exec`` is prefixed with ``cd /`` (``sandbox.py``),
    so ``_detect_workdir`` would always return "/" and the verifier would run
    ``test.sh`` from "/" -> repo-relative ``git``/``pytest`` fail -> reward 0 for every
    rollout. This matches the standalone ``environment/swerebench.py`` grader, which
    already uses ``"/" + repo.split("/")[1]``.
    """
    name = (repo or "").rstrip("/").split("/")[-1]
    if not name:
        raise SkipRow(f"cannot derive workdir from repo {repo!r}")
    return "/" + name


def _task_toml(instance_id: str, image_name: str, workdir: str) -> str:
    # Prebuilt image used directly (no Dockerfile build). ``workdir`` is the repo
    # checkout dir (= image WORKDIR); harbor runs the agent and the verifier there.
    # Set it EXPLICITLY -- harbor's ``pwd`` auto-detect resolves to "/" for these
    # tasks (see ``_repo_workdir``). ``build_timeout_sec`` here covers the image pull.
    return (
        "[task]\n"
        f'name = "{instance_id}"\n\n'
        "[verifier]\n"
        "timeout_sec = 1800\n\n"
        "[agent]\n"
        "timeout_sec = 1800\n\n"
        "[environment]\n"
        f'docker_image = "{image_name}"\n'
        f'workdir = "{workdir}"\n'
        "build_timeout_sec = 1800.0\n"
        "cpus = 2\n"
        "memory_mb = 8192\n"
    )


_GRADE_PY = r'''#!/usr/bin/env python3
"""Grade a SWE-rebench-V2 (pytest) run: parse the test log for FAIL_TO_PASS /
PASS_TO_PASS resolution and write /logs/verifier/reward.json.

Self-contained stdlib port of swebench.harness.log_parsers.python.parse_log_pytest
+ grading.get_eval_tests_report / get_resolution_status (PASS_AND_FAIL eval type).
"""
import json
import sys

# swebench TestStatus values.
PASSED, FAILED, SKIPPED, ERROR, XFAIL, XPASS = (
    "PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL", "XPASS",
)
_STATUSES = (PASSED, FAILED, SKIPPED, ERROR, XFAIL, XPASS)


def parse_log_pytest(log):
    """line.startswith(<STATUS>) -> {test_id: status} (pytest -rA summary lines)."""
    sm = {}
    for line in log.split("\n"):
        if not any(line.startswith(s) for s in _STATUSES):
            continue
        if line.startswith(FAILED):
            line = line.replace(" - ", " ")
        parts = line.split()
        if len(parts) <= 1:
            continue
        sm[parts[1]] = parts[0]
    return sm


def _passed(case, sm):
    return case in sm and sm[case] in (PASSED, XFAIL)


def main():
    log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    cfg = json.load(open("/tests/config.json", encoding="utf-8"))
    f2p, p2p = cfg["FAIL_TO_PASS"], cfg["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

    sm = parse_log_pytest(log)
    targets = list(f2p) + list(p2p)
    passed = [t for t in targets if _passed(t, sm)]
    cases_total = len(targets)
    cases_passed = len(passed)
    # Resolved (SWE-bench): every FAIL_TO_PASS and PASS_TO_PASS test passes.
    resolved = all(_passed(t, sm) for t in f2p) and all(_passed(t, sm) for t in p2p)
    fraction = (cases_passed / cases_total) if cases_total else 0.0

    report = {
        # rewards.py: score_raw (0..100) drives `fractional`; is_solved is the
        # clean solve flag; cases_* are diagnostics.
        "score_raw": 100.0 * fraction,
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "is_solved": bool(resolved),
        "reward": 1.0 if resolved else fraction,
        "fail_to_pass_passed": sum(1 for t in f2p if _passed(t, sm)),
        "fail_to_pass_total": len(f2p),
        "pass_to_pass_passed": sum(1 for t in p2p if _passed(t, sm)),
        "pass_to_pass_total": len(p2p),
    }
    with open("/logs/verifier/reward.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[grade] resolved={resolved} cases={cases_passed}/{cases_total}")
    sys.exit(0 if resolved else 1)


if __name__ == "__main__":
    main()
'''


def _test_sh(*, base_commit: str, test_patch: str, test_cmd: str, test_files: list[str]) -> str:
    import shlex

    files_arg = " ".join(shlex.quote(f) for f in test_files) if test_files else ""
    reset = (
        f"git checkout {shlex.quote(base_commit)} -- {files_arg} 2>/dev/null || true"
        if files_arg
        else 'echo "no test files to reset"'
    )
    # Remove untracked test files the patch would otherwise refuse to overwrite,
    # without touching source files the agent added.
    rm_untracked = (
        "\n".join(
            [
                f"for path in {files_arg}; do",
                '    if [ -e "$path" ] && ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then',
                '        rm -rf -- "$path"',
                "    fi",
                "done",
            ]
        )
        if files_arg
        else ""
    )
    return f"""#!/bin/bash
set -uo pipefail

# --- Activate the repo's environment (swerebench python images: conda 'testbed').
# Non-fatal: if the image instead installs the repo system-wide, pytest is already
# on PATH. (A non-login shell never sources ~/.bashrc, so we activate explicitly.)
for _c in /opt/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh; do
  if [ -f "$_c" ]; then . "$_c" && conda activate testbed 2>/dev/null || true; break; fi
done

# harbor runs us inside the detected workdir (= the repo dir); don't cd elsewhere.
git config --global --add safe.directory '*'

# --- Restore the test files to base, then apply the held-out test patch.
{reset}
{rm_untracked}
git apply --verbose - <<'EOF_SWEREBENCH_TESTPATCH'
{test_patch}
EOF_SWEREBENCH_TESTPATCH

# --- Run the tests, capturing stdout+stderr.
LOG_FILE=$(mktemp)
export LOG_FILE
set +e
{{ {test_cmd} ; }} > >(tee "$LOG_FILE") 2>&1
set -e

# --- Revert the test files so the repo is left as the agent had it.
{reset}

# --- Grade (writes /logs/verifier/reward.json; exit 0 iff fully resolved).
python3 /tests/grade.py "$LOG_FILE"
"""


def _solve_sh(patch: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail

# Oracle check only: apply the dataset's gold patch, then the verifier should
# report the issue resolved (reward 1.0). Never uploaded during agent rollouts.
# harbor runs us inside the detected workdir (= the repo dir); don't cd elsewhere.
git config --global --add safe.directory '*'
cat > /tmp/swerebench_gold.patch <<'EOF_SWEREBENCH_GOLDPATCH'
{patch}
EOF_SWEREBENCH_GOLDPATCH
# Strict first, then 3-way, then fuzzy `patch` (gold diffs occasionally need fuzz).
git apply --verbose /tmp/swerebench_gold.patch \
  || git apply --3way /tmp/swerebench_gold.patch \
  || patch -p1 --fuzz=3 -i /tmp/swerebench_gold.patch
"""


def _config_json(row: dict[str, Any]) -> str:
    """The instance the grader reads (FAIL_TO_PASS/PASS_TO_PASS as JSON strings,
    matching the swebench config.json convention; the grader accepts both)."""
    cfg = {
        "repo": row["repo"],
        "instance_id": row["instance_id"],
        "base_commit": row["base_commit"],
        "version": str(row.get("created_at") or "0"),
        "FAIL_TO_PASS": json.dumps(list(row.get("FAIL_TO_PASS") or [])),
        "PASS_TO_PASS": json.dumps(list(row.get("PASS_TO_PASS") or [])),
        "language": row.get("language"),
        "image_name": row.get("image_name"),
        "install_config": row.get("install_config"),
    }
    return json.dumps(cfg, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# One row -> one staged harbor task dir
# ---------------------------------------------------------------------------
def build_task_dir(row: dict[str, Any], dest: Path) -> None:
    """Render the harbor task dir for one HF row. Raises ``SkipRow`` if unusable."""
    lang = (row.get("language") or "").strip().lower()
    if lang not in _SUPPORTED_LANGUAGES:
        raise SkipRow(f"unsupported language {lang!r}")
    install_config = row.get("install_config") or {}
    parser = install_config.get("log_parser")
    if parser not in _SUPPORTED_PARSERS:
        raise SkipRow(f"unsupported log_parser {parser!r}")
    test_cmd = install_config.get("test_cmd")
    if not test_cmd:
        raise SkipRow("no install_config.test_cmd")
    image_name = row.get("image_name")
    if not image_name:
        raise SkipRow("no image_name")
    if not (row.get("FAIL_TO_PASS")):
        raise SkipRow("empty FAIL_TO_PASS")
    test_patch = row.get("test_patch") or ""
    if not test_patch.strip():
        raise SkipRow("empty test_patch")

    instance_id = _safe_id(row["instance_id"])

    (dest / "tests").mkdir(parents=True, exist_ok=True)
    (dest / "solution").mkdir(parents=True, exist_ok=True)

    workdir = _repo_workdir(row.get("repo"))
    (dest / "task.toml").write_text(_task_toml(instance_id, image_name, workdir), encoding="utf-8")
    (dest / "instruction.md").write_text(row.get("problem_statement") or "", encoding="utf-8")
    (dest / "tests" / "config.json").write_text(_config_json(row), encoding="utf-8")
    (dest / "tests" / "grade.py").write_text(_GRADE_PY, encoding="utf-8")
    test_sh = dest / "tests" / "test.sh"
    test_sh.write_text(
        _test_sh(
            base_commit=row["base_commit"],
            test_patch=test_patch,
            test_cmd=test_cmd if isinstance(test_cmd, str) else " && ".join(test_cmd),
            test_files=_test_files(test_patch),
        ),
        encoding="utf-8",
    )
    solve_sh = dest / "solution" / "solve.sh"
    solve_sh.write_text(_solve_sh(row.get("patch") or ""), encoding="utf-8")
    for f in (test_sh, solve_sh):
        f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# HF load + materialize
# ---------------------------------------------------------------------------
def load_hf_rows(repo: str, split: str, limit: int | None) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `datasets` to pull SWE-rebench-V2 from HuggingFace.") from exc

    # Streaming so we can filter to a small in-scope subset without materializing 32k rows.
    yielded = 0
    for row in load_dataset(repo, split=split, streaming=True):
        if limit is not None and yielded >= limit:
            break
        yield dict(row)
        yielded += 1


def materialize(
    out_dir: Path,
    *,
    key: str = DATASET_KEY,
    repo: str = HF_DATASET,
    split: str = "train",
    language: str = "python",
    min_grade: str | None = "A",
    eval_n: int = 0,
    seed: int = 0,
    eval_only: bool = False,
    limit: int | None = None,
    scan_limit: int | None = None,
) -> tuple[int, int, int]:
    """Stream HF rows -> staged harbor task dirs -> ``harbor.convert``.

    ``language`` / ``min_grade`` filter rows before rendering; ``limit`` caps
    accepted (in-scope) tasks while ``scan_limit`` caps rows examined. Returns
    ``(n_train, n_eval, skipped)``.
    """
    if language not in _SUPPORTED_LANGUAGES:
        raise SystemExit(
            f"language={language!r} unsupported; this converter grades {sorted(_SUPPORTED_LANGUAGES)} "
            "only (the public SWE-bench-fork lacks string-keyed parsers for other languages)."
        )

    staging = out_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    task_dirs: list[Path] = []
    skipped = 0
    scanned = 0
    for row in load_hf_rows(repo, split, scan_limit):
        scanned += 1
        if limit is not None and len(task_dirs) >= limit:
            break
        if (row.get("language") or "").strip().lower() != language:
            skipped += 1
            continue
        if not _passes_quality(row, min_grade):
            skipped += 1
            continue
        dest = staging / _safe_id(row["instance_id"])
        try:
            build_task_dir(row, dest)
        except SkipRow as e:
            skipped += 1
            shutil.rmtree(dest, ignore_errors=True)
            logger.warning("[swerebench2slime] skip %s: %s", row.get("instance_id"), e)
            continue
        task_dirs.append(dest)
        if len(task_dirs) % 100 == 0:
            logger.info("[swerebench2slime] staged %d tasks (scanned %d)", len(task_dirs), scanned)

    logger.info("[swerebench2slime] staged %d tasks (scanned %d, skipped %d)", len(task_dirs), scanned, skipped)
    if not task_dirs:
        raise SystemExit("no in-scope rows staged (check --language / --min-grade / --limit)")
    try:
        n_train, n_eval, conv_skipped = harbor.convert(
            sorted(task_dirs), out_dir, key=key, dataset=DATASET_KEY,
            eval_n=eval_n, seed=seed, eval_only=eval_only,
        )
        return n_train, n_eval, skipped + conv_skipped
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True, help="output dir (JSONL + tasks/); use the slime-data volume")
    parser.add_argument("--key", default=DATASET_KEY, help="dataset key = task_path prefix + /data subdir")
    parser.add_argument("--repo", default=HF_DATASET, help="HuggingFace dataset repo id")
    parser.add_argument("--split", default="train", help="HuggingFace split")
    parser.add_argument("--language", default="python", help="keep rows with this language (python only, for now)")
    parser.add_argument(
        "--min-grade", default="A",
        help="keep rows with meta.llm_metadata.code <= this grade (A best) and no detected issues; 'none' disables",
    )
    parser.add_argument("--eval-n", type=int, default=0, help="rows held out into eval.jsonl (rest -> train.jsonl)")
    parser.add_argument("--eval-only", action="store_true", help="all rows -> eval.jsonl (no train.jsonl)")
    parser.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    parser.add_argument("--limit", type=int, help="max in-scope tasks to convert")
    parser.add_argument("--scan-limit", type=int, help="max HF rows to examine (before filtering)")
    args = parser.parse_args(argv)

    min_grade = None if (args.min_grade or "").lower() == "none" else args.min_grade
    n_train, n_eval, skipped = materialize(
        args.out_dir, key=args.key, repo=args.repo, split=args.split,
        language=args.language, min_grade=min_grade,
        eval_n=args.eval_n, seed=args.seed, eval_only=args.eval_only,
        limit=args.limit, scan_limit=args.scan_limit,
    )
    out_dir = args.out_dir.resolve()
    print(f"converted train={n_train} eval={n_eval} ({skipped} skipped) -> {out_dir} (task_path prefix '{args.key}')")
    print(f"next: oracle-check a few rows, then publish out-dir to HF with path_in_repo={args.key} (see README.md).")
    return 0 if (n_train + n_eval) else 1


if __name__ == "__main__":
    raise SystemExit(main())
