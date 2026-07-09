"""Grading in a FRESH sandbox (not the agent's dirtied one — prevents reward hacking via leftover
state). Two task formats, dispatched on metadata:
- SWE-rebench (default): apply the agent patch + held-out test patch, run pytest, reward 1.0 iff all
  FAIL_TO_PASS and PASS_TO_PASS pass. pytest only (dataset filtered to parse_log_pytest).
- R2E-Gym (task has ``expected_output_json``): run the image's baked run_tests.sh over the held-out
  /r2e_tests, compare per-test results to expected_output_json — exact-match binary + baseline F2P dense.
"""

import json
import re

from .sandbox import Sandbox


def _passed_tests(log: str) -> set[str]:
    """Test ids that PASSED in pytest output (mirrors SWE-rebench parse_log_pytest)."""
    passed = set()
    for line in log.splitlines():
        if line.startswith("PASSED"):
            parts = line.split()
            if len(parts) >= 2:
                passed.add(parts[1])
    return passed


def grade_detailed(task: dict, model_patch: str, *, timeout: int = 1800) -> dict:
    """Grade a patch and return the structured outcome (binary reward + baseline-relative dense reward
    + why). Runs the test suite TWICE in the same sandbox: once on the VANILLA repo (gold test patch
    only, no model patch) to measure the true base pass-set, then with the model patch. ``reward`` is
    the strict binary (all required pass) used for solved_frac; ``dense`` is the continuous training
    signal = fraction of genuinely-broken FAIL_TO_PASS the patch fixes, discounted by PASS_TO_PASS it
    regresses — both measured against the vanilla baseline so already-passing tests earn no credit and
    already-broken tests cost no penalty (handles label noise / env drift)."""
    if "expected_output_json" in task:  # R2E-Gym task → its own grading path
        return grade_r2e_detailed(task, model_patch, timeout=timeout)
    cfg = task["install_config"]
    workdir = "/" + task["repo"].split("/")[1]
    test_cmds = cfg["test_cmd"] if isinstance(cfg["test_cmd"], list) else [cfg["test_cmd"]]
    apply = "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn"  # upstream eval flags
    # Files the held-out test patch touches. After applying the model patch, reset these to base so the
    # agent's edits to them can't conflict with test.patch (the SWE-bench eval approach). Without it, a
    # model patch that modified a held-out test file makes `git apply test.patch` fail → the fix never gets
    # scored → false-negative reward 0. Existing files revert to HEAD; files test.patch creates are removed
    # so the create applies cleanly.
    test_files = [ln[6:].strip() for ln in task["test_patch"].splitlines() if ln.startswith("+++ b/")]
    reset = [f'git checkout HEAD -- "{f}" 2>/dev/null || rm -f "{f}"' for f in test_files]

    # lifetime must cover BOTH the base and patched test runs (each bounded by `timeout`) plus boot —
    # else a heavy-test task whose base run eats the budget gets its second exec reaped (NotFoundError).
    sandbox = Sandbox(task["image_name"], cwd=workdir, lifetime=2 * timeout + 120)
    try:
        sandbox.write_file("/tmp/model.patch", model_patch)
        sandbox.write_file("/tmp/test.patch", task["test_patch"])
        # BASE: vanilla repo + gold tests, no model patch → which required tests pass without any fix.
        base_script = "\n".join(["set -e", "git reset --hard HEAD", f"{apply} /tmp/test.patch", *test_cmds])
        _, base_output = sandbox.exec(base_script, cwd=workdir, timeout=timeout)
        # PATCHED: model patch + gold tests.
        script = "\n".join(
            [
                "set -e",
                "git reset --hard HEAD",
                f"{apply} /tmp/model.patch",
                *reset,
                f"{apply} /tmp/test.patch",
                *test_cmds,
            ]
        )
        _, output = sandbox.exec(script, cwd=workdir, timeout=timeout)
    finally:
        sandbox.terminate()

    base_passed = _passed_tests(base_output)
    passed = _passed_tests(output)
    f2p = list(task["FAIL_TO_PASS"])
    p2p = list(task["PASS_TO_PASS"])
    required = f2p + p2p
    missing = [t for t in required if t not in passed]
    # reward 1.0 iff there is a failing test to fix AND every required test now passes
    reward = 1.0 if (f2p and not missing) else 0.0

    # Baseline-relative dense reward. Credit only F2P that genuinely fail in vanilla shape and now pass;
    # discount only P2P that genuinely pass in vanilla shape and now regress. Multiplicative so a no-op
    # (fixes nothing) → 0 and a full clean solve → 1.0, continuous in between.
    fixable = [t for t in f2p if t not in base_passed]
    progress = (len([t for t in fixable if t in passed]) / len(fixable)) if fixable else (1.0 if reward else 0.0)
    p2p_base = [t for t in p2p if t in base_passed]
    p2p_frac = (len([t for t in p2p_base if t in passed]) / len(p2p_base)) if p2p_base else 1.0
    dense = round(progress * p2p_frac, 4)
    return {
        "reward": reward,
        "dense": dense,
        "passed": sorted(passed),
        "base_passed": sorted(base_passed),
        "required": required,
        "missing": missing,
        "n_fixable": len(fixable),
        "progress": round(progress, 4),
        "p2p_frac": round(p2p_frac, 4),
        "output": output,
    }


def grade(task: dict, model_patch: str, *, timeout: int = 1800) -> float:
    return grade_detailed(task, model_patch, timeout=timeout)["reward"]


# ── R2E-Gym grading ──────────────────────────────────────────────────────────
_R2E_REPO = "/testbed"

# Some R2E images force pytest color, and R2E even baked ANSI codes into expected_output_json keys for
# ~60% of pillow tasks. Unstripped, the colored 'PASSED <nodeid>' lines never match and the run parses
# as 0 tests → false reward 0 on a green run (verified live: 248 false-negative submissions, all pillow).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _parse_r2e(output: str) -> dict:
    """pytest -rA lines ('PASSED <nodeid>') → {normalized_id: PASSED|FAILED}. Normalize the nodeid
    'r2e_tests/test_1.py::Class::method' → 'Class.method' to match expected_output_json keys."""
    res = {}
    for ln in _strip_ansi(output).splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR)\s+(\S+)", ln.strip())
        if not m:
            continue
        st, nid = m.groups()
        # drop the file part ('r2e_tests/test_1.py::') and join Class::method as Class.method; a bare
        # nodeid with no '::' passes through unchanged.
        key = nid.split("::", 1)[-1].replace("::", ".")
        res[key] = "PASSED" if st == "PASSED" else "FAILED"
    return res


def grade_r2e_detailed(task: dict, model_patch: str, *, timeout: int = 1800) -> dict:
    """R2E-Gym grading in a fresh sandbox. The image ships a dirty tree (R2E's install/test setup) at
    /testbed + held-out tests at /r2e_tests + a baked run_tests.sh. We commit the setup-dirt as the base
    (so the model patch — a diff vs that same committed base in the agent's sandbox — applies cleanly),
    symlink the tests in, apply the patch, run run_tests.sh, and compare per-test results to
    expected_output_json. ``reward`` = R2E exact-match (every test matches expected, incl. expected-fails);
    ``dense`` = fraction of baseline-relative F2P (tests the fix must flip) the patch actually flips."""
    expected = task["expected_output_json"]
    if isinstance(expected, str):
        expected = json.loads(expected)
    base = task.get("base_results") or {}
    if isinstance(base, str):
        base = json.loads(base)
    # Normalize keys: R2E's stored keys carry ANSI codes on a subset of tasks (see _ANSI_RE above).
    expected = {_strip_ansi(k): v for k, v in expected.items()}
    base = {_strip_ansi(k): v for k, v in base.items()}

    sandbox = Sandbox(task["image_name"], cwd=_R2E_REPO, lifetime=timeout + 300)
    try:
        sandbox.write_file("/tmp/model.patch", model_patch)
        script = "\n".join(
            [
                "git config user.email r2e@local && git config user.name r2e",
                "git add -A && git commit -q -m r2e-base || true",  # same base the agent diffed against
                "ln -sf /r2e_tests /testbed/r2e_tests",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /tmp/model.patch "
                "|| git apply --whitespace=nowarn /tmp/model.patch || true",
                # Anti-reward-hack: the runner + hidden tests must be pristine AFTER the model patch. The
                # agent's repo had run_tests.sh deleted pre-commit, so no honest diff touches it — but a
                # crafted patch could rewrite it (it's tracked here) to print fake PASSED lines. Restore it
                # and re-assert the hidden-test symlink before running.
                "git checkout HEAD -- run_tests.sh 2>/dev/null || true",
                "rm -rf /testbed/r2e_tests && ln -s /r2e_tests /testbed/r2e_tests",
                "bash run_tests.sh 2>&1",
            ]
        )
        _, output = sandbox.exec(script, cwd=_R2E_REPO, timeout=timeout)
    finally:
        sandbox.terminate()

    parse = _parse_r2e(output)
    exact = len(parse) >= len(expected) and all(parse.get(k) == expected[k] for k in expected)
    reward = 1.0 if exact else 0.0
    # dense: baseline-relative F2P = tests whose base result differs from expected (what the fix must change).
    # Fall back to expected-PASSED tests if no baseline. dense = fraction of those the patch flips to expected.
    f2p = (
        [k for k in expected if base.get(k) != expected[k]]
        if base
        else [k for k in expected if expected[k] == "PASSED"]
    )
    fixed = [k for k in f2p if parse.get(k) == expected[k]]
    dense = round(len(fixed) / len(f2p), 4) if f2p else (1.0 if reward else 0.0)
    return {
        "reward": reward,
        "dense": dense,
        "passed": sorted(k for k, v in parse.items() if v == "PASSED"),
        "base_passed": sorted(k for k, v in base.items() if v == "PASSED"),
        "required": list(expected),
        "missing": [k for k in expected if parse.get(k) != expected[k]],
        "n_fixable": len(f2p),
        "progress": dense,
        "p2p_frac": 1.0,
        "output": output,
    }
