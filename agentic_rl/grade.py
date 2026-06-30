"""SWE-rebench grading in a FRESH sandbox (not the agent's dirtied one — prevents reward
hacking via leftover state): apply the agent patch + held-out test patch, run the tests,
reward 1.0 iff all FAIL_TO_PASS and PASS_TO_PASS pass. pytest only (dataset filtered to
parse_log_pytest); add a parser for other frameworks.
"""

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
    cfg = task["install_config"]
    workdir = "/" + task["repo"].split("/")[1]
    test_cmds = cfg["test_cmd"] if isinstance(cfg["test_cmd"], list) else [cfg["test_cmd"]]
    apply = "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn"  # upstream eval flags
    # Files the held-out test patch touches. After applying the model patch, reset these to base so the
    # agent's edits to them can't conflict with test.patch (the SWE-bench eval approach). Without it, a
    # model patch that modified a held-out test file makes `git apply test.patch` fail -> the fix never
    # gets scored -> false-negative reward 0 (verified: 2/3 sampled tasks). Existing files revert to HEAD;
    # files test.patch creates are removed so the create applies cleanly.
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
