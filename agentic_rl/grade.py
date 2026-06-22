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


def grade(task: dict, model_patch: str, *, timeout: int = 1800) -> float:
    cfg = task["install_config"]
    workdir = "/" + task["repo"].split("/")[1]
    test_cmds = cfg["test_cmd"] if isinstance(cfg["test_cmd"], list) else [cfg["test_cmd"]]
    apply = "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn"  # upstream eval flags

    sandbox = Sandbox(task["image_name"], cwd=workdir, lifetime=timeout)
    try:
        sandbox.write_file("/tmp/model.patch", model_patch)
        sandbox.write_file("/tmp/test.patch", task["test_patch"])
        script = "\n".join(
            ["set -e", "git reset --hard HEAD", f"{apply} /tmp/model.patch", f"{apply} /tmp/test.patch", *test_cmds]
        )
        _, output = sandbox.exec(script, cwd=workdir, timeout=timeout)
    finally:
        sandbox.terminate()

    passed = _passed_tests(output)
    if not task["FAIL_TO_PASS"]:
        return 0.0  # no failing test to fix → not a valid task; `all([])` would falsely resolve
    resolved = all(t in passed for t in task["FAIL_TO_PASS"]) and all(t in passed for t in task["PASS_TO_PASS"])
    return 1.0 if resolved else 0.0
