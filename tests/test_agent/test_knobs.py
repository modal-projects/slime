"""The knob registry (agentic_rl/knobs.py) must cover every knob the code reads."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentic_rl.knobs import REGISTRY, VALIDATED_PREFIXES, unknown_knobs, validate_environment

_READ_PATTERNS = [
    r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]+)"',
    r'os\.environ\[\s*"([A-Z][A-Z0-9_]+)"',
    r'os\.getenv\(\s*"([A-Z][A-Z0-9_]+)"',
    r'environ\.get\(\s*"([A-Z][A-Z0-9_]+)"',
    r'\benv\.get\(\s*"([A-Z][A-Z0-9_]+)"',
    r'_env_int\(\s*(?:env,\s*)?"([A-Z][A-Z0-9_]+)"',
    r'_env_float\(\s*(?:env,\s*)?"([A-Z][A-Z0-9_]+)"',
    r'_env_bool\(\s*"([A-Z][A-Z0-9_]+)"',
    r'_bounded_fraction\(\s*env,\s*"([A-Z][A-Z0-9_]+)"',
    # Named-constant reads: NAME_ENV = "..." consumed via os.environ.get(NAME_ENV).
    r'^[A-Z_]+_ENV\s*=\s*"([A-Z][A-Z0-9_]+)"',
]
# Reads the regexes cannot see (variable indirection).
_INDIRECT_READS = {
    "ASYNC_RL_ROLLOUT_PREFETCH_BATCHES",  # core/fully_async.py knob tuple
    "ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG",  # core/fully_async.py knob tuple
}
# Read in agentic_rl but owned by other tooling, not run knobs.
_EXEMPT = {
    "FRONTIER_CS_ALG_ROOT",  # convert2slime converter input
}


def _scan_read_knobs() -> set[str]:
    found = set(_INDIRECT_READS)
    for path in sorted((_REPO_ROOT / "agentic_rl").rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if "__pycache__" in rel or rel.startswith("agentic_rl/profiles/"):
            continue
        text = path.read_text()
        for pattern in _READ_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                found.add(match.group(1))
    return {name for name in found if name.startswith(VALIDATED_PREFIXES)} - _EXEMPT


def _scan_script_knobs() -> set[str]:
    found = set()
    for path in sorted((_REPO_ROOT / "agentic_rl" / "slime_scripts").glob("*.sh")):
        for match in re.finditer(
            r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]+)=", path.read_text(), re.MULTILINE
        ):
            found.add(match.group(1))
    return {name for name in found if name.startswith(VALIDATED_PREFIXES)}


def test_every_knob_read_in_code_is_registered():
    missing = sorted(_scan_read_knobs() - set(REGISTRY))
    assert not missing, f"register these in agentic_rl/knobs.py: {missing}"


def test_every_knob_set_by_launch_scripts_is_registered():
    missing = sorted(_scan_script_knobs() - set(REGISTRY))
    assert not missing, f"launch scripts set unregistered knobs: {missing}"


def test_registry_has_no_orphans_pointing_at_missing_consumers():
    for knob in REGISTRY.values():
        consumer = _REPO_ROOT / "agentic_rl" / knob.consumer
        assert consumer.is_file(), f"{knob.name} names a missing consumer {knob.consumer}"


def test_validation_flags_typos_with_suggestion():
    with pytest.raises(ValueError, match="RETRO_MIN_SCORE"):
        validate_environment({"RETRO_MIN_SCROE": "0.1"})
    assert unknown_knobs({"RETRO_MIN_SCORE": "0.1", "PATH": "/usr/bin"}) == []


def test_launch_rejects_typoed_knob_end_to_end():
    from agentic_rl.retro.launch_config import build_launch_configs

    with pytest.raises(ValueError, match="unknown environment knob"):
        build_launch_configs({"RETRO_PHASE2_GROUP": "32", "LAUNCH_STAMP": "20260826-000000"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
