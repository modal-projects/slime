#!/usr/bin/env bash
# Provision mini-swe-agent into an isolated uv venv. uv is our package manager;
# we NEVER fall back to the image's pip, because many task images (SWE-bench-Pro)
# ship a poisoned PIP_INDEX_URL pointing at a dead build-time mirror ->
# `pip install uv` hits "Connection refused .../uv/". Instead, mirror harbor's
# bootstrap: ensure curl via the OS package manager, install uv from the pinned
# astral script (falling back to pip ONLY with an explicit PyPI index), and
# retry network steps to absorb transient GitHub release resets. Measured across
# all 731 SWE-bench-Pro images at conc 50: 99.7% success (vs the original
# pip-fallback's ~52% on no-curl images). See profiles/PROVISIONING_VENV_VS_VOLUME.md.
#
# Config comes from the environment -- mini_swe_agent.py exports the host-resolved
# MSWE_AGENT_VENV / MSWE_AGENT_PYTHON_VERSION / MSWE_PIP_SPEC in front of this
# script. The defaults below keep it runnable (and shellcheck-able) standalone.
set -e

: "${MSWE_AGENT_VENV:=/opt/mswe-agent}"
: "${MSWE_AGENT_PYTHON_VERSION:=3.11}"
: "${MSWE_PIP_SPEC:=mini-swe-agent==2.3.1}"
VENV_PY="$MSWE_AGENT_VENV/bin/python"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
retry() { for i in 1 2 3; do bash -c "$1" && return 0; sleep $((i * 4)); done; return 1; }

if ! command -v uv >/dev/null 2>&1; then
  # Ensure curl via whatever package manager the image ships (best-effort: if
  # none works we still try the pip path below).
  command -v curl >/dev/null 2>&1 || retry "apt-get update && apt-get install -y curl || apk add --no-cache curl || yum install -y curl || dnf install -y curl" || true
  # uv via the pinned astral script (bypasses the image's pip entirely); the pip
  # fallback forces a clean PyPI index so a poisoned image config can't win, then
  # retries with --break-system-packages so PEP 668 ("externally-managed") images
  # don't hard-fail. Try plain first: older pip (<23) lacks the flag but also
  # doesn't enforce PEP 668, so the plain attempt already succeeds there.
  retry "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh" \
    || retry "python3 -m pip install --index-url https://pypi.org/simple --root-user-action=ignore uv || python3 -m pip install --index-url https://pypi.org/simple --root-user-action=ignore --break-system-packages uv"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

rm -rf "$MSWE_AGENT_VENV"
retry "uv venv --python $MSWE_AGENT_PYTHON_VERSION $MSWE_AGENT_VENV"
retry "uv pip install --python $VENV_PY $MSWE_PIP_SPEC"

# Verify the agent's REAL import (the top package doesn't pull in pydantic).
# Two distinct failure modes this guards against:
#  * `-P` (MANDATORY): the exec runs with cwd = the image WORKDIR (e.g.
#    /testbed), so a task repo whose root *is* an agent dependency shadows the
#    venv. The pydantic SWE-gym tasks ship /testbed/pydantic/, which the bare
#    `python -c` imports instead of the venv's pydantic -> repo-pydantic vs
#    venv-pydantic_core skew crashes ("no attribute 'dict_not_none'"). -P drops
#    cwd from sys.path -- the SAME guard the runner launch uses -- so the venv
#    wins. Deterministic: was hitting 100% of pydantic tasks as a loud
#    `exception:RuntimeError` from _ensure_provisioned.
#  * reinstall: a partial wheel (corrupt uv cache / index contention at high
#    conc) can leave a native ext unimportable; --reinstall --no-cache repairs
#    it. A still-broken venv then fails LOUDLY here (set -e) instead of as a
#    silent zero-turn adapter_session_empty.
for _ in 1 2 3; do MSWEA_SILENT_STARTUP=1 "$VENV_PY" -P -c 'import minisweagent.agents.default' 2>/dev/null && break; retry "uv pip install --python $VENV_PY --reinstall --no-cache $MSWE_PIP_SPEC" || true; done
MSWEA_SILENT_STARTUP=1 "$VENV_PY" -P -c 'import minisweagent.agents.default'
