#!/usr/bin/env bash
# All-turns capture-overhead A/B — dose-0 twin of frontiercs_retro0_s4.sh.
#
# Identical arm except RETRO_CAPTURE_MODE=all_turns: every post-tool boundary
# is rsync-staged in-sandbox (~75 ms/turn measured), one tarball artifact +
# one checkpoint blob per admitted trajectory (~11 MiB measured), age-out GC
# at 8 updates. RETRO_GROUP_RATIO=0 keeps replay OFF, so any step-100 delta vs
# frontiercs_retro0_s4 is pure capture overhead — the gate before any
# lease-policy arm (RETRO_LEASE_POLICY) goes live.
#
# Same 2026-08-24 canon inference stack (launcher defaults: 20260810a image /
# sglang 0.5.15.post1 / 4xTP2 / EAGLE 3/1/4 / det off). 100 updates.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export WANDB_GROUP=qwen3.6-27b-frontier-cs-retro0-allturns
export RETRO_PHASE2_GROUPS=32
export RETRO_PHASE2_ROLLOUTS=100
export ROLLOUT_PREFETCH_BATCHES=4
export FRESH_MAX_BEHAVIOR_LAG=4
export RETRO_GROUP_RATIO=0
export RETRO_PREFETCH_BATCHES=0
export RETRO_MAX_BEHAVIOR_LAG=1
export RETRO_CAPTURE_MODE=all_turns
export RETRO_ARTIFACT_COMPRESSION=gzip
export RETRO_GC_MAX_AGE=8

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (all-turns capture @ dose 0 — overhead A/B vs retro0-s4)"
uv run --with modal modal run -d agentic_rl/launch/modal_train.py::train
