#!/usr/bin/env bash
# Arm 2/3 — retro stack at retro dose ZERO (harness-parity twin of arm 1).
#
# Exactly the retro mixed rollout path, but RETRO_GROUP_RATIO=0: the retro leg
# returns immediately (retro_group_split -> target 0) and the fresh leg gets
# the whole 32-group batch under the same prefetch-4 pool and lag-4 gate as
# the vanilla arm. Remaining deltas vs arm 1, by construction: in-episode
# snapshot capture stays ON (tar of /app around the 50% turn on qualifying
# episodes), plus manifest-store bookkeeping. If step-100 matches arm 1, the
# retro harness overhead is confirmed neutral and vanilla is a valid control
# for every retro arm. (Strict bit-parity variant: add RETRO_MIN_TURN=9999 to
# suppress capture — but then this arm adds nothing over arm 1.)
#
# Same 2026-08-24 canon inference stack as arm 1 (launcher defaults:
# 20260810a image / sglang 0.5.15.post1 / 4xTP2 / EAGLE 3/1/4 / det off).
# 100 updates, save every 10.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export WANDB_GROUP=qwen3.6-27b-frontier-cs-retro0-s4
export RETRO_PHASE2_GROUPS=32
export RETRO_PHASE2_ROLLOUTS=100
export ROLLOUT_PREFETCH_BATCHES=4
export FRESH_MAX_BEHAVIOR_LAG=4
export RETRO_GROUP_RATIO=0
export RETRO_PREFETCH_BATCHES=0
export RETRO_MAX_BEHAVIOR_LAG=1

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (retro path @ ratio 0, prefetch 4, gate 4, 20260810a/sglang 0.5.15.post1, det off)"
uv run --with modal modal run -d agentic_rl/launch/modal_train.py::train
