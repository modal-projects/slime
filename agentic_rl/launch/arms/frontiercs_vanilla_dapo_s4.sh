#!/usr/bin/env bash
# Arm 1/3 — VANILLA DAPO/GRPO baseline (the definitive no-retro control).
#
# Stock slime fully-async rollout + plain agentic generate: no retro lane, no
# snapshot capture, no manifest store. Staleness knobs mirror "a normal async
# RL run with max staleness 4": prefetch 4 (capacity) + hard behavior-lag
# gate 4 (enforced).
#
# Inference stack = the 2026-08-24 canon, baked into the launcher defaults:
# slimerl/slime:nightly-dev-20260810a-cu129 (sglang 0.5.15.post1), 4xTP2/node,
# EAGLE 3/1/4, deterministic OFF. rollout_sim: 844 tok/GPU/s vs 728 on the old
# 0.5.12 image; 0.5.18 (+16% more) is blocked until upstream slime moves to
# the cu13/torch-2.13 stack — do NOT set SGLANG_VERSION here.
# 100 updates, checkpoint every 10 (launcher defaults) -> endpoint iter_0000099.
#
# Pair with frontiercs_retro0_s4.sh: identical except that arm runs the retro
# mixed path at ratio 0 (fresh-only batches but capture still on), so the pair
# A/Bs the retro harness overhead at dose zero.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export ROLLOUT_MODE=vanilla
export WANDB_GROUP=qwen3.6-27b-frontier-cs-vanilla-dapo-s4
export RETRO_PHASE2_GROUPS=32
export RETRO_PHASE2_ROLLOUTS=100
export ROLLOUT_PREFETCH_BATCHES=4
export FRESH_MAX_BEHAVIOR_LAG=4

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (vanilla fully-async, prefetch 4, gate 4, 20260810a/sglang 0.5.15.post1, det off)"
uv run --with modal modal run -d agentic_rl/launch/modal_train.py::train
