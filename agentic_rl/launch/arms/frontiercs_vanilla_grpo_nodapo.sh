#!/usr/bin/env bash
# Baseline addition — vanilla GRPO WITHOUT the DAPO dynamic-sampling filter.
#
# Identical to frontiercs_vanilla_dapo_s4.sh except DAPO_FILTER=0: every
# generated group trains (zero-std groups contribute zero advantage and dilute
# the batch instead of costing extra generation). Against the vanilla-DAPO arm
# this isolates exactly one question: does DAPO's adaptive oversampling
# (currently ~2.2x generation per trained batch) buy reward, at what wall cost?
# Scoring note: with the filter off, rollout/raw_reward IS the unbiased fresh
# pre-filter mean (nothing is filtered); compare against the DAPO arm's
# dynamic_sampling/raw_reward_all, which has the same denominator semantics.
# Same canon stack (20260810a / 0.5.15.post1 / 4xTP2 / EAGLE 3/1/4 / det off),
# prefetch 4, gate 4, 100 updates, save every 10.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export ROLLOUT_MODE=vanilla
export DAPO_FILTER=0
export WANDB_GROUP=qwen3.6-27b-frontier-cs-vanilla-grpo-nodapo
export RETRO_PHASE2_GROUPS=32
export RETRO_PHASE2_ROLLOUTS=100
export ROLLOUT_PREFETCH_BATCHES=4
export FRESH_MAX_BEHAVIOR_LAG=4

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (vanilla GRPO, DAPO off, prefetch 4, gate 4)"
uv run --with modal modal run -d agentic_rl/launch/modal_train.py::train
