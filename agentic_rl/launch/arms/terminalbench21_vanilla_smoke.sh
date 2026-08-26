#!/usr/bin/env bash
# Terminal-Bench 2.1 — vanilla fully-async SMOKE (family onboarding proof).
#
# First training arm on the terminal_bench family (RUNBOOK §7 step 7): stock
# fully-async rollout + plain agentic generate over the derived 69/20 split
# (seed 20260826; download_data materializes it on the volume — run that once
# first, then the oracle check; see agentic_rl/envs/terminal_bench/README.md).
#
# 4-group engineering smoke: 1 train node + 1 rollout node, 1 update, DAPO off
# (launcher gates it to full topology). Scale with RETRO_PHASE2_GROUPS=32
# RETRO_PHASE2_ROLLOUTS=100 once the smoke is green.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export ROLLOUT_MODE=vanilla
export TRAIN_DATASET=terminal_bench_2_1
export WANDB_GROUP=qwen3.6-27b-tb21-vanilla-smoke
export RETRO_PHASE2_GROUPS=${RETRO_PHASE2_GROUPS:-4}
export RETRO_PHASE2_ROLLOUTS=${RETRO_PHASE2_ROLLOUTS:-1}
export ROLLOUT_PREFETCH_BATCHES=${ROLLOUT_PREFETCH_BATCHES:-1}
export FRESH_MAX_BEHAVIOR_LAG=${FRESH_MAX_BEHAVIOR_LAG:-1}
# TB tasks are shorter than Frontier-CS episodes; per-row budgets ride in
# metadata, these cap the agent loop itself.
export AGENTIC_MAX_STEPS=${AGENTIC_MAX_STEPS:-50}
export AGENTIC_EPISODE_TIMEOUT=${AGENTIC_EPISODE_TIMEOUT:-1800}

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (terminal_bench_2_1, vanilla smoke)"
uv run --with modal modal run -d agentic_rl/launch/modal_train.py::train
