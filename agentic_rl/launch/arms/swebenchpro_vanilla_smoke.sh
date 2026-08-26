#!/usr/bin/env bash
# SWE-Bench Pro — vanilla fully-async SMOKE (family onboarding proof).
#
# First training arm on the swebenchpro family (RUNBOOK §7 step 7): stock
# fully-async rollout over the derived 650/81 split (per-row random, seed
# 20260826; download_data materializes it — run that once first, then the
# oracle check; see agentic_rl/envs/swebenchpro/README.md). In-place grading,
# per-task environment/Dockerfile images (keepalive + no-provisioning quirks
# already handled in core/sandbox.py).
#
# 4-group engineering smoke: 1 train node + 1 rollout node, 1 update. Scale
# with RETRO_PHASE2_GROUPS=32 RETRO_PHASE2_ROLLOUTS=100 once green.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export ROLLOUT_MODE=vanilla
export TRAIN_DATASET=swebenchpro
export WANDB_GROUP=qwen3.6-27b-swebenchpro-vanilla-smoke
export RETRO_PHASE2_GROUPS=${RETRO_PHASE2_GROUPS:-4}
export RETRO_PHASE2_ROLLOUTS=${RETRO_PHASE2_ROLLOUTS:-1}
export ROLLOUT_PREFETCH_BATCHES=${ROLLOUT_PREFETCH_BATCHES:-1}
export FRESH_MAX_BEHAVIOR_LAG=${FRESH_MAX_BEHAVIOR_LAG:-1}
export AGENTIC_MAX_STEPS=${AGENTIC_MAX_STEPS:-75}
export AGENTIC_EPISODE_TIMEOUT=${AGENTIC_EPISODE_TIMEOUT:-1800}

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (swebenchpro, vanilla smoke)"
uv run --with modal modal run -d agentic_rl/launch/modal_train.py::train
