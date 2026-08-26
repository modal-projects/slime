#!/usr/bin/env bash
# Baseline addition — REPLICATE of the vanilla DAPO/GRPO baseline (seed pair).
#
# Byte-identical config to frontiercs_vanilla_dapo_s4.sh under a new W&B group
# and launch stamp. Purpose: an endpoint noise floor for THE go-forward
# baseline config — the only existing replicate pair (rlag4/rlag8) is a
# round-1 retro config, and its floor turned out metric-dependent (~0.011
# pre-filter fresh). Deterministic inference is off, so sampling diverges
# between the pair by construction; prompt order is identical (fixed seed,
# no shuffle).
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export ROLLOUT_MODE=vanilla
export WANDB_GROUP=qwen3.6-27b-frontier-cs-vanilla-dapo-s4-r2
export RETRO_PHASE2_GROUPS=32
export RETRO_PHASE2_ROLLOUTS=100
export ROLLOUT_PREFETCH_BATCHES=4
export FRESH_MAX_BEHAVIOR_LAG=4

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (vanilla DAPO replicate, prefetch 4, gate 4)"
uv run --with modal modal run -d agentic_rl/retro/modal_train.py::train
