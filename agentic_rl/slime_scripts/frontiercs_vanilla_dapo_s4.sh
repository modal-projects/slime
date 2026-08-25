#!/usr/bin/env bash
# Arm 1/3 — VANILLA DAPO/GRPO baseline (the definitive no-retro control).
#
# Stock slime fully-async rollout + plain agentic generate: no retro lane, no
# snapshot capture, no manifest store. Staleness knobs mirror "a normal async
# RL run with max staleness 4": prefetch 4 (capacity) + hard behavior-lag
# gate 4 (enforced). Inference stack = SGLang 0.5.18, 4xTP2/node, EAGLE 3/1/4
# (rollout_sim 2026-08-23: +34% GPU tok/s and +25% KV pool vs the 0.5.12 base
# image; layout and EAGLE params are already the launcher defaults).
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
export SGLANG_VERSION=0.5.18
# Deterministic inference stays OFF (launcher default since the inference A/B:
# ~1.3x per-stream decode tax, training does not need reproducible sampling).

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (vanilla fully-async, prefetch 4, gate 4, sglang ${SGLANG_VERSION})"
uv run --with modal modal run -d agentic_rl/retro/modal_train.py::train
