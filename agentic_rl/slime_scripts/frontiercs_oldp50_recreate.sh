#!/usr/bin/env bash
# Arm 3/3 — old-P50 recreation (forensic: which stack change moved reward).
#
# Reproduces the pre-2026-08-18 semantics on the current tree:
#   RETRO_SEQUENTIAL_LEGS=1     fresh leg completes, THEN retro leg runs, with
#                               the RetroBuffer built after the fresh leg so
#                               age-0 snapshots are leasable (snapshot age 0-4;
#                               the concurrent schedule can only reach 1-4).
#   ROLLOUT_PREFETCH_BATCHES=1  the old mixed path hard-pinned the fresh pool
#                               to one batch (the nominal staleness=4 never
#                               applied).
#   FRESH_MAX_BEHAVIOR_LAG=0    gate DISABLED (old runs were unenforced;
#                               measured fresh lag reached 3).
#   RETRO_MAX_BEHAVIOR_LAG=1    keeps the crossed-version drop, which was the
#                               old retro lane's only real constraint.
#   AGENTIC_QUERY_TIMEOUT=600   old per-turn /generate cap.
#   RETRO_DETERMINISTIC=1       old cohort ran deterministic inference ON.
#   (no SGLANG_VERSION)         stays on the 0.5.12 base image like old P50.
#
# Old P50 ran 85 updates; this runs 100 with saves every 10 — compare at the
# shared 70-84 window AND at 100. If steps 70-84 reproduce ~0.3426, the round-2
# stack changes explain the old-P50 gap and per-axis toggles can attribute it;
# if it lands near p50_v2's 0.2939, something outside the known axes drifted.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export LAUNCH_STAMP=${LAUNCH_STAMP:-$(date +%Y%m%d-%H%M%S)}

export WANDB_GROUP=qwen3.6-27b-frontier-cs-retro-final-p50-legacy
export RETRO_PHASE2_GROUPS=32
export RETRO_PHASE2_ROLLOUTS=100
export RETRO_GROUP_RATIO=0.25
export RETRO_SEQUENTIAL_LEGS=1
export ROLLOUT_PREFETCH_BATCHES=1
export FRESH_MAX_BEHAVIOR_LAG=0
export RETRO_MAX_BEHAVIOR_LAG=1
export RETRO_PREFETCH_BATCHES=0
export AGENTIC_QUERY_TIMEOUT=600
export RETRO_DETERMINISTIC=1

echo "Launching ${WANDB_GROUP}-${LAUNCH_STAMP} (sequential legs, prefetch 1, ungated fresh, det on, sglang 0.5.12 base)"
uv run --with modal modal run -d agentic_rl/retro/modal_train.py::train
