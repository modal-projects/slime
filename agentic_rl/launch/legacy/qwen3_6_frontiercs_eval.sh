#!/usr/bin/env bash
# LEGACY (guide-repo stack): exploratory single-checkpoint eval via the old
# EXPERIMENT_CONFIG classes. The strict held-out avg@3 protocol now launches
# from THIS repo — see `python -m agentic_rl.eval.frontier_cs.plan`
# (ROLLOUT_MODE=eval -> agentic_rl/launch/modal_train.py). RUNBOOK §7 step 4.

set -euo pipefail
cd "${GUIDE:-$HOME/Documents/Research/async-rl/multinode-training-guide}"
export EXPERIMENT_CONFIG=frontier_cs.w_qwen3_6_frontier_cs_eval
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev} WANDB_PROJECT=${WANDB_PROJECT:-Modal}

# uv run --no-dev modal run slime/modal_train.py::download_data
uv run --no-dev modal run -d slime/modal_train.py::train
