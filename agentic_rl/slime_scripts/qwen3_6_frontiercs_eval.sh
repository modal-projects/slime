#!/usr/bin/env bash

set -euo pipefail
cd "${GUIDE:-$HOME/Documents/Research/async-rl/multinode-training-guide}"
export EXPERIMENT_CONFIG=w_qwen3_6_frontier_cs_eval
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev} WANDB_PROJECT=${WANDB_PROJECT:-Modal}

# uv run --no-dev modal run slime/modal_train.py::download_data
uv run --no-dev modal run -d slime/modal_train.py::train
