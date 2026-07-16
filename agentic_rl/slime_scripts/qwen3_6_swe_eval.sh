#!/usr/bin/env bash

set -euo pipefail
cd "${GUIDE:-$HOME/Documents/Research/async-rl/multinode-training-guide}"
export EXPERIMENT_CONFIG=w_qwen3_6_swe_eval_2n

# export WANDB_PROJECT="fully-async-rl-modal"
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export WANDB_GROUP="qwen3.6-35b-a3b-swe-eval-2n"
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev} 



# uv run --no-dev modal run slime/modal_train.py::download_data
uv run --no-dev modal run -d slime/modal_train.py::train
