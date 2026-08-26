#!/usr/bin/env bash

set -euo pipefail
cd "${GUIDE:-$HOME/Documents/Research/async-rl/multinode-training-guide}"
export EXPERIMENT_CONFIG=glm47_flash_swe_eval

# export WANDB_PROJECT="fully-async-rl-modal"
export WANDB_PROJECT=${WANDB_PROJECT:-Modal}
export WANDB_GROUP="glm4.7-flash-swe-eval"
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev} 



# uv run --no-dev modal run slime/modal_train.py::download_model
# uv run --no-dev modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint
# uv run --no-dev modal run slime/modal_train.py::download_data
uv run --no-dev modal run -d slime/modal_train.py::train
