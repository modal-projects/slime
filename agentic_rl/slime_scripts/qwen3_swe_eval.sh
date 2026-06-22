#!/usr/bin/env bash
# Eval Qwen3-30B-A3B on the published agentic-RL eval set
# (https://huggingface.co/datasets/junlin-modal/agentic-rl-evalsets).
# Results land in W&B as eval/<dataset>. Select with
# EVAL_DATASETS=swebench_verified_100,usaco_50 (default: all; see
# configs/w_qwen3_swe_eval.py). To (re)build the eval set, see prepare_eval_data.sh.
set -euo pipefail
cd "${GUIDE:-$HOME/Documents/Research/async-rl/multinode-training-guide}"
export EXPERIMENT_CONFIG=w_qwen3_swe_eval
export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev} WANDB_PROJECT=${WANDB_PROJECT:-Modal}

# uv run --no-dev modal run slime/modal_train.py::download_data
uv run --no-dev modal run -d slime/modal_train.py::train
