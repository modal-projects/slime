# export EXPERIMENT_CONFIG=w_qwen3_swe_colocate_2n
export EXPERIMENT_CONFIG=w_qwen3_swe_colocate_1n
# export EXPERIMENT_CONFIG=w_qwen3_swe_sync
export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal


cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide




uv run --no-dev modal run -d slime/modal_train.py::train
