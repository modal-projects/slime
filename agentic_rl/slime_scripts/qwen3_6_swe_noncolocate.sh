# export EXPERIMENT_CONFIG=w_qwen3_6_swe_rebench_v2_noncolocate_5n
# export WANDB_GROUP=qwen3.6-35b-a3b-swe-rebench-v2-noncolocate-5n

# export EXPERIMENT_CONFIG=w_qwen3_6_swe_rebench_v2_noncolocate_3n
# export WANDB_GROUP=w_qwen3_6_swe_rebench_v2_noncolocate_3n

export EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n
export WANDB_GROUP=qwen3.6-27b-swe-rebench-v2-noncolocate-5n

export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal


cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide



# uv run --no-dev modal run slime/modal_train.py::download_model
# uv run --no-dev modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint
# uv run --no-dev modal run slime/modal_train.py::download_data
uv run --no-dev modal run -d slime/modal_train.py::train
