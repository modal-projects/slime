export EXPERIMENT_CONFIG=qwen3_dapo
export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal
export WANDB_GROUP=qwen3-30b-a3b-dapo-math-1n



cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide



# uv run --no-dev modal run slime/modal_train.py::download_model
# uv run --no-dev modal run slime/modal_train.py::download_data
# uv run --no-dev modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint
uv run --no-dev modal run -d slime/modal_train.py::train
