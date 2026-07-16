export EXPERIMENT_CONFIG=w_qwen3_6_swe_rebench_async


export WANDB_GROUP=qwen3.6-35b-a3b-swe-rebench-agentic-async

export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal


cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide




# ── Train (non-colocated, sync on-policy, 2× H200:8) ──────────────────────────────
uv run --no-dev modal run -d slime/modal_train.py::train
