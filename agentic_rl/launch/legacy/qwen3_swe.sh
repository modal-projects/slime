export EXPERIMENT_CONFIG=w_qwen3_swe_async
export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal
export WANDB_GROUP=qwen3-30b-a3b-swe-gym-lite-async


cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide



# ── One-time data: SWE-Gym-Lite on slime-data volume ─────────────────────────
# Pull+convert from HuggingFace inside the container:
# uv run --no-dev modal run slime/modal_train.py::download_data

# ── Model + checkpoint ───────────────────────────────────────────────────────
# uv run --no-dev modal run slime/modal_train.py::download_model
# Reuse w_qwen3_dapo's Qwen3-30B-A3B_torch_dist (TP=4, PP=1, identical); run
# only if not yet converted:
# uv run --no-dev modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

# ── Train (async, non-colocated, 2× H200:8) ──────────────────────────────────
uv run --no-dev modal run -d slime/modal_train.py::train
