# export EXPERIMENT_CONFIG=w_qwen3_6_swe_colocate_1n
# export EXPERIMENT_CONFIG=w_qwen3_6_usaco_colocate_1n
# export EXPERIMENT_CONFIG=w_qwen3_6_swe_colocate_2n
# export EXPERIMENT_CONFIG=w_qwen3_6_dapo_colocate_2n
export EXPERIMENT_CONFIG=w_qwen3_6_dapo_colocate_1n_tggym

export WANDB_GROUP=qwen3.6-35b-a3b-dapo-math-colocate-1n-tggym-nodpattention

export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal


cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide



# ── One-time data: SWE-Gym-Lite on slime-data volume (skip if qwen3 runs did) ─
# uv run --no-dev modal run slime/modal_train.py::download_data

# ── Model + checkpoint ───────────────────────────────────────────────────────
# uv run --no-dev modal run slime/modal_train.py::download_model
# Qwen3.6-35B-A3B needs its OWN torch_dist conversion (qwen3.5 arch, TP=2); the
# qwen3 Qwen3-30B-A3B_torch_dist can't be reused:
# uv run --no-dev modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

# ── Train (colocated, sync on-policy, 2× H200:8) ──────────────────────────────
uv run --no-dev modal run -d slime/modal_train.py::train
