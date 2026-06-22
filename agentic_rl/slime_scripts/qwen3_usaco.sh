export EXPERIMENT_CONFIG=w_qwen3_usaco_1n
export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal
export WANDB_GROUP=qwen3-30b-a3b-usaco


cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide



# ── One-time data: harbor USACO onto slime-data volume (/data/usaco) ─────────
# In-container: sparse-clone harbor-datasets + convert (convert2slime/harbor.py)
# → usaco.jsonl + tasks/usaco__<id>/:
# uv run --no-dev modal run slime/modal_train.py::download_data
# Quick path — convert locally, then upload the out dir:
# (cd ../slime && python3 -m agentic_rl.environment.convert2slime.harbor \
#   --tasks-dir <harbor-datasets>/datasets/usaco --out-dir /tmp/usaco --name usaco)
# uv run --no-dev modal volume put slime-data /tmp/usaco /usaco

# ── Oracle sanity check (no GPUs): reference solutions; expect reward=1.0 ────
# (cd ../slime && uv run --with modal python -m agentic_rl.environment.harbor \
#   /tmp/usaco/usaco.jsonl --task-root /tmp/usaco --limit 3)

# ── Model + checkpoint ───────────────────────────────────────────────────────
# Reuses the same Qwen3-30B-A3B_torch_dist as w_qwen3_swe_async / w_qwen3_dapo
# (TP=4, PP=1); run only if not yet converted:
# uv run --no-dev modal run slime/modal_train.py::download_model
# uv run --no-dev modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

# ── Train (async, non-colocated, 1× H200:8) ──────────────────────────────────
uv run --no-dev modal run -d slime/modal_train.py::train
