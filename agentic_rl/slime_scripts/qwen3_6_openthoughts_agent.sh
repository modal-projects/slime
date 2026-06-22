export EXPERIMENT_CONFIG=w_qwen3_6_openthoughts_agent_2n
export MODAL_ENVIRONMENT=junlin-dev
export WANDB_PROJECT=Modal
export WANDB_GROUP=qwen3.6-35b-a3b-openthoughts-agent-colocate-2n


cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide



# ── One-time data: OpenThoughts-Agent-v1-RL onto slime-data volume ───────────
# Pull open-thoughts/OpenThoughts-Agent-v1-RL from HF + convert
# (convert2slime/openthoughts_agent.py → harbor schema) into
# /data/openthoughts_agent (openthoughts_agent.jsonl + tasks/openthoughts_agent__<id>/):
# uv run --no-dev modal run slime/modal_train.py::download_data
# Quick path — convert locally, then upload the out dir:
# (cd ../slime && uv run --with datasets python -m \
#   agentic_rl.environment.convert2slime.openthoughts_agent \
#   --out-dir /tmp/openthoughts_agent)
# uv run --no-dev modal volume put slime-data /tmp/openthoughts_agent /openthoughts_agent


# ── Train (colocated, sync on-policy, 2× H200:8) ──────────────────────────────
uv run --no-dev modal run -d slime/modal_train.py::train
