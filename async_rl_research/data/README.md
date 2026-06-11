# Prompt data (artifacts only)

This directory holds generated dataset artifacts (`*.jsonl`, materialized
harbor task dirs). The converter code lives in
[`../env/convert2slime/`](../env/convert2slime/) — one converter per task
env, paired by filename (`env/swe_gym.py` ↔ `env/convert2slime/swe_gym.py`).

## SWE-Gym

```bash
python -m async_rl_research.env.convert2slime.swe_gym --lite --out async_rl_research/data/swe_gym_lite.jsonl
python -m async_rl_research.env.convert2slime.swe_gym --input raw_swe_gym.jsonl --out swe_gym.jsonl
```

Rows carry no `metadata.task_type` (SWE-Gym is the default env). Each row:
`prompt` = problem statement; `metadata` = `instance_id`, prebuilt `image`,
`workdir`, `eval_cmd`/`swepro`, `pre_commands`. See the converter docstring
for the field-by-field mapping.

## Harbor (USACO, ...)

```bash
# from a local harbor task tree (e.g. a harbor-datasets checkout subtree)
python -m async_rl_research.env.convert2slime.harbor \
    --tasks-dir ~/harbor-datasets/datasets/usaco --out-dir async_rl_research/data/usaco

# or straight from a harbor registry (needs `pip install harbor`)
python -m async_rl_research.env.convert2slime.harbor \
    --registry ~/harbor/registry.json --dataset usaco --out-dir async_rl_research/data/usaco
```

Rows carry `metadata.task_type: "harbor"` and a `task_path` relative to the
out dir; the rollout resolves it via `ASYNC_RL_TASK_ROOT=<out-dir>`. Put the
out dir on the slime-data volume so head workers can read the task files.
Before training, sanity-check the plumbing with the reference solutions (no
model involved):

```bash
export ASYNC_RL_TASK_ROOT=$(pwd)/async_rl_research/data/usaco
python -m async_rl_research.env.harbor $ASYNC_RL_TASK_ROOT/usaco.jsonl --limit 3   # expect reward=1.0
```
