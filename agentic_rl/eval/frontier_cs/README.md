# Frontier-CS held-out avg@3 evaluation

This directory defines one common evaluation protocol for the ordinary-RL
baseline and Retro P25/P50/P75 checkpoints. It does not launch jobs by itself.

## Held-out split

The published `junlin-modal/frontier-cs` dataset was partitioned with
`eval_n=38, seed=0`:

- training: `/data/frontier_cs/train.jsonl` (150 tasks)
- held-out evaluation: `/data/frontier_cs/eval.jsonl` (38 tasks)

Training configs read only `train.jsonl`. The eval config reads the complete
`eval.jsonl`. `split.py` additionally fails if task IDs or judge problem IDs
overlap, if either split contains duplicates, or if the eval count is not 38.
The validated train/eval SHA-256 hashes are pinned in `arms.json`; evaluation
refuses to run if the downloaded split changes.

## Common protocol

Every checkpoint uses the same evaluation environment:

- 38 held-out tasks, three independent samples per task
- `avg@3`: mean reward within each task, then mean across 38 tasks
- temperature 1.0, top-p 1.0, top-k -1
- 24,576 generated tokens per turn and 65,536 total context tokens
- 75 agent turns, 1,800 seconds per episode
- 600-second verifier timeout and 60-second command timeout
- forced think closure disabled
- deterministic request seeding from rollout seed `20260802`
- two-node eval topology: one TP4×CP2 model-loading node and one node with four
  TP2 rollout engines (model-only checkpoint load reshards DP2 to DP1)

These are the current training rollout limits used by the P25/P50/P75 runs.
The common protocol intentionally overrides the older permissive eval recipe
(65,536 tokens per turn, 131,072 context, 3,600 seconds, temperature 0.6).

## Checkpoints

`arms.json` is the source of truth for the four checkpoint roots. Each path is:

```text
/checkpoints/swe_ckpts/<source_run_tag>
```

The comparison registry pins every arm to complete iteration 79 so the held-out
evaluation is checkpoint-matched.

P50/P75 contain incomplete `iter_0000084` directories without `.metadata` or
`metadata.json`; they must not be loaded. Baseline and P25 do have complete
iteration-84 checkpoints, but using those would confound the four-way comparison.

## Dry-run plan

From the `slime` repository:

```bash
python -m agentic_rl.eval.frontier_cs.plan
```

The command prints, but never executes:

1. one dataset download and split-validation command;
2. one Modal eval command per selected arm;
3. one post-processing command per arm.

Select a subset or emit machine-readable JSON:

```bash
python -m agentic_rl.eval.frontier_cs.plan baseline p50
python -m agentic_rl.eval.frontier_cs.plan --format json
```

The Modal config is:

```text
frontier_cs.w_qwen3_6_27b_frontier_cs_heldout_avg3
```

It saves `rollout_eval_0.pt` and writes strict `summary.json` beside it during
`modal_train.py::post_process_data`.

## Local aggregation

If a dump is available locally:

```bash
python -m agentic_rl.eval.frontier_cs.aggregate \
  /path/to/rollout_eval_0.pt \
  --output /path/to/summary.json
```

Aggregation fails by default unless all 38 tasks have exactly three distinct
sample indices. Besides `avg_at_k`, the summary includes task standard error,
pass@3, sample solve rate, all per-task rewards, and a dump hash.

The completed checkpoint-matched comparison is recorded in
`results/iteration-79.json`, including paired task-level bootstrap intervals
and W&B links.
