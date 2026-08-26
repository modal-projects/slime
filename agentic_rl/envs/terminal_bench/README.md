# Terminal-Bench 2.1 (`terminal_bench_2_1`)

Harbor IS the terminal-bench task format, so this family has **zero env
code**: rows carry `task_type: "harbor"` and run through `envs/harbor/env.py`
(multi-step episodes, in-place `test.sh` → `reward.json` grading, prebuilt
`docker_image` per task).

## Dataset

Published: [`junlin-modal/terminal-bench-2.1`](https://huggingface.co/datasets/junlin-modal/terminal-bench-2.1)
— `terminal_bench_2_1/eval.jsonl` (**89 tasks**) + one task dir each, converted
with the canonical harbor converter (`envs/harbor/convert.py`).

**v1 scope gates** (each excluded task raises `SkipTask` at conversion —
rerun the converter over the raw TB 2.1 checkout to re-audit):

| gate | why |
|---|---|
| `os != linux` | Modal sandboxes are linux |
| GPU / TPU requirements | rollout sandboxes are CPU |
| `mcp_servers` | no MCP plumbing in the episode loop |
| `network_mode` no-network / allowlist | Modal can't enforce per-phase network policy — running them would over-grant |
| docker-compose environments | single-container sandboxes only |
| no `docker_image` and no `environment/Dockerfile` | nothing to boot |

## Roles

- **Transfer eval** (primary): the full 89-task `eval.jsonl`, already wired as
  a guide-registry eval dataset since the SWE eval waves.
- **Training** (deliberate choice — the set is small): `TRAIN_DATASET=terminal_bench_2_1`
  derives a deterministic 69/20 split on the volume at `download_data` time
  (seed 20260826, per-row; `envs/datasets.py`). The published repo is never
  mutated; the split is recorded in `split-<seed>.json`. Verified 2026-08-26
  against the published eval.jsonl: `train_sha256=43c519499713…`,
  `eval_sha256=2e08985ed60a…` — a volume materialization must reproduce these.

## Onboarding checklist (Modal)

```bash
TRAIN_DATASET=terminal_bench_2_1 MODAL_ENVIRONMENT=junlin-dev \
  uv run --with modal modal run agentic_rl/launch/modal_train.py::download_data
# oracle-check a few tasks through the exact rollout path (needs sandboxes):
python -m agentic_rl.envs.harbor.env /data/terminal_bench_2_1/eval.jsonl --task-root /data --limit 3
# then the smoke arm:
bash agentic_rl/launch/arms/terminalbench21_vanilla_smoke.sh
```

Retro capture is **not** available for this family yet (needs a ScoreTrace
adapter over the per-step `reward.json` trace + the P1 capture split —
RUNBOOK §7.1); `ROLLOUT_MODE=retro` with this dataset fails at launch.
