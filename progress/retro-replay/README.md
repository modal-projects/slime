# Frontier-CS retro replay progress

This directory keeps the reviewable artifacts for the Frontier-CS retro-replay
experiments together:

- `frontier-cs-retro-replay.html` — self-contained experiment report.
- `figures/outcome-reward-ablation-final-submissions.{png,svg}` — O0/O1/O2.
- `figures/five-arm-final-submissions.{png,svg}` — O0/O1/O2 plus Retro A/B.
- `figures/position-ablation-final-submissions.{png,svg}` — P25/P50/P75
  with the original Retro A arm as a reference.
- `generate_figures.py` — rebuilds all figures from W&B retry lineages.

## Refresh the figures

From the repository root:

```bash
uv run --with wandb --with matplotlib \
  python progress/retro-replay/generate_figures.py
```

The script reads `junlinwang/Modal`, merges every matching W&B retry in
chronological order, and lets later retries replace overlapping optimizer
steps. Curves use a 7-step trailing mean; five-arm and position-ablation plots
are sampled every five optimizer steps. W&B credentials must already be
available locally.

## Comparison caveats

- O1 versus O2 is the clean historical reward-shaping comparison.
- Retro A versus Retro B is matched, but resumed steps 40–84 accepted no retro
  groups and fell back to fresh-only generation.
- P25/P50/P75 differ in target branch position; their latest available W&B
  steps may differ while runs are incomplete or awaiting restart.
- Cross-cohort comparisons are descriptive because prompt order, DAPO, async
  settings, and deterministic inference differ.
