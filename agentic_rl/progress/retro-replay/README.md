# Frontier-CS retro replay progress

This directory keeps the reviewable artifacts for the Frontier-CS retro-replay
experiments together:

- `frontier-cs-retro-replay.html` — self-contained experiment report, four
  tabs: research report, five-arm curves, P25/P50/P75, and the behavior-lag
  ("staleness") ablation. The staleness tab's four charts are inline SVG built
  from series embedded in the page (`var stal = {...}` in the page script), not
  from `figures/`; refresh them with the query in **Refresh the staleness
  series** below. Series hues are validated for colour-vision separation in
  both themes — keep `--s-rlag4` / `--s-rlag8` / `--s-oldp50` if you restyle.
- `figures/outcome-reward-ablation-final-submissions.{png,svg}` — O0/O1/O2.
- `figures/five-arm-final-submissions.{png,svg}` — O0/O1/O2 plus Retro A/B.
- `figures/position-ablation-final-submissions.{png,svg}` — P25/P50/P75
  with the original Retro A arm as a reference.
- `generate_figures.py` — rebuilds all figures from W&B retry lineages.

## Refresh the figures

From the repository root:

```bash
uv run --with wandb --with matplotlib \
  python agentic_rl/progress/retro-replay/generate_figures.py
```

The script reads `junlinwang/Modal`, merges every matching W&B retry in
chronological order, and lets later retries replace overlapping optimizer
steps. Curves use a 7-step trailing mean; five-arm and position-ablation plots
are sampled every five optimizer steps. W&B credentials must already be
available locally.

## Refresh the staleness series

The staleness tab's charts read from arrays embedded in the HTML. Rebuild them
from W&B (merging every fault-tolerance retry in each run *group*) with:

```bash
uv run --with wandb python agentic_rl/progress/retro-replay/refresh_staleness_series.py
```

That prints the merged per-step JSON; the tab samples it every five updates
onto a 7-step trailing mean. Note the counters `behavior_lag/rejected_groups`
and `dynamic_sampling/completed_groups` are unusable for the rlag4/rlag8 runs
(a since-fixed requeue bug recycled rejected groups), so the charts avoid them.

## Comparison caveats

- O1 versus O2 is the clean historical reward-shaping comparison.
- Retro A versus Retro B is matched, but resumed steps 40–84 accepted no retro
  groups and fell back to fresh-only generation.
- P25/P50/P75 are complete through step 84. Their canonical W&B runs merge
  restart segments into continuous 85-update lineages; the realized mean branch
  fractions are 0.364, 0.550, and 0.716.
- rlag4 versus rlag8 is a clean comparison but a null one: realized retro
  behavior lag never exceeded 1, so retro bounds of 4 and 8 never bound. Treat
  the pair as seed replicates and their 0.0016 endpoint gap as this harness's
  noise floor.
- Neither new arm is comparable to old P50 on reward: six things changed at
  once (leg scheduling, prefetch 1→4, both lag gates, snapshot age 0–4→1–4,
  query timeout 600→1200 s), and prefetch 4 truncated 13–23% of episodes on
  the 1800 s wall versus 0.7% before.
- Cross-cohort comparisons are descriptive because prompt order, DAPO, async
  settings, and deterministic inference differ.
