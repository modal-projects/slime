# The slime fork delta

The `agentic_rl/` overlay is *mostly* hook-injected (`--custom-generate-function-path`,
`--rollout-function-path`, `--custom-rollout-log-function-path`), but this branch does
carry real edits to `slime/` framework code. **This file is the authoritative inventory.
Any PR that touches `slime/` must update this table** — the old README claim of
"no edits to slime/" is exactly the kind of drift this file exists to prevent.

Measured against the merge-base with `main` (`git diff $(git merge-base HEAD main) --stat -- slime/`): 8 files, ~+233 lines.

| File | Δ | What |
|---|---|---|
| ~~`slime/rollout/fully_async_rollout.py`~~ | 0 | **De-forked 2026-08-26** → `agentic_rl/core/fully_async.py`; the slime file is back at upstream |
| ~~`slime/utils/arguments.py`~~ | 0 | **De-forked 2026-08-26**: the two knobs are `ASYNC_RL_ROLLOUT_PREFETCH_BATCHES` / `ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG` env vars, normalized onto `args` by `core.fully_async.resolve_async_rl_rollout_knobs` |
| `slime/utils/wandb_utils.py` | +107 | sglang engine-metrics scraper: polls the router's `/engine_metrics` → `sgl_engine/*` gauges (means across engines) |
| `slime/backends/megatron_utils/cp_utils.py` + `loss.py` | +83 | MAX-reduced metric channel → `train_rollout_logprob_abs_diff_max` (catches per-token logprob divergence that mean-reduction hides) |
| `slime/backends/megatron_utils/actor.py` | +5 | **absolute `weight_version` across resumes** — the behavior-lag gate and `async/*` metrics are wrong without it. Cannot be de-forked. |
| `slime/backends/sglang_utils/arguments.py` | +17 | sglang 0.5.15 ServerArgs rename aliasing (the `nightly-dev-20260810a` train-image pin needs it) |
| `slime/ray/rollout.py` | +17 | `metadata` passthrough (turn spans) + scraper wiring |
| `slime/backends/megatron_utils/data.py` | +3 | metric-loop skips for agentic metadata keys |
| `slime/utils/logging_utils.py` | +8 | scraper hook |

## Rules

1. **Prefer the overlay.** New behavior goes in `agentic_rl/` via the string-loaded
   hooks; a `slime/` edit needs a reason the hooks can't express (so far: weight-version
   accounting, metric plumbing below the hook surface, and image-pin compatibility).
2. **Update this table in the same commit** as any `slime/` change, and keep
   `RUNBOOK.md` §3.8 pointing here.
3. The de-forked async rollout file (`agentic_rl/core/fully_async.py`) is a copy, not a
   fork: if upstream's `slime/rollout/fully_async_rollout.py` moves in a way
   we care about, diff against it deliberately — nothing tracks it automatically.
