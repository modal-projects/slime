# Paused runs (stopped 2026-08-26, resumable)

Two round-3 arms were stopped mid-run on 2026-08-26 (~13:30 EDT) to free GPUs
(user request). Both have committed checkpoints on `slime-checkpoints` and
resume cleanly via Megatron's `latest_checkpointed_iteration.txt`.

| arm | stopped app | W&B step at stop | last saved ckpt | steps lost on resume |
|---|---|---|---|---|
| vanilla-dapo-s4 | ap-09y0pdZRuOrkdZ1KkGZ9Ai | 66/100 | iter_0000059 | 60–66 |
| vanilla-grpo-nodapo | ap-LEG4QSPUZtfSgnLSEZRTAF | 52/100 | iter_0000049 | 50–52 |

Checkpoint dirs (volume `slime-checkpoints`, env `junlin-dev`):
- `swe_ckpts/qwen3.6-27b-frontier-cs-vanilla-dapo-s4-20260825-001823`
- `swe_ckpts/qwen3.6-27b-frontier-cs-vanilla-grpo-nodapo-20260825-173008`

## How to resume

Re-run the same arm script with `LAUNCH_STAMP` pinned to the original stamp.
That makes `run_tag == state_tag`, so the run reuses the same checkpoint dir
(auto-loads the latest iter) AND lands in the same W&B group (merges with the
existing restart segments):

```bash
LAUNCH_STAMP=20260825-001823 bash agentic_rl/launch/arms/frontiercs_vanilla_dapo_s4.sh
```

```bash
LAUNCH_STAMP=20260825-173008 bash agentic_rl/launch/arms/frontiercs_vanilla_grpo_nodapo.sh
```

(`RESUME=<state_tag>` also works but with a fresh stamp it starts a NEW W&B
group while loading the old checkpoints — use the LAUNCH_STAMP form unless you
deliberately want a forked lineage. `RESUME_CKPT_STEP=<iter>` picks an earlier
checkpoint.)

## Why paused / what was pending at stop

- Both are the DAPO-value comparison pair; matched-step pre-filter read through
  step 47 showed grpo-nodapo ≈ vanilla-dapo (deltas −0.025/+0.031/−0.015,
  pooled ≈ 0) with ~30% cheaper steps — GRPO-primary gate passing so far.
- The other two arms (retro0-s4, retro-final-p50-legacy) were left running to
  their 100-step endpoints.
- On completion of the resumed runs: endpoint step-100 comparison + held-out
  avg@3 on iter_0000099 (standing adjudication protocol).

Delete this file once both runs are resumed and finished.
