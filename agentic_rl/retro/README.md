# Frontier-CS asynchronous retro replay

This package now contains both the retro runtime and its Modal launch surface:

- `modal_train.py` — Modal image, volumes, clustered Ray launch, data/model hooks,
  and HF→Megatron conversion.
- `launch_config.py` — Qwen3.6-27B topology, optimizer, checkpoint paths, reward
  arm, and retro ablation flags.
- `env.py`, `selector.py`, `snapshot.py`, `manifest.py`, `buffer.py`,
  `rollout.py` — capture and replay runtime.

The launcher no longer requires `multinode-training-guide`.

## Bootstrap

Run from the Slime repository root:

```bash
MODAL_ENVIRONMENT=junlin-dev \
  uv run --with modal modal run agentic_rl/retro/modal_train.py::download_model

MODAL_ENVIRONMENT=junlin-dev \
  uv run --with modal modal run agentic_rl/retro/modal_train.py::download_data

MODAL_ENVIRONMENT=junlin-dev \
  uv run --with modal modal run \
  agentic_rl/retro/modal_train.py::convert_hf_to_megatron_checkpoint
```

The hooks use the existing Modal volumes:

- `huggingface-cache`
- `slime-data`
- `slime-checkpoints`

## Training

The safe default is a two-node, four-group, one-update smoke:

```bash
MODAL_ENVIRONMENT=junlin-dev WANDB_PROJECT=Modal \
RETRO_REWARD_ARM=final \
RETRO_TARGET_TRAJECTORY_FRACTION=0.50 \
  uv run --with modal modal run agentic_rl/retro/modal_train.py::train
```

A full P75 Arm-A pilot:

```bash
MODAL_ENVIRONMENT=junlin-dev WANDB_PROJECT=Modal \
RETRO_REWARD_ARM=final \
RETRO_TARGET_TRAJECTORY_FRACTION=0.75 \
RETRO_MAX_FRACTION_ERROR=0.40 \
RETRO_SELECTOR_ASSIGNMENT=hashed \
RETRO_CAPTURE_PROMISING_RATIO=0.50 \
RETRO_POOL_PROMISING_RATIO=0.50 \
RETRO_POOL_ORDER=newest \
RETRO_MIN_POLICY_AGE=0 \
RETRO_MAX_POLICY_AGE=4 \
RETRO_PHASE2_GROUPS=32 \
RETRO_PHASE2_ROLLOUTS=20 \
  uv run --with modal modal run -d agentic_rl/retro/modal_train.py::train
```

Use `RETRO_REWARD_ARM=best` for Arm B. Resume with the original state tag:

```bash
RESUME=qwen3.6-27b-frontier-cs-retro-final-p75-<stamp> \
RETRO_REWARD_ARM=final \
RETRO_TARGET_TRAJECTORY_FRACTION=0.75 \
RETRO_PHASE2_GROUPS=32 \
RETRO_PHASE2_ROLLOUTS=85 \
  uv run --with modal modal run -d agentic_rl/retro/modal_train.py::train
```

Inspect the resolved config without provisioning GPUs:

```bash
RETRO_REWARD_ARM=final RETRO_TARGET_TRAJECTORY_FRACTION=0.75 \
  uv run --with modal modal run agentic_rl/retro/modal_train.py
```

## Selected position ablations

Hold every other control fixed and launch Arm A with:

```text
RETRO_TARGET_TRAJECTORY_FRACTION=0.25
RETRO_TARGET_TRAJECTORY_FRACTION=0.50
RETRO_TARGET_TRAJECTORY_FRACTION=0.75
```

All three use `RETRO_MAX_FRACTION_ERROR=0.40`, hashed 50/50 capture
assignment, a 4/4 promising/recovery replay pool, newest-first selection, and
policy age 0–4.
