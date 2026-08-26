# SWE-Bench Pro (`swebenchpro`)

Follows the swe_rebench pattern (`envs/swe_rebench/convert.py` was the
template): HF rows → harbor task dirs using the prebuilt per-task images, rows
carry `task_type: "harbor"` and run through `envs/harbor/env.py` — **zero env
code** in this family.

## Dataset

Published: [`junlin-modal/swebenchpro`](https://huggingface.co/datasets/junlin-modal/swebenchpro)
— `swebenchpro/eval.jsonl` (**731 tasks**, 11 distinct repos: ansible,
internetarchive, flipt, qutebrowser, teleport, protonmail, element-web, vuls,
nodebb, …) + per-task dirs with `environment/Dockerfile` and a self-contained
verifier.

## Decisions (RUNBOOK §7 asked for these explicitly)

- **Grading: in-place** (the harbor default; the sandbox state is the
  deliverable). This matches how the eval waves already ran these tasks. The
  fresh-sandbox anti-reward-hack option the native swerebench env used remains
  possible but is NOT wired — revisit if reward hacking shows up in dumps.
- **Train/heldout split: per-row random, 650/81, seed 20260826** (derived on
  the volume at `download_data`; published repo untouched; recorded in
  `split-<seed>.json`). Verified 2026-08-26 against the published eval.jsonl:
  `train_sha256=4787d5a7805f…`, `eval_sha256=2403927527e3…` — a volume
  materialization must reproduce these. Same-repo issues can land on both sides — the standard
  SWE-bench framing. The group-disjoint alternative (hold out whole repos via
  `SplitSpec(group_regex=r"^instance_.+?(?=-[0-9a-f]{40})")`) is supported by
  `envs/datasets.py` but high-variance with only 11 repo groups.

## Known image quirks (already handled below this family)

- Empty `ENTRYPOINT` + no keepalive → container exits rc128 on boot: the
  shared sandbox boots with `sleep infinity` (core/sandbox.py).
- Broken/poisoned pip indexes + missing curl in some images: the grader is
  self-contained (vendored stdlib test parsing) and **nothing provisions at
  episode time**.

## Onboarding checklist (Modal)

```bash
TRAIN_DATASET=swebenchpro MODAL_ENVIRONMENT=junlin-dev \
  uv run --with modal modal run agentic_rl/launch/modal_train.py::download_data
python -m agentic_rl.envs.harbor.env /data/swebenchpro/eval.jsonl --task-root /data --limit 3
bash agentic_rl/launch/arms/swebenchpro_vanilla_smoke.sh
```

Retro capture is **not** available for this family yet (ScoreTrace adapter +
P1 split pending — RUNBOOK §7.1); `ROLLOUT_MODE=retro` fails at launch.
