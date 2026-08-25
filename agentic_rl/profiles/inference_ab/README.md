# inference_ab — SGLang deterministic-inference A/B rollout profile

Isolates the rollout **inference** performance of the Frontier-CS retro runs
(Retro A/B, P25/P50/P75) from sandboxes, judges, and training, and answers one
question: **how much decode speed does `--enable-deterministic-inference`
cost?** The retro arms run it on and decode ~75 tok/s/stream; the pre-retro
baselines ran it off and decoded ~135–164 tok/s/stream.

## What it does

One Modal container = one production engine slice: 2× H200, TP2, booted with
the exact ServerArgs the retro `launch_config.py` produces (EAGLE 3/1/4,
`mem_fraction_static=0.85`, prod cuda-graph-bs list, `mamba_scheduler_strategy=
extra_buffer`, `random_seed=20260802`, plus the ray-inherited
`CUDA_DEVICE_MAX_CONNECTIONS=1` / `NCCL_NVLS_ENABLE=1`). The only variable is
the deterministic flag.

Against it, `workload.py` simulates 5 rollout steps shaped like the measured
P50 run:

- real Frontier-CS prompts from `slime-data:/data/frontier_cs/train.jsonl`;
- 16 episodes/step as 2 groups × 8 siblings sharing a prompt
  (`n_samples_per_prompt=8` → same radix-cache sharing as production);
- in-flight cap 8 (production averages ~7 in-flight episodes per engine);
- 12 turns/episode; each turn is a `/generate` POST with full `input_ids` and
  `return_logprob=True` (byte-identical request shape to
  `agentic_rl.model._generate`), then a canned tool-observation block and a
  ~4 s jittered "sandbox exec" pause;
- deterministic arm passes per-sibling `sampling_seed` exactly like
  `sglang_rollout.GenerateState`.

Reported per step and overall: per-stream decode tok/s (production definition:
episode output tokens / summed request latencies), engine tok/s, EAGLE
`spec_accept_length`/`spec_accept_rate` (from `meta_info` — the training runs
never recorded these), prefix-cache hit rate, episode/step wall, plus the
resolved `server_info` (attention/sampling backend, radix on/off) so the A/B
documents *what* the deterministic flag actually switched.

## Run

```bash
export MODAL_ENVIRONMENT=junlin-dev   # required — see modal-run-env-mismatch
uv run --with modal modal run -d agentic_rl/profiles/inference_ab/modal_profile.py --arm on
uv run --with modal modal run -d agentic_rl/profiles/inference_ab/modal_profile.py --arm off
```

Knobs: `--steps 5 --episodes 16 --concurrency 8 --turns 12 --max-new 4096
--pause 4.0`. Full JSON (per-turn records included) lands in
`slime-data:/data/profiles/inference_ab/<stamp>-det-{on,off}.json`; a summary
is printed to the app logs.

```bash
modal volume get slime-data profiles/inference_ab/ ./results/ --env junlin-dev
```

## Fidelity caveats

- 12 turns × ~1.5k tok reaches ~30k context; production episodes average 31
  turns / 53k. Decode speed falls with context, so absolute tok/s reads a bit
  high; the A/B *ratio* is the result.
- The policy is the base checkpoint, not a trained one — EAGLE accept length
  on trained-policy text may differ.
- One engine, no router: cross-engine interference (weight resync pauses,
  router imbalance) is out of scope here.
