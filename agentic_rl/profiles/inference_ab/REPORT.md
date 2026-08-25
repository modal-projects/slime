# Deterministic-inference A/B — Qwen3.6-27B Frontier-CS engine slice

**Date:** 2026-08-18 · **Status:** final — both arms completed 5/5 simulated
rollout steps on 2×H200 (TP2), engine flags identical to the retro training
runs, sglang 0.5.12.post1 (the production image).

| Arm | App | Result JSON (slime-data volume) |
|---|---|---|
| det-**on** (mirrors Retro A/B, P25/P50/P75) | boot 176 s | `profiles/inference_ab/20260818-091422-det-on.json` |
| det-**off** (mirrors pre-retro baselines) | boot 316 s | `profiles/inference_ab/20260818-091458-det-off.json` |

Workload per arm: 5 steps × 16 episodes (2 prompt groups × 8 siblings, real
Frontier-CS prompts), in-flight cap 8, 12 turns/episode, `/generate` with full
`input_ids` + `return_logprob=True`, ~4 s jittered tool pauses. Identical
prompts, seeds, and schedule across arms.

## TL;DR

**`--enable-deterministic-inference` costs ~1.3× per-stream decode speed on
this engine** (steady-state mean 95 → 127 tok/s when disabled; marginal
decode rate 130 → 155 tok/s; per-request overhead 0.31 → 0.18 s). EAGLE
speculative decode is unaffected (accept length ~2.5 in both arms). The tax
is **not** the attention backend — both arms resolve to `fa3` — it is the
pytorch sampling backend, `disable_custom_all_reduce` + `NCCL_ALGO=tree`, and
batch-invariant kernels. Deterministic mode therefore explains **part, but
not all**, of the production 2× decode gap (149 → 75 tok/s baseline → retro);
see "What this does and doesn't explain".

## Steady-state results (steps 1–4; step 0 carries warmup in both arms)

| Metric | det-on | det-off | off/on |
|---|---|---|---|
| decode tok/s per stream, mean | 95.2 | 126.6 | **1.33×** |
| decode tok/s per stream, p50 | 98.4 | 126.0 | 1.28× |
| decode tok/s per stream, p10 (worst episodes) | 74.4 | 109.9 | **1.48×** |
| marginal decode rate (latency-vs-tokens fit) | 130 | 155 | 1.19× |
| per-request fixed overhead (fit intercept) | 0.31 s | 0.18 s | 0.58× |
| EAGLE accept length | 2.53 | 2.51 | ≈1 |
| prefix-cache hit rate | 0.86 | 0.85 | ≈1 |
| episode elapsed p50 (incl. 4 s pauses) | 107.6 s | 93.0 s | 0.86× |
| engine agg tok/s | 189 | 193 | ≈1 (not saturated at cap 8) |

Per-turn latency decomposes as `overhead + ms/token`: deterministic mode
nearly doubles the fixed per-request cost (0.31 s vs 0.18 s — pytorch
sampling + non-custom all-reduce path) and adds ~19% to the per-token cost.
Short turns feel the overhead; long turns feel the marginal rate.

## What the flag actually switched (`/get_server_info`)

| Field | det-on | det-off |
|---|---|---|
| attention_backend | fa3 | **fa3 (same!)** |
| sampling_backend | **pytorch** | flashinfer |
| disable_custom_all_reduce | **true** (+ NCCL_ALGO=tree) | false |
| disable_radix_cache | false | false |
| speculative_algorithm | EAGLE (active, accept ~2.5) | EAGLE |
| max_total_num_tokens | 1,347,694 | 1,347,531 |

Two production beliefs died here: (1) deterministic mode does *not* lose the
radix cache or EAGLE on this model, and (2) the attention backend is the same
either way — `fa3` is the default resolution for this hybrid-GDN model on
H200 regardless.

## What this does and doesn't explain

Production shows 135–164 tok/s (baselines, det off) vs 70–76 tok/s (retro
arms, det on) — a ~2× gap. This A/B reproduces a **1.28–1.48×** gap under
matched conditions, so deterministic inference is the largest single
contributor but likely not the whole story. Remaining candidates, untested
here:

- **Context length**: production episodes average 53k total tokens vs this
  workload's ~10–30k. Batch-invariant attention (fixed KV splits) plausibly
  scales worse with context; rerun with `--turns 30` to test.
- **Multi-engine contention**: production runs 4 TP2 engines per node
  sharing NVLink; NCCL tree all-reduce under contention may cost more than in
  this single-engine container.
- **Trained- vs base-policy text** shifts turn lengths and EAGLE acceptance.

## Implications for the rollout-speed plan

- Disabling deterministic inference is worth ~1.3× on the decode leg of the
  agent phase (generate is 539 s of the 726 s P50 episode → roughly −120 s
  per episode, more if the context-length effect compounds). Whether to give
  up deterministic sibling sampling is a research-design call, not an
  engineering one.
- If determinism stays, the same seed-comparability can be kept cheaper:
  deterministic *sampling* (per-sibling `sampling_seed` with the pytorch
  sampler) does not require batch-invariant kernels or the all-reduce
  downgrade — worth checking whether seeded sampling alone gives
  reproducibility sufficient for the retro-replay design.
- The concurrency/pipelining fixes (parallel fresh+retro legs, staleness 2)
  are independent of this and remain the top item — the engines are ~95%
  idle regardless of arm.

## Files / repro

- Harness: `modal_profile.py` (engine boot, flag-identical ServerArgs),
  `workload.py` (episode simulator + metrics).
- Full per-turn records in the two JSONs on the `slime-data` volume
  (`modal volume get slime-data profiles/inference_ab/ ./ --env junlin-dev`).
- Rerun: `README.md` § Run. Knobs: `--steps/--episodes/--concurrency/--turns/
  --max-new/--pause`.
