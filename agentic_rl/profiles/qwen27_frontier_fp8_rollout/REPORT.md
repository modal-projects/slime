# FP8 rollout vs BF16 — Qwen3.6-27B frontier-cs (noncolocate, 6 nodes)

**Date:** 2026-07-13 · **Status:** final — the FP8 app (`ap-m3dtGAvBCPbrb6qMDdx244`)
was stopped ~16:44 PDT after 3 completed steps (0–2, mid-rollout of step 3/4);
all comparisons below use those steps vs the BF16 lineage.

| Run | W&B | Config |
|---|---|---|
| FP8 rollout | [`7n10kxhw`](https://wandb.ai/junlinwang/Modal/runs/7n10kxhw) | `w_qwen3_6_27b_frontier_cs_noncolocate_5n_fp8` (serves official `Qwen/Qwen3.6-27B-FP8`, blockwise e4m3 128×128; BF16 Megatron training unchanged) |
| BF16 baseline (current) | [`o22dad4t`](https://wandb.ai/junlinwang/Modal/runs/o22dad4t) | `w_qwen3_6_27b_frontier_cs_noncolocate_5n`, steps 60–76 |
| BF16 baseline (first launch, for step-matched compare) | [`zk7x91il`](https://wandb.ai/junlinwang/Modal/runs/zk7x91il) | same, steps 10–39 |

Topology identical in all runs: 2 train nodes (TP4×CP2) + 4 rollout nodes
(16× TP2 SGLang engines, H200), EAGLE spec decode, batch 32×8=256 episodes.

## TL;DR

**FP8 rollout works end-to-end (no GDN stall, healthy training), but on this
workload it is only ~+7% on pure decode and ~0% on batch-level rollout
throughput.** The concrete wins are memory: weights 25.6 → 14.5 GB/GPU (−44%)
and KV pool 1.34M → 1.54M tokens/engine (+15%). It is not the ~2× decode
speedup hoped for in the 35B-A3B config notes.

## The comparison must be response-length-matched

`perf/tokens_per_gpu_per_sec` = generated tokens ÷ rollout wall time ÷ 32 GPUs,
so it scales with how much the policy writes. The fresh FP8 policy writes
~31k-token responses; the trained BF16 policy at steps 60–76 writes ~47–50k.
Comparing FP8 step 0–2 against BF16 steps 60–76 (174 vs 254 raw) is meaningless.
The right anchor is the BF16 first launch at the same response length:

| Metric (matched at ~31k mean response) | BF16 (zk7x91il, steps 11–25 @ 28–32k) | FP8 (7n10kxhw, steps 0–2 @ 30.5–31k) | Δ |
|---|---|---|---|
| Raw tok/s per rollout GPU (batch level) | ~159 (144–172) | ~162 (153–174) | **≈ 0 (+2%, within noise)** |
| Episode decode tok/s (LLM-time only, `agentic/decode_tok_per_s/mean`) | ~150 (145–156) | ~161 (155–167) | **≈ +7%** |
| Rollout wall time (s/batch of 256) | 1223–2001 | 1734–1984 | ≈ same |

Reference (not comparable directly): BF16 trained policy at 47–50k responses
runs 221–289 raw / 180–233 effective tok/s/GPU (mean 254/209 over steps 60–72).

## Memory / engine facts

| | BF16 | FP8 |
|---|---|---|
| Weights per GPU (TP2) | 25.64 GB | 14.48 GB (−44%) |
| KV pool per engine | 1,341,929 tok | 1,539,220 tok (+15%) |
| Weight load time | 33 s | 167 s (fp8 checkpoint load + DeepGEMM warmup on first start) |
| Spec accept (engine logs) | accept len ~3.2–3.6 | accept len ~2.6–3.8 (similar band) |

## Training health under FP8 rollout (first 3 steps)

- `train/ppo_kl` ≈ 0.0012, `pg_clipfrac` ≈ 0.3–0.4% — the rollout(FP8) vs
  train(BF16) logprob mismatch is small; no instability signature.
- `perf/update_weights_time` ≈ 0.8–0.9 s/resync (4 s on the first) — the
  BF16→FP8 blockwise requant during weight sync adds no measurable cost vs
  the baseline's 0.8 s.
- Raw reward 0.100 / 0.128 / 0.077 at steps 0–2 vs baseline first-launch
  0.066–0.127 over steps 10–20 — same band, no quality red flag yet.
- The feared FP8+GDN decode stall (35B-A3B, 2026-07-01) did **not** reproduce
  on the dense 27B: episodes complete normally, queues stay empty.

## Why FP8 barely moves throughput here

1. **Wall time is not decode-bound.** Episodes spend most of their elapsed time
   in sandbox tool exec (g++ compiles, stress loops) and judge grading; the
   batch is gated by straggler episodes (`longest_sample_tok/s` ~33–43 in both
   runs).
2. **Per-engine concurrency is low** (typically 1–4 running requests), and
   EAGLE already converts bandwidth into parallel token verification. The
   fraction of step time where weight-read bandwidth is the binding constraint
   is small.
3. Blockwise-FP8 GEMMs pay dequant/scale overhead at small batch, eating part
   of the bandwidth saving; the GDN layers keep their in_proj_a/b, conv1d, and
   state kernels in BF16 regardless.

FP8's +15% KV pool would matter at higher concurrency (e.g. larger batch or
over-sampling), not at this batch size.

## Verdict / recommendation

- Keep BF16 for this recipe if the goal is speed at batch 256 — FP8 gives ~0%
  end-to-end here.
- FP8 is *viable* (stable, cheap resync, more KV headroom), so it becomes
  interesting if you (a) raise concurrency per engine substantially, or (b)
  shrink the rollout fleet and lean on the memory savings.
- The FP8 run was stopped after step 3; if a tighter decode-speed estimate at
  the trained-policy length regime (~47k responses) is ever needed, relaunch
  the fp8 config and let it run ~10 steps.

## Files

- `metrics.json` — per-step numbers for all three runs + engine static facts.
- `fp8.launch.log` — full launch/run log of the FP8 run.
- Config: `multinode-training-guide/slime/configs/w_qwen3_6_27b_frontier_cs_noncolocate_5n_fp8.py`
  (docstring documents the checkpoint-vs-updater quantized-set verification).
