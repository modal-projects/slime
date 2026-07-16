# qwen3.6-27B SWE-rebench-v2 noncolocate — train/rollout time profile (round 1)

**Date:** 2026-07-06 → 2026-07-07
**Base config:** `w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n` (2 train nodes TP4×CP2→DP2 + 4 rollout nodes 16×TP2 SGLang, sync noncolocate, 6×H200:8)
**Protocol:** each variant = production config + one knob, `num_rollout=2`, `eval_interval=None`,
`save_debug_rollout_data=None`, run sequentially on Modal. Step 0 is warmup (engine boot, cuda-graph
capture, cold train step); **all comparisons use step 1 (warm)**.
**Held fixed (study design):** train/eval data + prefilter, `custom_config_path`, DAPO args
(`dynamic_sampling_filter_path`, `over_sampling_batch_size=48`), `global_batch_size=256`,
`actor_num_nodes=2`, `rollout_num_gpus=32`, `async_mode=False`.

The throwaway configs (`configs/train_profile/*`) were deleted after the study; the **exact
overrides** for each variant are recorded below and are all that is needed to reproduce
(`make_slime`-style: overrides applied on top of the production config).

## Variants and exact overrides

| variant | overrides vs production config | layout |
|---|---|---|
| baseline | (none) | train TP4×CP2→DP2, mt=32768; rollout 16×TP2, EAGLE on |
| mt64k | `max_tokens_per_gpu=65536` | same parallelism, 2× tokens per micro-step |
| tp2cp2 | `tensor_model_parallel_size=2` | train TP2×CP2→**DP4** |
| tp4cp1 | `context_parallel_size=1, max_tokens_per_gpu=65536` | train TP4×CP1→**DP4** — **not run** (see below) |
| roll_tp1 | `rollout_num_gpus_per_engine=1` | rollout 32×TP1 engines |
| nospec | `sglang_speculative_algorithm=None` (+ null num_steps/topk/draft companions) | rollout 16×TP2, EAGLE off |

W&B (project `Modal`, entity junlinwang): baseline `0d2s3i0w`, mt64k `7jw5m4j2`, tp2cp2 `m5p4t52y`,
roll_tp1 `y56vybsu`, nospec `5t7x9eqo`. Raw logs: `<variant>.run.log`; parsed numbers: `metrics.json`.

## Results — warm step (step 1)

| variant | rollout_time (s) | train_time (s) | step_time (s) | train tok/s | train TFLOPS | µbatches | update_weights (s) |
|---|---|---|---|---|---|---|---|
| baseline | 2212 | 947 | 3164 | 8806 | 123.6 | 69 | 0.75 |
| roll_tp1 | 2337¹ | 1021 | 3358 | 7681 | 106.2 | 64 | 0.97 |
| nospec | 1979 | 982 | 2966 | 8365 | 117.9 | 68 | 0.81 |
| mt64k | — | **OOM** (step-0 train) | — | — | — | — | — |
| tp2cp2 | — | **OOM** (step-0 train) | — | — | — | — | — |

¹ roll_tp1 step-1 rollout wall taken from `perf/train_wait_time` (the trainer-side rollout wait);
the rollout-side perf line for step 1 was lost when the log stream disconnected (app finished
detached; W&B run `y56vybsu` has the full series).

Step-0 (cold) reference, baseline: rollout 2346 s, train 1575 s, step 3930 s. Cold train is ~1.7×
warm (first-step compile/allocator warmup), confirming the 2-step protocol was needed.

## Noise floor — read this before the numbers

The training side of baseline / roll_tp1 / nospec is **identical**, yet warm train tok/s spans
7681–8806 (±7%) because each run trains on different sampled episodes (different length mix), and
micro-batch counts differ (64–69). The rollout side of baseline / mt64k / tp2cp2 is also identical,
yet step-0 rollout wall spans 1992–2346 s (±9%) — stochastic agent trajectories + Modal sandbox
scheduling. **Single-run deltas below ~10% are noise in both directions.**

- `rollout/response_len/mean` varied 8857–11481 across runs — the dominant confounder.
- `rollout/spec_accept_rate` logs as 0.0 (metric plumbing); engine logs show real accept len ~3.4–3.9,
  accept rate 0.8–0.95 at low batch.

## Findings

1. **Rollout dominates the step (~70%).** Warm `wait_time_ratio` is 0.67–0.70 in every completed
   run: ~2000–2350 s rollout vs ~950–1020 s train. Any training-side win is capped at ~30% of step
   wall. The single biggest available lever is overlapping the two phases (`async_mode=True`, which
   the base config docstring already anticipates), not tuning either side in isolation.

2. **No rollout knob tried so far beats baseline beyond noise.**
   - `nospec` (EAGLE off): 1979 s vs baseline 2212 s (−10%) — right at the noise edge, and its
     response lengths were longer than baseline's, which weakly suggests a real (small) win.
     EAGLE's accept len ~3.8 only pays at the low-concurrency tail; most of the step runs at high
     batch where drafting competes for compute. Worth one repeat before adopting.
   - `roll_tp1` (32×TP1): 2337 s — no better (slightly worse); per-request decode is slower on one
     GPU and the workload is already engine-parallel enough at 16×TP2. Not recommended.

3. **Both memory-expansion variants OOM in the same place: the vocab-parallel logits/entropy
   forward (`ppo_utils.py` `_VocabParallelEntropy` / chunked log-probs), not the transformer.**
   That stage allocates ~tokens_per_rank × vocab_shard fp32 buffers:
   - `mt64k`: inductor buffer (1016, 1, 62080) fp32 (62080 = 248320 vocab / TP4) with 131.8 GiB
     already allocated — 64k tokens/rank at TP4 doesn't fit with full recompute.
   - `tp2cp2`: `vocab_parallel_logits - logits_max` at TP2 (vocab shard 124k, plus 81 GiB
     weights+grads at TP2) — 486 MiB short.
   Implication: DP4 layouts or bigger token budgets are only reachable by first shrinking the
   logits stage (`log_probs_chunk_size` 1024 → 256/512), NOT by more recompute (the transformer
   is already fully recomputed). `tp4cp1` was cancelled untried since it needs the same 64k/rank
   budget that just OOMed at identical weights+grads.

4. **Weight resync is free.** Warm `update_weights_time` ≈ 0.8–1.0 s per step (2 GiB buckets).
   `update_weights_interval=1` costs nothing; no need to touch it.

## Recommended round 2 (in order of expected value)

1. **`async_mode=True`** (train_async.py, everything else unchanged): with wait_time_ratio 0.70,
   full overlap bounds step wall at max(rollout, train) ≈ rollout → up to ~30% throughput gain.
   Semantics shift slightly off-policy by one step (update_weights_interval stays 1) — needs your
   sign-off since it changes the training recipe, not just performance.
2. **`mt48k` + smaller logits chunks** (`max_tokens_per_gpu=49152`, `log_probs_chunk_size=512`):
   capture part of mt64k's micro-step reduction while fixing the actual OOM site.
3. **`tp2cp2` retry with `log_probs_chunk_size=256`**: it was only 486 MiB short; DP4 should cut
   train wall meaningfully if the logits stage fits. (If it fits, tp4cp1+chunk512 is next.)
4. **repeat `baseline` and `nospec` once each**: establishes the run-to-run σ that decides whether
   nospec's −10% rollout is real.

## Artifacts

- `REPORT.md` (this file), `metrics.json` (parsed per-step metrics per variant)
- `<variant>.run.log` — full streamed job logs (roll_tp1's tail recovered via `modal app logs`)
- W&B runs listed above (group `qwen27-swe-train-profile-<variant>-<ts>`)
