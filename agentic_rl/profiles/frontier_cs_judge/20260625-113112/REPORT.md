# Frontier-CS judge profiling report

- Judge: `https://ta-01kw00swyfb1tvz61e8sz8ezrs-8081-ziudnkl8x1g9g6br6u542u31x.w.modal.host` (Modal vm_runtime sandbox `sb-kcqkxvh4n`, cpu=8, mem=16384MB, **nproc=8** = real go-judge worker count)
- Boot: 7.8s · problems visible to judge: 188
- Subset: 41 problems profiled (stratified by n_cases × time-limit; 22 interactive) · oracle calibration: 3 AC / 38 partial

> Oracle = highest-scoring calibrated candidate per task (reference*.cpp where present, else strongest model solution). Phase-1 latencies are uncontended (concurrency=1); phase-2 fires AC oracles concurrently so latency reflects judge queueing, not solution slowness — real training mixes in slower TLE submissions.

## Concurrency sweep (end-to-end submit→verdict latency)

| concurrency | done | errors | drain (s) | throughput (/s) | p50 (s) | p90 (s) | p99 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|
| 50 | 50 | 0 | 131.78 | 0.38 | 30.701 | 103.832 | 131.17 | 131.76 |
| 100 | 100 | 0 | 129.47 | 0.77 | 50.446 | 98.673 | 128.919 | 129.45 |
| 200 | 200 | 0 | 255.89 | 0.78 | 126.79 | 220.71 | 249.485 | 255.87 |

## Per-task oracle eval time (concurrency=1, 10 trials)

Across 41 tasks: median-of-p50 = 2.314s, p90-of-p50 = 6.051s, max p90 = 61.654s

| pid | type | cases | TL(s) | oracle | score | p50 (s) | p90 (s) | mean | min | max |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | interactive | 10 | 0.25 | gpt5.2 | 0.0 | 1.006 | 1.022 | 1.006 | 0.982 | 1.022 |
| 229 | default | 7 | 2.0 | gpt5.2 | 38.230000000000004 | 1.022 | 1.115 | 1.091 | 0.99 | 1.766 |
| 144 | interactive | 10 | 4.0 | gpt5.2_2 | 44.93000000000001 | 1.042 | 1.138 | 1.058 | 0.999 | 1.15 |
| 60 | interactive | 10 | 3.0 | gpt5.2_2 | 93.27149 | 1.055 | 1.278 | 1.098 | 0.986 | 1.299 |
| 122 | interactive | 10 | 3.0 | gpt5.2 | 5.4 | 1.288 | 1.361 | 1.292 | 1.151 | 1.384 |
| 253 | interactive | 10 | 3.0 | gpt5.2 | 0.0 | 1.377 | 1.493 | 1.382 | 1.227 | 1.564 |
| 256 | interactive | 10 | 1.0 | gpt5.2_1 | 44.57 | 1.77 | 1.789 | 1.77 | 1.724 | 1.819 |
| 57 | interactive | 15 | 5.0 | gpt5.2_1 | 22.44666666666667 | 1.291 | 1.816 | 1.408 | 1.255 | 2.02 |
| 53 | interactive | 1 | 2.0 | gpt5.2_1 | 33.52 | 1.776 | 1.834 | 1.786 | 1.739 | 1.836 |
| 160 | interactive | 75 | 10.0 | gpt5.2_1 | 13.349066666666666 | 1.824 | 1.862 | 1.765 | 1.304 | 1.9 |
| 30 | interactive | 10 | 4.0 | gpt5.2 | 70.78999999999999 | 1.758 | 1.877 | 1.794 | 1.726 | 1.899 |
| 24 | default | 10 | 1.0 | gpt5.2_3 | 70.87999999999998 | 1.961 | 2.005 | 1.9 | 1.23 | 2.035 |
| 11 | default | 10 | 2.0 | gpt5.2 | 30.0 | 1.929 | 2.035 | 1.938 | 1.828 | 2.091 |
| 5 | default | 10 | 4.0 | gpt5.2_2 | 30.000000000000004 | 1.954 | 2.038 | 1.966 | 1.904 | 2.044 |
| 152 | default | 100 | 10.0 | gpt5.2 | 0.3127 | 2.001 | 2.057 | 2.007 | 1.965 | 2.068 |
| 168 | default | 150 | 10.0 | gpt5.2 | 0.0 | 2.117 | 2.136 | 2.05 | 1.445 | 2.154 |
| 0 | default | 70 | 2.0 | gpt5.2 | 43.771091350000006 | 2.103 | 2.159 | 2.102 | 2.037 | 2.184 |
| 47 | default | 15 | 1.0 | gpt5.2 | 88.80055333333333 | 2.308 | 2.354 | 2.289 | 2.147 | 2.382 |
| 120 | interactive | 100 | 1.0 | gpt5.2_1 | 69.92930000000001 | 2.314 | 2.481 | 2.411 | 2.262 | 3.269 |
| 83 | default | 1 | 3.0 | gpt5.2 | 100.0 | 2.366 | 2.545 | 2.397 | 2.301 | 2.553 |
| 140 | interactive | 10 | 1.0 | gpt5.2_1 | 20.0 | 2.53 | 2.626 | 2.542 | 2.474 | 2.696 |
| 86 | interactive | 10 | 2.0 | gpt5.2_1 | 0.0 | 2.539 | 2.631 | 2.516 | 2.004 | 2.772 |
| 104 | interactive | 10 | 2.0 | gpt5.2 | 0.0 | 2.643 | 2.801 | 2.668 | 2.59 | 2.819 |
| 165 | default | 150 | 10.0 | gpt5.2 | 6.235599999999996 | 2.854 | 2.901 | 2.8 | 2.144 | 3.053 |
| 70 | interactive | 11 | 5.0 | gpt5.2_2 | 83.60272727272726 | 2.976 | 3.154 | 2.746 | 2.041 | 3.171 |
| 4 | interactive | 5 | 5.0 | gpt5.2_1 | 83.96000000000001 | 3.803 | 4.055 | 3.837 | 3.537 | 4.117 |
| 228 | default | 20 | 1.0 | gpt5.2 | 100.0 | 4.103 | 4.189 | 4.058 | 3.473 | 4.193 |
| 178 | default | 20 | 1.0 | gpt5.2_1 | 92.80159 | 4.242 | 4.352 | 4.25 | 4.134 | 4.367 |
| 109 | default | 10 | 1.0 | gpt5.2_2 | 99.983745 | 4.534 | 4.817 | 4.367 | 3.796 | 4.845 |
| 16 | interactive | 10 | 5.0 | gpt5.2 | 50.096000000000004 | 5.002 | 5.362 | 5.056 | 4.763 | 5.521 |
| 263 | default | 10 | 10.0 | reference2 | 95.14022125208 | 5.409 | 5.663 | 5.358 | 5.004 | 5.826 |
| 187 | default | 10 | 2.0 | gpt5.2_1 | 81.34900000000002 | 5.657 | 5.778 | 5.541 | 5.088 | 5.856 |
| 149 | interactive | 75 | 10.0 | gpt5.2 | 0.0 | 1.793 | 5.778 | 5.732 | 1.722 | 41.202 |
| 153 | interactive | 75 | 10.0 | gpt5.2 | 0.0 | 1.829 | 5.934 | 5.859 | 1.777 | 42.098 |
| 183 | default | 10 | 2.0 | gpt5.2_2 | 100.0 | 6.04 | 6.12 | 6.046 | 5.934 | 6.218 |
| 61 | default | 57 | 3.0 | gpt5.2_1 | 0.0 | 5.997 | 6.266 | 6.089 | 5.92 | 6.551 |
| 192 | default | 30 | 1.0 | gpt5.2 | 0.0 | 6.051 | 6.543 | 6.135 | 5.519 | 6.926 |
| 73 | interactive | 100 | 1.0 | gpt5.2 | 80.08010000000003 | 20.354 | 20.946 | 20.483 | 20.089 | 21.022 |
| 209 | interactive | 10 | 10.0 | gpt5.2 | 0.0 | 21.107 | 22.052 | 20.611 | 18.144 | 22.555 |
| 150 | default | 100 | 10.0 | gpt5.2_2 | 24.090600000000002 | 28.068 | 29.412 | 29.189 | 27.853 | 39.249 |
| 231 | interactive | 10 | 15.0 | gpt5.2 | 0.0 | 61.501 | 61.654 | 61.453 | 61.053 | 61.7 |

## Artifacts

- `judge.json` · `stratification.json` · `oracles.json` · `cold_warm.json`
- `calibration.jsonl` · `phase1_trials.jsonl` · `phase2_requests.jsonl`
- `phase1_summary.{json,csv}` · `phase2_summary.{json,csv}`

## Findings

**1. The judge has exactly 8 workers (nproc=8).** go-judge `-parallelism` and Node `JUDGE_WORKERS` both default to the sandbox `nproc` (=8 at the production cpu=8). That is the hard concurrency ceiling everything below queues against.

**2. Cold per-problem compile tax (one-time, cached per pid).** The *first* submission for a problem is far slower than later ones because the judge compiles that problem's testlib checker (`chk.cc`) / interactor (`interactor.cc`) and caches it per-pid (judge_engine.js `getOrCompileChecker`/`getOrCompileInteractor`). Measured across 39 problems:

| | first submit | warm (cached) | cold penalty |
|---|---|---|---|
| p50 | 10.2s | 1.9s | +7.4s |
| p90 | 68.0s | 6.4s | +41.8s |
| max | 72.3s | – | +70.0s |

e.g. pid 168/165 (150-case default): ~72s cold → ~2.2s warm. *Training implication:* on a freshly-booted judge the first episode to touch each problem pays this tax; it amortizes once the checker is cached (the judge is one-per-worker and long-lived).

**3. Warm, uncontended grading is cheap.** Phase-1 ran post-calibration (checkers warm): per-task p50 median **2.3s**, p90-of-p50 **6.0s**; most tasks 1–6s. The heavy tail is solution/interactor-bound, not case-count-bound — pid 231 (interactive, 15s TL) is 61s even warm because the solution times out per case, while pid 168 (150 cases) is 2.1s because its solution fails fast.

**4. Under concurrency, latency explodes and throughput plateaus (~0.78 grades/s).** Firing N warm oracle submissions at once against the 8 workers: a 50/100/200 burst takes ~132 / 130 / 256s to fully drain; median submit→verdict latency climbs 31 → 50 → 127s; **zero errors at every level** — the judge queues gracefully, it does not drop. Throughput saturates by N=100 (0.77/s) and does not improve at N=200, confirming the 8-worker bottleneck; doubling load ~doubles drain time.

**5. Bottleneck implication for training.** One shared judge per rollout worker grades ~0.78 submissions/s. Both the final grade AND the agent's mid-episode `submit.sh` calls hit it. When many episodes submit near-simultaneously, grading serializes: a 200-wide final-grade burst alone needs ~4 min to clear, and mid-episode self-grades compound it. Levers: raise `FRONTIER_CS_JUDGE_CPU` (more go-judge workers), run multiple judges, or pre-warm checkers. NB: phase-2 used *fast* AC-ish oracles; real rollouts include slower TLE submissions, so production contention is worse than these numbers.