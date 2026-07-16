# Frontier-CS judge profiling report

- Judge: `https://ta-01kw06fm1qs5xbqffc6dbmnfrs-8081-8xnv3bxs20tj6h2w2qvnmg5lt.w.modal.host` (Modal vm_runtime sandbox `sb-egtxO2Bjw`, cpu=32, mem=65536MB, **nproc=32** = real go-judge worker count)
- Boot: 7.5s · problems visible to judge: 188
- Subset: 0 problems profiled (stratified by n_cases × time-limit; 22 interactive) · oracle calibration: 3 AC / 38 partial

> Oracle = highest-scoring calibrated candidate per task (reference*.cpp where present, else strongest model solution). Phase-1 latencies are uncontended (concurrency=1); phase-2 fires AC oracles concurrently so latency reflects judge queueing, not solution slowness — real training mixes in slower TLE submissions.

## Concurrency sweep (end-to-end submit→verdict latency)

| concurrency | done | errors | drain (s) | throughput (/s) | p50 (s) | p90 (s) | p99 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|
| 50 | 46 | 4 | 43.76 | 1.05 | 6.397 | 8.413 | 28.107 | 43.76 |
| 100 | 100 | 0 | 31.52 | 3.17 | 14.279 | 27.12 | 29.453 | 31.49 |
| 200 | 200 | 0 | 71.15 | 2.81 | 27.627 | 44.47 | 54.41 | 71.12 |

## Artifacts

- `judge.json` · `stratification.json` · `oracles.json`
- `calibration.jsonl` · `phase1_trials.jsonl` · `phase2_requests.jsonl`
- `phase1_summary.{json,csv}` · `phase2_summary.{json,csv}`