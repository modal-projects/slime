# Frontier-CS judge profiling report

- Judge: `https://ta-01kw0dpd38svn8nabcgzymwwvs-8081-1flev67yamuf7o6p6c4a2e9c2.w.modal.host` (Modal vm_runtime sandbox `sb-B5aQiqjOp`, cpu=64, mem=131072MB, **nproc=64** = real go-judge worker count)
- Boot: 5.2s · problems visible to judge: 188
- Subset: 0 problems profiled (stratified by n_cases × time-limit; 22 interactive) · oracle calibration: 3 AC / 38 partial

> Oracle = highest-scoring calibrated candidate per task (reference*.cpp where present, else strongest model solution). Phase-1 latencies are uncontended (concurrency=1); phase-2 fires AC oracles concurrently so latency reflects judge queueing, not solution slowness — real training mixes in slower TLE submissions.

## Concurrency sweep (end-to-end submit→verdict latency)

| concurrency | done | errors | drain (s) | throughput (/s) | p50 (s) | p90 (s) | p99 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|
| 50 | 50 | 0 | 17.72 | 2.82 | 3.487 | 14.131 | 16.699 | 17.72 |
| 100 | 100 | 0 | 23.18 | 4.31 | 11.492 | 15.089 | 22.017 | 23.16 |
| 200 | 200 | 0 | 54.35 | 3.68 | 13.854 | 20.694 | 50.14 | 54.33 |

## Artifacts

- `judge.json` · `stratification.json` · `oracles.json`
- `calibration.jsonl` · `phase1_trials.jsonl` · `phase2_requests.jsonl`
- `phase1_summary.{json,csv}` · `phase2_summary.{json,csv}`