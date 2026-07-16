# Frontier-CS judge profiling report

- Judge: `https://ta-01kw09fsnkz81d93fd53c5ayns-8081-gci6n7r0dyei8wvx93s2kfbrh.w.modal.host` (Modal vm_runtime sandbox `sb-3tQhUKs63`, cpu=32, mem=65536MB, **nproc=32** = real go-judge worker count)
- Boot: 145.9s · problems visible to judge: 188
- Subset: 0 problems profiled (stratified by n_cases × time-limit; 22 interactive) · oracle calibration: 3 AC / 38 partial

> Oracle = highest-scoring calibrated candidate per task (reference*.cpp where present, else strongest model solution). Phase-1 latencies are uncontended (concurrency=1); phase-2 fires AC oracles concurrently so latency reflects judge queueing, not solution slowness — real training mixes in slower TLE submissions.

## Concurrency sweep (end-to-end submit→verdict latency)

| concurrency | done | errors | drain (s) | throughput (/s) | p50 (s) | p90 (s) | p99 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|
| 50 | 50 | 0 | 42.81 | 1.17 | 13.974 | 27.504 | 36.061 | 42.8 |
| 100 | 100 | 0 | 29.96 | 3.34 | 18.248 | 25.194 | 28.088 | 29.95 |
| 200 | 200 | 0 | 82.59 | 2.42 | 25.23 | 44.197 | 50.105 | 82.57 |
| 50 | 50 | 0 | 30.43 | 1.64 | 7.816 | 28.54 | 29.705 | 30.42 |
| 50 | 50 | 0 | 41.83 | 1.2 | 6.101 | 10.29 | 39.05 | 41.82 |

## Artifacts

- `judge.json` · `stratification.json` · `oracles.json`
- `calibration.jsonl` · `phase1_trials.jsonl` · `phase2_requests.jsonl`
- `phase1_summary.{json,csv}` · `phase2_summary.{json,csv}`