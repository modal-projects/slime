# Frontier-CS judge profiling report

- Judge: `https://ta-01kw06fe2haj85vr0we2vb1tes-8081-py8i1zdxc10mibxl145gygc8h.w.modal.host` (Modal vm_runtime sandbox `sb-3XikfTimO`, cpu=16, mem=32768MB, **nproc=16** = real go-judge worker count)
- Boot: 4.4s · problems visible to judge: 188
- Subset: 0 problems profiled (stratified by n_cases × time-limit; 22 interactive) · oracle calibration: 3 AC / 38 partial

> Oracle = highest-scoring calibrated candidate per task (reference*.cpp where present, else strongest model solution). Phase-1 latencies are uncontended (concurrency=1); phase-2 fires AC oracles concurrently so latency reflects judge queueing, not solution slowness — real training mixes in slower TLE submissions.

## Concurrency sweep (end-to-end submit→verdict latency)

| concurrency | done | errors | drain (s) | throughput (/s) | p50 (s) | p90 (s) | p99 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|
| 50 | 50 | 0 | 62.14 | 0.8 | 40.282 | 53.248 | 59.731 | 62.13 |
| 100 | 100 | 0 | 73.08 | 1.37 | 23.218 | 53.45 | 58.175 | 73.06 |
| 200 | 200 | 0 | 96.81 | 2.07 | 52.572 | 89.78 | 96.416 | 96.78 |

## Artifacts

- `judge.json` · `stratification.json` · `oracles.json`
- `calibration.jsonl` · `phase1_trials.jsonl` · `phase2_requests.jsonl`
- `phase1_summary.{json,csv}` · `phase2_summary.{json,csv}`