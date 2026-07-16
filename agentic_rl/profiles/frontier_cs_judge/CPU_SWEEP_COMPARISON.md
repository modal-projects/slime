# Frontier-CS judge — CPU/worker scaling sweep (cpu 8 vs 16 vs 32)

Same warm oracle pool (33 problems from the cpu=8 calibration), same seed -> **identical workload**; only the judge box size varies. Memory scaled with CPU at 2GB/worker (16/32/64GB) so workers are not RAM-starved. nproc = real go-judge worker count = the concurrency ceiling.

| cpu (nproc / mem) | concurrency | done | errors | drain (s) | throughput (/s) | p50 (s) | p90 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|
| 8 (8 / 16G) | 50 | 50 | 0 | 131.78 | 0.38 | 30.701 | 103.832 | 131.76 |
| 8 (8 / 16G) | 100 | 100 | 0 | 129.47 | 0.77 | 50.446 | 98.673 | 129.45 |
| 8 (8 / 16G) | 200 | 200 | 0 | 255.89 | 0.78 | 126.79 | 220.71 | 255.87 |
| 16 (16 / 32G) | 50 | 50 | 0 | 62.14 | 0.8 | 40.282 | 53.248 | 62.13 |
| 16 (16 / 32G) | 100 | 100 | 0 | 73.08 | 1.37 | 23.218 | 53.45 | 73.06 |
| 16 (16 / 32G) | 200 | 200 | 0 | 96.81 | 2.07 | 52.572 | 89.78 | 96.78 |
| 32 (32 / 64G) | 50 | 46 | 4 | 43.76 | 1.05 | 6.397 | 8.413 | 43.76 |
| 32 (32 / 64G) | 100 | 100 | 0 | 31.52 | 3.17 | 14.279 | 27.12 | 31.49 |
| 32 (32 / 64G) | 200 | 200 | 0 | 71.15 | 2.81 | 27.627 | 44.47 | 71.12 |

## Scaling at the 200-burst (the loaded regime)

| cpu | drain (s) | vs cpu=8 | throughput (/s) | p50 (s) | p90 (s) |
|---|---|---|---|---|---|
| 8 | 255.89 | 1.00x | 0.78 | 126.79 | 220.71 |
| 16 | 96.81 | 2.64x | 2.07 | 52.572 | 89.78 |
| 32 | 71.15 | 3.60x | 2.81 | 27.627 | 44.47 |

## Findings

**More workers help, with sharply diminishing returns past 16.** Sustained throughput: cpu=8 ~0.78/s (plateaus by level 100) -> cpu=16 ~2.0/s (still climbing at 200) -> cpu=32 ~3.2/s peak at level 100, dipping to 2.8/s at 200. The 200-burst drains 256s -> 97s -> 71s. So 8->16 is a ~2.6x win (drain) but 16->32 only buys a further ~1.4x — the Node single-threaded orchestrator, volume I/O, and per-submission case fan-out become the bottleneck, not CPU. (The super-linear 8->16 is partly the added RAM: cpu=8 ran at 16GB, the tightest.)

**Latency roughly halves per doubling** at level 200: p50 127s -> 53s -> 28s; p90 221s -> 90s -> 45s.

**cpu=32 introduced judge-side errors** (4x `AxiosError 500` at level 50, 0 at 100/200). Root cause confirmed below — go-judge work-queue overflow, not a resource limit. In training a 500 becomes a spurious reward-0. cpu=8/16 had zero errors at every level.

**Recommendation: 16 workers is a safe default; 32/64 is viable with the retry fix below.** cpu=16 cuts the 200-burst drain 2.6x (256s->97s) and p50 latency 2.4x with zero errors out of the box. cpu=32 adds ~40% more throughput on top and — with the fix — runs clean.

## Root cause of the cpu=32 errors & the fix (resolved)

The 500s were **not** a CPU/memory/pid/fd limit. Surfacing go-judge's response body (judge_engine.js was discarding it as bare `String(e)`) revealed the real reason: `gojudge: worker queue is full`. go-judge has a bounded internal work queue; when a synchronized burst of `/run` requests exceeds (parallelism + queue) it rejects the overflow with HTTP 500 **before executing**. Amplified at high worker counts because faster compiles let all concurrent submissions' per-case fan-out flood go-judge at the same instant (cold checker-compiles add to the flood — a cold first burst threw 6 errors, warm bursts 0). Sandbox limits probe was healthy: pid_max 49152, ulimit -u 257215, trivial load.

**Fix** (`judge/src/gojudge.js`): bounded retry with exponential backoff + jitter (<=8 attempts, 150ms->3s) on `/run` when the body matches "queue is full" / on connection resets — safe + idempotent because the request was rejected pre-execution. Plus `judge_engine.js` now appends the go-judge body to the error string so this is never invisible again.

**Validation** (patched cpu=32 / 64GB, warm, sweep 50,100,200,50,50): **0 errors / 450 submissions**, throughput intact (level 100 3.34/s, level 200 2.42/s, p50/p90 unchanged). Run dir: `profiles/frontier_cs_judge/fix-cpu32-20260625-140039`.

## cpu=64 / 128GB + full scaling curve

Re-ran the sweep at cpu=64 (retry fix in place; 0 errors). 200-burst (loaded regime), error-free runs only:

| cpu (nproc / mem) | drain @200 (s) | throughput @200 (/s) | p50 @200 (s) | p90 @200 (s) | Δ throughput per doubling |
|---|---|---|---|---|---|
| 8 / 16G | 256 | 0.78 | 127 | 221 | — |
| 16 / 32G | 97 | 2.07 | 53 | 90 | 2.6x |
| 32 / 64G | 83 | 2.42 | 25 | 44 | ~1.4x* |
| 64 / 128G | 54 | 3.68 | 14 | 21 | ~1.5x |

(*32 row = the fix run; identical seeded workload, but ~±15% run-to-run variance — the orig cpu=32 measured 71s / 2.81/s at level 200.) Best sustained throughput (level 100-200): ~0.78 -> ~2.0 -> ~3.0 -> ~4.0 /s.

**Diminishing returns past 16, but more workers keep helping — now error-free with the retry fix.** 8->16 ~2.6x (super-linear, partly the added RAM); 16->32 ~1.5x; 32->64 ~1.3x. Doubling 32->64 (2x cores + 2x RAM) buys ~30% more throughput and roughly halves p50/p90 latency at 200 (25->14s, 44->21s). All zero errors.

**Bottom line.** cpu=32 / 64GB is the knee of the curve: a big jump over 16, healthy ~3/s throughput, zero errors. cpu=64 / 128GB is the max-headroom option — best latency + throughput + zero errors. Run dir: `profiles/frontier_cs_judge/cpu64-20260625-151631`.

**Modal caps a sandbox at 64 cores** (`InvalidError: Function CPU request out of bounds. Must be between 0.125 and 64 cores` — a cpu=72 attempt was rejected). So **64 is the maximum single-judge size**; the default is now cpu=64 / 131072MB. Going past 64 effective workers would require horizontal scaling — multiple judge sandboxes behind a round-robin of URLs — which autostart does not currently do (one judge per worker via a module global).
