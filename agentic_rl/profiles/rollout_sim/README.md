# rollout_sim — production-scale rollout inference simulator

Replays **real Frontier-CS episodes** against a **one-node engine fleet**
identical to production, so inference-stack changes can be measured under the
real load shape without sandboxes, judges, or training.

Production reference (retro/rlag runs): 4 rollout nodes × 8 H200 = 16 TP2
SGLang engines; 32 groups × 8 = 256 episodes per step queued against the
fleet (in-flight pool up to the staleness window; measured average ~111
in-flight in the P50 run). One node's share ≈ **4 engines, 64 in-flight
episodes** — the simulator's defaults.

## Pieces

- `prepare_trace.py` — extracts replayable traces from a real rollout dump
  (`save_debug_rollout_data` on the `slime-checkpoints` volume): real prompt
  ids, real per-turn generation lengths (`loss_mask==1` runs), real
  observation token blocks (`loss_mask==0` runs), real per-turn non-LLM pause
  (`timing.agent − timing.generate`). Trace `p50_rlag4_r4` (rollout_4 of the
  rlag4 run, 196 episodes: 30.3 turns / 37k gen tokens / 11.9k prompt /
  5.4 s pause per turn on average) lives at
  `slime-data:/data/profiles/rollout_sim/traces/`.
- `modal_sim.py` — boots N engines on one H200:8 container with the exact
  production ServerArgs, then drives the in-flight episode pool with sticky
  hash routing. Each turn generates **exactly the recorded number of tokens**
  (`ignore_eos`) from the episode's live context, appends the generated
  tokens (radix cache behaves like production) plus the recorded observation
  block, then sleeps the episode's measured pause. Same token counts in every
  config → wall-clock and tok/s are directly comparable across runs.

## Run

```bash
export MODAL_ENVIRONMENT=junlin-dev

# baseline: prod engine config, deterministic OFF (the new default)
uv run --with modal modal run -d agentic_rl/profiles/rollout_sim/modal_sim.py \
    --trace p50_rlag4_r4 --label baseline

# examples of optimization passes
... --deterministic --label det-on          # the old retro config
... --spec-draft 0 --label no-eagle         # disable speculative decode
... --spec-steps 4 --spec-topk 2 --spec-draft 8 --label eagle-448
... --inflight 128 --label queue-2x         # saturate harder
... --engines 2 --tp 4 --label tp4          # fewer, bigger engines
... --mem-fraction 0.9 --label mem09
... --server-extra "--schedule-conservativeness 0.3" --label sched03
```

Results: `slime-data:/data/profiles/rollout_sim/results/<stamp>-<label>.json`
(summary + per-episode records + tok/min timeline + 5 s occupancy samples),
summary printed to app logs. Headline metric for comparing configs:
**`gpu_tok_per_s_saturated`** (compare to production
`perf/tokens_per_gpu_per_sec` ≈ 153 for P50); `*_run_avg` is reported too but
includes the closed-pool drain. Also: `inflight_episodes` /
`inflight_requests[_per_engine]` (what the engines actually saw), per-stream
`decode_tok_per_s`, `episode_elapsed`, EAGLE `spec_accept_length`,
`prefix_cache_hit_rate`.

## Baseline result (2026-08-21)

`20260821-075246-baseline-detoff.json` — prod engine flags, deterministic
OFF, 4×TP2, 64 in-flight, all 196 episodes, pause_scale 1.0:

| Metric | Value | Production reference |
|---|---|---|
| **fleet tok/s, saturated window** | **~4560** | — |
| **node tok/GPU/s, saturated** | **~570** | 153 (P50, det on) · 221–289 (baselines, det off) |
| fleet tok/s, whole-run average | 2871 (= 359 tok/GPU/s) | — |
| decode tok/s per stream (mean / p50 / p90) | 111.6 / 109.6 / 150.0 | ~75 (P50) · ~135–149 (baselines) |
| EAGLE accept length | 3.18 | never recorded in training |
| prefix-cache hit rate | 0.979 | 0.98 ✓ |
| episode elapsed (mean / p90 / max) | 512 / 803 / 1342 s | 780 (P50 fresh mean) |
| episode errors | 0 / 196 | — |

Read: while the pool is full (16 episodes/engine, deterministic off) one node
sustains **~3.7×** the P50 run's realized per-GPU throughput. The engines are
not the binding constraint in the training runs.

**Use `*_saturated`, not the run average.** The pool here is *closed* (a fixed
episode list), so the run ends in a long straggler drain — 18 of 42 minutes
carrying only 8% of the tokens — which drags any whole-run average down by
~35%. Real training refills the pool continuously, so the saturated window is
the comparable number. `simulate()` measures pool occupancy every 5 s and
defines the saturated window as the minutes where in-flight episodes were
≥90% of `--inflight` (falling back to ≥50%-of-peak throughput if sampling is
missing). An earlier version of this report quoted a "steady" figure that
merely dropped the first and last minute and therefore still averaged in the
whole drain — it understated throughput as 2962 tok/s.

## Concurrency ablation (2026-08-22) — KV usage is the operating constraint

Saturated-window throughput vs in-flight episodes per node (4×TP2, EAGLE 3/1/4,
bf16, det off):

| inflight/node | eps/engine | GPU tok/s | decode/stream | KV p50/p90/max | TTFT |
|---|---|---|---|---|---|
| 32 | 8 | 389 | 152.8 | 0.09 | — |
| 64 | 16 | 570 | 111.6 | — | — |
| **96** | **24** | **728** ← peak | 85.2 | **0.35** / 0.52 / 0.73 | — |
| 128 | 32 | 687 | 87.6 | 0.37 / **0.92 / 0.99** | 4.8 s |
| 192 | 48 | **287** (collapsed) | 34.6 | **0.97** / — / 1.00 | 11.9 s → 49 s |

96 is the peak. 128 is slightly *worse* despite 33% more concurrency because it
bursts to KV 0.99 and pays intermittent preemption; 192 sits there permanently.

**EAGLE on vs off at inflight 96** (matched 288-episode workload, 287 clean
episodes each): 728 vs 476 GPU tok/s (**1.53×**), per-stream 85.2 vs 49.6
(1.72×), episode p50 649 s vs 1032 s (0.63×), accept length 3.15. The draft
model costs 15% of the KV pool (1.347M vs 1.585M) — but disabling EAGLE moves
you *toward* the cliff, not away: slower streams mean episodes live ~60% longer,
so KV peaks *higher* (0.80 vs 0.73) despite the larger pool. Keep EAGLE.

**The cliff.** At inflight 192 the KV pool fills and throughput collapses to
2.5× *below* inflight 96. The minute-by-minute trace shows the mechanism:

| minute | 1 | 6 | 11 | 13 | 16 | 25 |
|---|---|---|---|---|---|---|
| tok/min (k) | 44 | 539 | 354 | 163 | 96 | 105 |
| KV usage | 0.14 | 0.56 | 0.80 | 0.91 | 0.96 | 0.97 |
| TTFT (s) | 3.6 | 3.5 | 6.7 | 14.9 | 27.3 | 42.4 |

TTFT is flat while KV < 0.8, then explodes ~14× as KV pins near 1.0. Note that
minute 6 reached 1123 tok/GPU/s — *better* than inflight 96 — so the config is
faster right until KV fills, then falls off a cliff.

**Rules this implies.**

1. **Keep engine `token_usage` below ~0.8.** That, not throughput, is the
   leading indicator; TTFT is the early warning (flat, then vertical).
2. **Never trust a short run on this workload.** Episodes start with ~12k
   contexts and grow to ~57k, so KV occupancy climbs for ~13 minutes before
   equilibrating. A 16-episode partial of the 192 arm reported 733 tok/GPU/s —
   2.5× the true sustained value. Always read the equilibrium minutes.
3. In production, log per-engine `sglang:token_usage`; slime does not surface it
   today, so this cliff would be invisible until throughput dropped.

## SGLang version / topology / spec-algo probes (2026-08-23)

All at inflight 96, 288 episodes, same trace. Standard metric set.

| config | GPU tok/s | decode/stream | TTFT | KV p50/p90/max | KV pool | cache hit | accept | ep p50 |
|---|---|---|---|---|---|---|---|---|
| v0.5.12 TP2 EAGLE (prior) | 728 | 85.2 | n/a | 0.35/0.52/0.73 | 1,347,417 | 0.980 | 3.15 | 649 s |
| **v0.5.18 TP2 EAGLE** ✅ | **978** (1.34×) | **126.3** | 1.21 s | 0.22/0.38/0.57 | **1,688,962** | 0.979 | 3.16 | **480 s** |
| v0.5.18 TP2 DFLASH-8 | 934 (1.28×) | 118.6 | 1.31 s | 0.31/0.56/0.86 | 1,190,099 | 0.979 | **3.32** | 513 s |
| v0.5.18 TP1×8 EAGLE | 788 (1.08×) | 93.7 | 1.91 s | 0.38/**0.77/0.99** | 530,685 | 0.942 | 3.18 | 602 s |
| v0.5.15.post1 TP2 EAGLE (slime nightly-20260810a) ★ | 844 (1.16×) | 102.0 | 1.65 s | 0.25/0.40/0.60 | 1,698,067 | 0.975 | 3.17 | 551 s |

**Upgrading SGLang 0.5.12 → 0.5.18 is worth +34% on its own** — larger than any
knob in the concurrency/spec ablations — and it grows the KV pool 25%
(1.347M → 1.689M), cutting KV pressure from 0.35 to 0.22 and buying back
headroom against the cliff. Episodes finish 26% faster.

**★ 0.5.15.post1 is the canonical prod version (2026-08-24).** 0.5.18 cannot go
into the slime training image: its dependency closure forces torch 2.11+cu129 →
torch 2.13 + the CUDA-13 stack, which Megatron/Apex/TE are not built against,
and upstream slime's own Dockerfile pins `v0.5.15.post1-cu129`. The newest slime
nightly (`slimerl/slime:nightly-dev-20260810a-cu129`) ships 0.5.15.post1 on the
unchanged torch stack and captures ~half the 0.5.12→0.5.18 gain (+16%, 728→844;
measured here with that exact image via `ROLLOUT_SIM_IMAGE`). All slime-fork
sglang touchpoints import clean against it and the prod-flag boot smoke passed.
Both launch chains now pin it (`agentic_rl/retro/launch_config.py`,
`multinode-training-guide/slime/configs/base.py`). Revisit 0.5.18 (+16% more)
when upstream slime moves to the cu13 stack.

**TP2 beats TP1.** Eight TP1 engines replicate the 54GB weights 8× instead of
4×, leaving only 530k tokens/engine (4.25M fleet vs 5.39M). At the same node
concurrency KV runs p90 0.77 / max 0.99 — brushing the cliff where TP2 sits at
0.38 — and the smaller radix tree drops cache hit to 0.942.

**DFLASH block 8 loses narrowly to EAGLE** (934 vs 978) despite a *better*
accept length (3.32 vs 3.16): the 8-token verify window costs more compute per
step than the extra accepted tokens repay, and its draft checkpoint gives up 30%
of the KV pool (1.190M vs 1.689M). Worth retrying at block 4-6 before dismissing.

## Measured replay shape (baseline run)

Per turn the client re-**sends** the whole growing context, but the engine only
**computes** the new observation block — the generated tokens it just produced
are already in its radix tree:

| | per episode (30 turns avg) |
|---|---|
| `input_ids` bytes sent across turns | 953,455 tokens |
| of that, radix-cache hits | 933,133 (97.9%) |
| **actually prefilled** | **20,322** (~680/turn ≈ one observation block) |
| decoded | 36,722 |
| final context | 56,697 (max 64,512) |

Occupancy: 64 in-flight *episodes* (semaphore held for the whole episode), of
which ~67% are awaiting `/generate` at any moment — the rest are in a tool
pause. So the engines see ~43 concurrent requests at saturation ≈ **11 per
engine**, against 16 episodes/engine. (Whole-run averages including the drain:
40 episodes, 27 requests, 6.7/engine.)

## Fidelity notes

- Generated content diverges from the recorded trajectory (the point is
  shape: real turn lengths, real contexts sizes, real pauses, real prompt
  sharing across the 8-sibling groups is *not* preserved since traces are
  per-episode — prompt prefixes still repeat when `--episodes` exceeds the
  trace pool).
- The replay appends the **model's own** output each turn, never the recorded
  ground-truth tokens. Splicing recorded tokens back in would diverge from
  what the engine holds at the first generated token, invalidating the radix
  prefix and forcing a full re-prefill of the whole context every turn — tens
  of thousands of extra prefill tokens per turn and a workload nothing like
  production.
- Episodes are assigned to engines round-robin (`ep_idx % engines`), sticky for
  the episode's lifetime. Production routes by `session_id` consistent hashing
  through the router; round-robin is used here because it guarantees balance
  and is reproducible (`hash()` on strings is PYTHONHASHSEED-randomized, so it
  would shuffle load between otherwise-identical A/B runs).
- `ignore_eos` forces recorded lengths, so EAGLE accept length is measured on
  live text but throughput comparisons are token-count-matched.
- One node only: no router process, no cross-node weight-resync pauses.
- The trace comes from a deterministic-on run; turn lengths are policy
  behavior, not engine behavior, so they transfer.
