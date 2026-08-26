# Async Retro RL on Frontier-CS — Codebase Runbook

> **Audience:** anyone (human or agent) starting work in this repo with zero context.
> **Branch:** `junlin/agentic_async_rl`. **Date audited:** 2026-08-25.
> Every claim below references the actual file. If a path 404s, the runbook is stale — fix it here first.

## 0. What this repo is

A fork of **slime** (the Megatron+SGLang RL trainer — `slime/`, `train.py`, `train_async.py`) plus an overlay package **`agentic_rl/`** that turns it into a fully-async agentic RL system for coding/competitive-programming tasks, trained on Modal. The current research program is **retro replay on Frontier-CS**: capture mid-episode snapshots of promising/recovering agent states, and branch fresh GRPO groups from them ("retro" lane) alongside normal fresh episodes.

Quick answers to "where is X":

| What | Where |
|---|---|
| Launch scripts (the things you actually run) | `agentic_rl/launch/arms/frontiercs_*.sh` |
| The launcher + config compiler behind them | `agentic_rl/launch/modal_train.py`, `agentic_rl/launch/launch_config.py` |
| Retro replay logic | `agentic_rl/retro/` |
| Rollout core (agent loop ↔ sglang seam) | `agentic_rl/generate.py`, `agentic_rl/model.py`, `agentic_rl/sandbox.py` |
| Task environments (harbor / frontier_cs) | `agentic_rl/environment/` |
| Judge (grading server) | `agentic_rl/envs/frontier_cs/judge/` |
| Reward shaping | `agentic_rl/rewards/rewards.py`, `agentic_rl/turn_reward.py`, `agentic_rl/turn_advantage.py` |
| Held-out eval harness + results | `agentic_rl/eval/frontier_cs/` (⚠ launch configs live in the external `multinode-training-guide` repo) |
| Metrics / W&B / debugging | `agentic_rl/metrics.py`, `agentic_rl/obs/dashboard/` |
| Local modifications to slime itself | 10 files, ~+495 lines — see §3.10 |
| Tests | `tests/test_agent/` (20 files) |

## 1. Workflow at a glance

```
OFFLINE DATA PREP (once per dataset)
  envs/<family>/convert.py  (harbor · frontier_cs · swe_rebench)
      → task dirs + train.jsonl/eval.jsonl  → HF → Modal volume `slime-data` (/data)

LAUNCH (per experiment arm)
  agentic_rl/launch/arms/frontiercs_<arm>.sh        # exports RETRO_*/ROLLOUT_MODE/DAPO_FILTER env
    └─ modal run -d agentic_rl/launch/modal_train.py::train
        └─ build_launch_configs()  (launch/launch_config.py:401)  # env → slime CLI flags + runtime env + Modal image
        └─ 6-node H200 clustered fn: rank0 = Ray head → submit `train_async.py <flags>` as Ray job

TRAINING LOOP (inside Ray job)
  fresh lane:  agentic_rl.core.fully_async  ──┐
  retro lane:  agentic_rl.retro.rollout.generate_retro_mixed (retro mode only)
                                                    ├─→ agentic_rl.generate.generate   # 1 call = 1 episode
                                                    │     env = load_env(metadata.task_type)
                                                    │     RecordingModel ↔ sglang /generate (token-in-token-out)
                                                    │     bash ↔ Modal sandbox; submit.sh ↔ judge server
                                                    │     reward computed inline (no RM step)
                                                    └─→ Samples w/ per-turn weight_versions
  behavior-lag gate + DAPO filter (agentic_rl/core/fully_async.py) → Megatron GRPO update
  retro capture: RetroFrontierCsEnv stages /app snapshots mid-episode → manifest JSONL → replay pool

EVAL (held-out avg@3, offline protocol)
  python -m agentic_rl.eval.frontier_cs.plan   # prints per-arm modal commands (run from guide repo)
    → eval-only slime run (num_rollout=0) → rollout_eval_0.pt dump
    → python -m agentic_rl.eval.frontier_cs.aggregate → summary.json → hand-rolled results/*.json

OBSERVABILITY
  agentic_rl/metrics.py → W&B (agentic/*, async/*, retro/*)   |   dumps → agentic_rl/obs/dashboard/
```

## 2. Directory map

Legend: ✅ live · 🟡 legacy (works, superseded) · 📦 external dependency. Layout = the family-scoped structure from §7 (physical moves landed 2026-08-26; old module paths remain importable via deprecation shims for one window — see `tests/test_agent/test_layout_shims.py`).

```
slime/  (repo root — slime fork)
├── RUNBOOK.md                          ✅ this file (kept at root: it documents the fork delta + repo-wide contracts)
├── train.py, train_async.py            ✅ unmodified upstream entries (eval-only branch: train.py:35)
├── slime/                              ✅ the framework — small audited delta, see agentic_rl/docs/SLIME_DELTA.md
│   ├── utils/wandb_utils.py            ✅ +107: sgl-router /engine_metrics scraper
│   └── backends/megatron_utils/…       ✅ small deltas (loss max-metric, absolute weight_version)
├── scripts/models/qwen3.5-27B.sh       ✅ MODEL_ARGS sourced by the launcher
│
├── agentic_rl/                         ✅ THE OVERLAY
│   ├── README.md                       ✅ design doc (model seam first-principles)
│   ├── knobs.py                        ✅ registry of every project env knob + launch-time validation
│   ├── config_example.yaml             ✅ template for --custom-config-path (agentic_* knobs)
│   │
│   ├── core/                           ✅ the model seam (family-agnostic mechanism)
│   │   ├── generate.py                 ✅ slime hook (--custom-generate-function-path): 1 call = 1 episode
│   │   ├── model.py                    ✅ RecordingModel: in-process mini-swe model, token-exact recording
│   │   ├── sandbox.py                  ✅ Modal sandbox = mini-swe bash Environment + deadlines/retries
│   │   ├── prompts.py                  ✅ pinned prompt scaffold + BASH_TOOL + submit sentinel
│   │   ├── timing.py                   ✅ PhaseTimer (used by harbor episodes)
│   │   └── fully_async.py              ✅ de-forked async rollout (--rollout-function-path, vanilla arms)
│   │
│   ├── envs/                           ✅ task families, one dir each
│   │   ├── base.py                     ✅ RolloutEnv contract + ENVS registry (load_env)
│   │   ├── datasets.py                 ✅ dataset-key registry (TRAIN_DATASET) + deterministic volume-side splits
│   │   ├── README.md                   ✅ converter workflow + dataset registry guide
│   │   ├── harbor/                     ✅ shared substrate: env.py (in-place grading, oracle CLI) + convert.py (canonical writer)
│   │   ├── frontier_cs/                ✅ env.py (judge injection, server-side scoring) + convert.py + submissions.py + judge/ (Node+go-judge: autostart, client, server/)
│   │   ├── swe_rebench/                ✅ convert.py (HF → harbor task dirs); env = harbor
│   │   ├── terminal_bench/             ◐ TB 2.1 family: 89 harbor tasks published; 69/20 split pinned; smoke arm ready — Modal oracle+smoke pending
│   │   ├── swebenchpro/                ◐ SWE-Bench Pro family: 731 harbor tasks published; 650/81 split pinned; smoke arm ready — Modal oracle+smoke pending
│   │   └── legacy/                     🟡 quarantined: native swerebench env + openthoughts converter
│   │
│   ├── rewards/                        ✅ family-agnostic reward layers
│   │   ├── rewards.py                  ✅ step shapes (SHAPERS) + episode outcomes (OUTCOMES)
│   │   ├── turn_reward.py              ✅ O3 rollout side          └── turn_advantage.py  ✅ O3 train side
│   │
│   ├── retro/                          ✅ retro replay runtime (§7.1 re-abstraction landed 2026-08-26)
│   │   ├── pool.py                     ✅ ReplayPool + Lease: the ONE owner of transitions + snapshot GC
│   │   ├── protocols.py                ✅ the coupling surface: ScoreTrace / SnapshotBackend / AgentCheckpoint
│   │   ├── backends/                   ✅ modal_snapshot.py (Modal images) · miniswe_checkpoint.py (Chain)
│   │   ├── manifest.py / buffer.py     ✅ record schema + internal JSONL store/queue (behind pool.py)
│   │   ├── env.py                      ✅ RetroFrontierCsEnv: capture (fresh) + replay (branch) episodes
│   │   ├── selector.py                 ✅ branch-point choice (PROMISING/RECOVERY, target fraction)
│   │   ├── agent.py / group.py         ✅ SnapshottingAgent · manifest → 8-sample branch group
│   │   ├── prefetch.py                 ✅ cross-step queue-ahead worker (holds Leases; staleness dose)
│   │   ├── rollout.py                  ✅ generate_retro_mixed (--rollout-function-path in retro mode)
│   │   ├── generate.py                 ✅ 6-line task_type stamp wrapper → core.generate
│   │   └── README.md                   ✅ bootstrap + arm recipes
│   │
│   ├── launch/                         ✅ THE launch surface
│   │   ├── modal_train.py              ✅ ENTRY: modal run …::{train,download_model,download_data,post_process_data,convert_hf_to_megatron_checkpoint}
│   │   ├── launch_config.py            ✅ env-var → full slime config compiler (RetroSlimeConfig + HeldoutEvalSlimeConfig)
│   │   ├── arms/frontiercs_*.sh        ✅ live training arms (~30-line env-var setters)
│   │   ├── arms/{terminalbench21,swebenchpro}_vanilla_smoke.sh  ◐ family onboarding smokes (TRAIN_DATASET switch)
│   │   └── legacy/                     🟡 old guide-repo-stack scripts (SWE/GLM arms, exploratory eval)
│   │
│   ├── eval/frontier_cs/               ✅ held-out avg@3 protocol harness (fully in-repo since step 4)
│   │   ├── arms.json                   ✅ source of truth: protocol block + 11 checkpoint arms
│   │   ├── protocol.py                 ✅ typed registry (EvalProtocol, ArmSpec)
│   │   ├── plan.py                     ✅ ENTRY: python -m …plan → prints in-repo modal commands
│   │   ├── aggregate.py                ✅ ENTRY: dump.pt → strict per-task avg@k summary.json
│   │   ├── rollup.py                   ✅ ENTRY: per-arm summaries → results JSON (bootstrap SE + paired)
│   │   ├── split.py                    ✅ train/eval disjointness + SHA-256 pin
│   │   └── results/                    ✅ iteration-79.json + heldout-avg3-20260824.json (tracked)
│   │
│   ├── obs/                            ✅ observability
│   │   ├── metrics.py                  ✅ --custom-rollout-log-function-path: agentic/* async/* retro/*
│   │   └── dashboard/                  ✅ Bun/TS rollout-dump viewer (Modal web app)
│   │
│   ├── docs/                           ✅ SLIME_DELTA.md · notes_remote_judge_integration.md · progress/ (reports)
│   ├── profiles/                       ✅ perf harnesses (judge, inference A/B, rollout sim, fp8, train profile)
│   │
│   └── (shims)                         🟡 one deprecation window: generate.py, metrics.py, model.py, sandbox.py,
│       environment/…, retro/{launch_config,modal_train}.py — each re-exports its new module
│
├── tests/test_agent/                   ✅ CPU-only suite (fakes for all 4 external boundaries); 9 files in CI
└── 📦 ../multinode-training-guide/     EXTERNAL repo: legacy EXPERIMENT_CONFIG stack — no longer load-bearing
                                        (held-out eval ported in-repo, RUNBOOK §7 step 4)
```

## 3. Subsystem guides

### 3.1 Launch surface — `launch/` (arms/*.sh + modal_train.py + launch_config.py)

The `frontiercs_*.sh` scripts are pure env-var wrappers; **all real config lives in `launch_config.py`**. Chain:

```
frontiercs_<arm>.sh  →  modal run -d agentic_rl/launch/modal_train.py::train
  build_launch_configs(os.environ)         launch/launch_config.py:401
    → RetroSlimeConfig.cli_args()          :349   # every slime CLI flag, reflected from attrs
    → .environment                         :276   # ASYNC_RL_* runtime env → Ray runtime_env
    → ModalLaunchConfig                    :36    # image slimerl/slime:nightly-dev-20260810a-cu129, H200
  train() @clustered(6, rdma=True)         launch/modal_train.py:371
    rank0: Ray head → JobSubmissionClient.submit_job(train_async.py + flags)   :418
    all ranks: periodic checkpoints-volume commit + teardown barrier            :244-354
```

Mode switch: `ROLLOUT_MODE=vanilla|retro` (launch_config.py:84) selects `--rollout-function-path`
(`agentic_rl.core.fully_async.generate_rollout_fully_async` vs `agentic_rl.retro.rollout.generate_retro_mixed`, :129-133) and `--custom-generate-function-path` (:142-145). `DAPO_FILTER=0` disables the dynamic-sampling filter. Vanilla shares everything else, making it the true control.

Reward is **inline** — `rm_type=None`, no `--custom-rm-path` (launch_config.py:168); slime skips its RM step because `sample.reward` is already set (`slime/rollout/sglang_rollout.py:279-283`).

Topology (launch_config.py:102-111, 222-240): Qwen3.6-27B, 2 actor nodes ×8 H200 (TP4/CP2) + 32 rollout GPUs (16 sglang engines TP2, EAGLE on) = 6 Modal nodes, non-colocated, `MODAL_ENVIRONMENT=junlin-dev`.

### 3.2 Rollout core — the model seam

One `generate()` call = one episode, run in a wide worker thread pool (`generate.py:46-57`). The design intercepts at the model API: stock mini-swe `DefaultAgent` runs **in-process on the head node**; its model is `RecordingModel` (`model.py:88`), which POSTs raw token ids to sglang `/generate` and records exact `(tokens, loss_mask, logprobs, weight_version)` per turn. Only bash crosses into the Modal sandbox (`sandbox.py:115 exec`). Key invariants:

- Never re-render a past assistant turn — only the new observation delta is rendered (`model.py:292 _render_continuation`), so tokenizer round-trip drift can't split prompt from training target.
- Per-turn `weight_versions` recorded → enables TIS and the behavior-lag gate.
- Weight update aborts in-flight generation → `Status.ABORTED` → slime recycles (train only; `generate.py:144-146`); dead-sandbox episodes ship as fully-masked reward-0 `_ship_null` (`generate.py:246`), never bare ABORTED.
- Sibling fan-out divides reward by k (`generate.py:222 s.reward = result.reward / k`).

### 3.3 Task environments — `environment/`

`RolloutEnv` (base.py:61) owns one family's whole episode; rows route by `metadata.task_type` — `"harbor"`, `"frontier_cs"`, `"swerebench"`, or any `"pkg.module:Class"` spec (base.py:151 `load_env` — this is how retro injects its env). The schema each env reads is written **only** by its family's `envs/<family>/convert.py`.

**Frontier-CS episode, end-to-end** (the path that matters):

1. `FrontierCsEnv.normalize_metadata` (frontiercs.py:76) — needs `verifier.env.PROBLEM_ID`; prompt = `statement.txt` + `AGENT.md` (:83).
2. Judge autostart once per worker (frontiercs.py:123 → verifier_server/autostart.py:73): vm_runtime Modal sandbox, `slime-data` mounted, exports `FRONTIER_CS_JUDGE_URL`. Self-heals; idle self-exit code 86.
3. Per-episode `agent_id = uuid4()` injected into the sandbox env (:126-139) → the agent's iterative `bash /app/submit.sh` self-grades mid-episode against the judge.
4. Sandbox boot + agent loop via `HarborEnv._episode` (harbor.py:152) → `base.run_agent_leg` (base.py:80).
5. Final grade is **server-side**: `_verify` (frontiercs.py:186) reads `{workdir}/solution.cpp` and calls `judge_client.grade_solution` (verifier_server/client.py:74); reward = `clamp01(score/100)`. Mid-episode submissions come back trusted via `fetch_agent_submissions` + `merge_server_submissions` (submissions.py:112) with forgery tripwires; the sandbox log is enrichment only.
6. Shaping: per-step `rewards.shape` (SHAPERS: fractional|binary|thresholded), then episode-level `shape_outcome` (OUTCOMES: final|best|disc_sum; `ASYNC_RL_OUTCOME_REWARD`, `ASYNC_RL_OUTCOME_GAMMA`, `ASYNC_RL_SOLVED_BONUS` — rewards.py:126-209).

Oracle check (reference solution through the exact rollout path):
`python -m agentic_rl.envs.harbor.env out/tasks.jsonl --task-root out --limit 3` (harbor.py:327).

### 3.4 Retro replay — `retro/`

Capture → activate → replay lifecycle:

1. **Capture** (fresh episodes under `RetroFrontierCsEnv`, env.py:47): a post-tool callback feeds the submissions log to `EventSelector` (selector.py:71), which classifies PROMISING/RECOVERY states; qualifying states are staged (`cp -a /app → /tmp/retro_candidates/…` + head-side `ChainCheckpoint`, env.py:214). At episode end, the candidate nearest `RETRO_TARGET_TRAJECTORY_FRACTION` is snapshotted (Modal directory snapshot, snapshot.py:41) and appended as a **tentative** manifest to the JSONL `ManifestStore` (env.py:301, buffer.py:16).
2. **Activate**: only groups the fresh lane accepts into a training batch flip their candidates TENTATIVE→AVAILABLE (accept/reject hooks, rollout.py:365-383); rejected snapshots are deleted.
3. **Replay**: `RetroBuffer.lease` filters by event type + policy age (buffer.py:144); `make_branch_group` (group.py:22) deep-copies the template into a **hard-pinned 8-sample group**, stamping `task_type = "agentic_rl.retro.env:RetroFrontierCsEnv"` + `retro_manifest`. Branch episodes restore the workspace + model chain (fully masked inherited prefix, retro/model.py:75) and continue with the remaining step/time budget.
4. **Mix**: `generate_retro_mixed` (rollout.py:592) runs fresh and retro legs concurrently; `retro_group_split` (rollout.py:386) sets the dose from `ASYNC_RL_RETRO_GROUP_RATIO` (0 ⇒ harness-overhead control); shortfall backfills with fresh groups.

**Three orthogonal staleness axes** (do not conflate):
- **Policy age** of the snapshot: `manifest.policy_age()` bounded by `ASYNC_RL_RETRO_MIN/MAX_POLICY_AGE` (default 0/4).
- **Prefetch depth**: `ASYNC_RL_RETRO_PREFETCH_BATCHES` (default 0 = made-to-order). Depth>0 starts `RetroPrefetchWorker` (prefetch.py:111) whose queued groups age across updates — this is what produces realized behavior lag >1.
- **Behavior-lag hard gate**: `ASYNC_RL_RETRO_MAX_BEHAVIOR_LAG` (falls back to `--rollout-max-behavior-lag`), enforced at drain (prefetch.py:344) and post-generation (rollout.py:305).

The fresh lane has its own pair: `--rollout-prefetch-batches` (capacity) vs `--rollout-max-behavior-lag` (guarantee) — see §3.10.

### 3.5 Reward layers (three, stackable)

| Layer | Where | Wire-up | Knobs |
|---|---|---|---|
| Per-step shape | `rewards/rewards.py` SHAPERS | called by env | `ASYNC_RL_REWARD_SHAPE`, `_THRESHOLD` |
| Episode outcome | `rewards.py` OUTCOMES (final/best/disc_sum) | called by harbor.py:223 | `ASYNC_RL_OUTCOME_REWARD`, `_GAMMA`, `_SOLVED_BONUS` |
| Turn-level mix (O3) | `turn_reward.py` (rollout) + `turn_advantage.py` (train) | `--custom-reward-post-process-path` + `--custom-advantage-function-path` | `ASYNC_RL_TURN_REWARD=next_sub`, `ASYNC_RL_TURN_MIX_WEIGHT` (0.3) |

O3 alignment is wall-clock: `turn_ts` from model.py:221 joined to judge-server timestamps; turn advantages painted over response-relative `turn_spans` with CP-aware re-slicing (turn_advantage.py:60), degrading to pure outcome on any shape surprise. Off unless a config sets both hook paths (only the guide-repo `…_o3_turn.py` does today).

### 3.6 Eval — `eval/frontier_cs/` (fully in-repo since 2026-08-26)

Strict held-out protocol: 38 tasks × avg@3, deterministic sampling (seed 20260802 + sample index), pinned in `arms.json` and typed by `protocol.py`. Workflow (all commands run from THIS repo — the guide-repo dependency is gone, RUNBOOK §7 step 4):

```
python -m agentic_rl.eval.frontier_cs.plan [arm …]     # prints 3 modal commands per arm
#   ROLLOUT_MODE=eval … ::download_data   (pull + sha-validated 150/38 split proof)
#   ROLLOUT_MODE=eval … ::train           (num_rollout=0 eval-only branch of slime train.py)
#   ROLLOUT_MODE=eval … ::post_process_data  (strict avg@3 summary.json next to the dump)
python -m agentic_rl.eval.frontier_cs.aggregate <rollout_eval_0.pt> --output summary.json  # offline re-agg
python -m agentic_rl.eval.frontier_cs.rollup --summary p50=… --pair "p50 - base" --output results/…  # roll-up
```

Load-bearing facts:
- The eval config is `HeldoutEvalSlimeConfig` (launch_config.py), a `RetroSlimeConfig` sibling selected by `ROLLOUT_MODE=eval`; a registry arm key alone fills checkpoint identity (`FRONTIER_CS_EVAL_ARM=p50`), ad-hoc arms add `FRONTIER_CS_EVAL_RUN_TAG`. Ported attribute-for-attribute from the guide's `w_qwen3_6_27b_frontier_cs_heldout_avg3` (parity-diffed 2026-08-26); the guide copy is now legacy.
- Eval image is pinned to `slimerl/slime:nightly-dev-20260529a` — the 20260810a bump (sglang 0.5.15.post1) renames ServerArgs fields and breaks that slime's `--sglang-*` bridge; the eval config therefore emits the OLD names (`sglang_cuda_graph_bs`, `sglang_mamba_scheduler_strategy`), never the 0.5.15 `_decode` variants.
- Checkpoints: `/checkpoints/swe_ckpts/<run_tag>` at step 79 (74 for the O-cohort); iter_84 ckpts of 5/6 runs are unrecoverable — use 79.
- `results/*.json` roll-ups (bootstrap `paired` CIs, 50k reps, seed 20260817, unit=task) are reproduced by `eval/frontier_cs/rollup.py` (guarded by `tests/test_agent/test_eval_rollup.py`).
- Training-time eval (`eval_interval` during RL) is the same generate path but logs only micro-average `eval/<dataset>` to W&B — the harness recomputes strictly from the dump.

### 3.7 Metrics & debugging

- `metrics.py:490 log_rollout_data` (`--custom-rollout-log-function-path`) → `agentic/*` (episode behavior, exec stats, submissions, outcome uplift), `async/*` (behavior lag vs trainer version), `retro/*` + per-lane `agentic/fresh|retro/*` splits. **Metric audit caveat:** headline rewards are post-DAPO-filter; use `dynamic_sampling/raw_reward_all` for unbiased reward.
- Debug dumps: `--save-debug-rollout-data` → `slime-checkpoints` volume `/checkpoints/swe_rollout_dumps/<tag>/rollout_{id}.pt`; view with `agentic_rl/obs/dashboard/` (Modal web app) or analyze with `eval/frontier_cs/aggregate.py`.
- sglang engine gauges: local scraper in `slime/utils/wandb_utils.py` polls the router's `/engine_metrics` → `sgl_engine/*` (means across engines).
- Experiment reports: `agentic_rl/progress/retro-replay/` (HTML report + `generate_figures.py` rebuilding from W&B).

### 3.8 The slime fork delta (§ the "no slime edits" claim is stale)

Relative to merge-base with `main`: 8 framework files, ~+233 lines (was 10 / ~+495 before the 2026-08-26 de-fork). The overlay is *mostly* hook-injected, but these are real fork edits an agent must know about:

| File | Δ | What |
|---|---|---|
| ~~`slime/rollout/fully_async_rollout.py`~~ | ~~+215~~ | **De-forked 2026-08-26** → `agentic_rl/core/fully_async.py` (prefetch pool sizing, behavior-lag hard gate, DAPO filter at collection, hooks, `RolloutFnTrainOutput`); slime file back at upstream |
| ~~`slime/utils/arguments.py`~~ | ~~+40~~ | **De-forked 2026-08-26**: the knobs are now `ASYNC_RL_ROLLOUT_PREFETCH_BATCHES` / `ASYNC_RL_ROLLOUT_MAX_BEHAVIOR_LAG` env vars, normalized onto `args` by `core.fully_async.resolve_async_rl_rollout_knobs` |
| `slime/utils/wandb_utils.py` | +107 | engine-metrics scraper |
| `slime/backends/megatron_utils/{cp_utils,loss}.py` | +83 | MAX-reduced metric channel → `train_rollout_logprob_abs_diff_max` |
| `slime/backends/megatron_utils/actor.py` | +5 | absolute `weight_version` across resume (lag gate correctness) |
| `slime/backends/sglang_utils/arguments.py` | +17 | sglang 0.5.15 ServerArgs rename aliasing (uncommitted) |
| `slime/ray/rollout.py` | +17 | `metadata` passthrough (turn spans), scraper wiring |
| `slime/backends/megatron_utils/data.py`, `slime/utils/logging_utils.py` | +11 | metric-loop skips, scraper hook |

## 4. Contracts & invariants (don't break these)

1. **`metadata.task_type` is the routing key**; its schema is written only by the family's `envs/<family>/convert.py` converter. `task_path` resolves against `ASYNC_RL_TASK_ROOT` (= `/data` on Modal).
2. **Reward is set inline in generate**; `rm_type=None`. Never add a custom-rm path without removing the inline reward.
3. **Group width 8** for retro branch groups (group.py:30) and `n_samples_per_prompt=8` for fresh — GRPO normalization assumes it.
4. **`weight_versions` are absolute** across resumes (actor.py fix). The behavior-lag gate and `async/*` metrics depend on this.
5. **ManifestStore is append-only JSONL, last-record-wins** (buffer.py:50); LEASED entries are released on reload. Don't edit in place.
6. **Image pins**: train = `nightly-dev-20260810a-cu129`; held-out eval = `nightly-dev-20260529a`. Don't "upgrade" either casually — sglang 0.5.18 forces torch 2.13/cu13 (incompatible), 0.5.15.post1 renames ServerArgs.
7. **`MODAL_ENVIRONMENT=junlin-dev`** on every modal command — otherwise you run in `main` with empty throwaway volumes and fake success.
8. **Server-side scores only** for Frontier-CS reward (`FRONTIER_CS_SERVER_VERIFY=1` default); the sandbox submissions log is untrusted.
9. **Prompt scaffold is pinned in-repo** (`prompts.py`) — do not "fix" it to track upstream mini-swe.
10. String-loaded entry points (configs reference `agentic_rl.generate.generate`, `agentic_rl.retro.rollout.generate_retro_mixed`, `agentic_rl.metrics.log_rollout_data`, `agentic_rl.retro.env:RetroFrontierCsEnv`) — **moving/renaming these modules breaks launch configs in this repo AND the guide repo**.

## 5. Common tasks

```bash
# Launch a training arm (from repo root)
bash agentic_rl/launch/arms/frontiercs_retro0_s4.sh          # or any frontiercs_*.sh

# Inspect the resolved config without GPUs
RETRO_REWARD_ARM=final RETRO_TARGET_TRAJECTORY_FRACTION=0.75 \
  uv run --with modal modal run agentic_rl/launch/modal_train.py     # show_config local entrypoint

# One-time bootstrap (model / data / megatron ckpt)
MODAL_ENVIRONMENT=junlin-dev uv run --with modal modal run agentic_rl/launch/modal_train.py::download_model
MODAL_ENVIRONMENT=junlin-dev uv run --with modal modal run agentic_rl/launch/modal_train.py::download_data
MODAL_ENVIRONMENT=junlin-dev uv run --with modal modal run agentic_rl/launch/modal_train.py::convert_hf_to_megatron_checkpoint

# Held-out eval (plan prints per-arm in-repo modal commands; aggregate/rollup post-process)
python -m agentic_rl.eval.frontier_cs.plan
python -m agentic_rl.eval.frontier_cs.aggregate <dump.pt> --output summary.json
python -m agentic_rl.eval.frontier_cs.rollup --summary p50=<summary.json> --pair "p50 - base"

# Oracle-check a converted dataset
python -m agentic_rl.envs.harbor.env out/tasks.jsonl --task-root out --limit 3

# Tests (CPU-only, fakes for sandbox/sglang/tokenizer/CLI)
uv run pytest tests/test_agent -x
```

**Adding things (the intended extension points):**
- *New task family*: create `envs/<family>/` with `convert.py` (schema writer) + `env.py` (RolloutEnv subclass, or reuse harbor) + register in `envs/base.py` `ENVS`. Run the oracle check before training.
- *New reward shape/outcome*: add to `SHAPERS`/`OUTCOMES` in `rewards/rewards.py`; select via `ASYNC_RL_REWARD_SHAPE` / `ASYNC_RL_OUTCOME_REWARD`. No env edits needed.
- *New training arm*: new ~30-line `launch/arms/frontiercs_<arm>.sh` setting env vars; add any new knob to `launch_config.py` (+ its test `tests/test_agent/test_retro_launch_config.py`).
- *New eval arm*: append to `eval/frontier_cs/arms.json`; `protocol.py` validates on load.

## 6. Known debt (cleanup inventory)

| # | Item | Evidence | Suggested action |
|---|---|---|---|
| 1 | `agentic_rl/README.md` claims "no edits to slime/" — false (+495 lines / 10 files) | §3.10 | Rewrite claim; add a `docs/SLIME_DELTA.md` and keep it in PR checklists |
| 2 | ✅ RESOLVED 2026-08-26 (step 4): held-out eval ported in-repo (`HeldoutEvalSlimeConfig`, `ROLLOUT_MODE=eval`). Old guide `EXPERIMENT_CONFIG` classes remain only as legacy history | eval §3.6; `qwen3_6_frontiercs_eval.sh:4` cds into the guide repo | Port `heldout_avg3` into an in-repo eval entrypoint (an `EvalConfig` sibling of `RetroSlimeConfig` + `modal_train.py::eval`); retire guide dependency |
| 3 | ✅ RESOLVED 2026-08-26 (step 5): `agentic_rl/knobs.py` registers every project knob (type, default, consumer, scope); `build_launch_configs` validates and fails on typos; `tests/test_agent/test_knobs.py` scans read sites so the registry cannot drift | — | — |
| 4 | Dead retro code: `phase1.py`, `survey.SurveyWriter`, `generate_retro_survey` (rollout.py:30,120), `snapshot.create_from_filesystem_snapshot`, `prefetch.reset_worker`, FILESYSTEM snapshot kind | retro map §3.4 | Delete (Phase-1 study is concluded; report lives in `progress/`) |
| 5 | `async_rl_research/` = stale predecessor copy; `dashboard/app.py:8-9` still documents old paths | root listing | Salvage `notes_remote_judge_integration.md` into `docs/`, delete the dir, fix dashboard docstring |
| 6 | `tests/test_agent/test_frontier_cs_eval.py` asserts 4 arms, registry has 11 → fails; not in CI matrix | test:32 vs arms.json | Fix assertion, add to `.github/workflows/pr-test.yml` |
| 7 | ✅ RESOLVED 2026-08-26 (step 4): `eval/frontier_cs/rollup.py` reproduces the roll-up (50k reps, seed 20260817, unit=task) | §3.6 | — |
| 8 | Legacy eval-data path: `evalset.py` + `prepare_eval_data.sh` + HF `agentic-rl-evalsets` superseded by guide `datasets.py` registry | envs/README.md migration list | Delete after last legacy config migrates |
| 9 | `envs/legacy/swerebench_env.py` env vestigial (live SWE path = harbor conversion); `openthoughts_agent.py` converter unused | env map Q6 | Move to `environment/legacy/` or delete with its two configs |
| 10 | Dead knobs: `agentic_max_boot_retries` (set everywhere, read nowhere — boot_retries hard-coded in sandbox.py:67); `agentic_grade_timeout` only pads sandbox lifetime | env map Q6 | Wire or delete; document `agentic_eval_timeout` as the real grading budget |
| 11 | Legacy in-sandbox verifier (`CANONICAL_EVALUATE_PY`) ships in every task dir but never runs (server-verify default on) | frontiercs.py:50-52 | Keep as explicit fallback but mark; or stop baking it |
| 12 | Minor: `_verify` reads judge URL from env while `_collect_artifacts` reads `md["judge_url"]` (can diverge); redundant re-filter buffer.py:179; unused `eval_config`/`sglang_config` fields launch_config.py:24 | env+retro maps | One-line fixes during the move |

## 7. Proposed target structure

Goal: an agent lands in `agentic_rl/`, reads one README per layer, and can add an arm/env/reward without touching another layer or an external repo. **Organizing principle: task families are first-class.** A family dir under `envs/` owns everything family-specific — its converter (schema writer), its env subclass, its infra (e.g. the Frontier-CS judge), its eval protocol — while the substrate (harbor runtime, sandbox, model seam, reward layers) stays shared. Frontier-CS is the first family, not the framework.

```
agentic_rl/
├── RUNBOOK.md                  # this file (moved), kept current
├── knobs.py                    # ✅ registry of every ASYNC_RL_*/RETRO_*/FRONTIER_CS_* env knob (landed 2026-08-26)
├── core/                       # the model seam (generate, model, sandbox, prompts, timing)
├── envs/
│   ├── base.py                 # RolloutEnv contract + registry (one entry per family dir)
│   ├── harbor/                 # the shared substrate: HarborEnv runtime + canonical converter
│   ├── frontier_cs/            # env + converter + judge/ (verifier_server moves HERE) + eval protocol
│   ├── terminal_bench/         # ★ planned: TB 2.1 (harbor-native — mostly converter-gate work)
│   ├── swebenchpro/            # ★ planned: SWE-Bench Pro (swerebench-style HF→harbor converter)
│   └── legacy/                 # native swerebench env + openthoughts converter (quarantined)
├── rewards/                    # rewards.py + turn_reward.py + turn_advantage.py (family-agnostic)
├── retro/                      # trimmed; family-agnostic core + per-family capture env (see below)
├── launch/                     # launch_config.py + modal_train.py + arms/*.sh + eval entrypoint ★
├── eval/                       # per-family: eval/frontier_cs/ today; eval/<family>/ is the pattern
├── obs/                        # metrics.py + dashboard/
└── docs/                       # SLIME_DELTA.md ★, progress/, salvaged design notes
```

**Retro across families.** The capture/replay core (`manifest.py`, `buffer.py`, `snapshot.py`, `prefetch.py`, `group.py`, `rollout.py`, `model.py`, `agent.py`) is already family-agnostic; only two seams are Frontier-CS-specific: `retro/env.py` subclasses `FrontierCsEnv`, and `selector.py` reads the judge submissions log as its PROMISING/RECOVERY signal. To retro a new family: give it a `envs/<family>/retro_env.py` capture subclass and a family-appropriate `EventSelector` signal source (for harbor-style families, the per-step `reward.json` trace is the natural analog of the submissions log).

**Planned task families:**

- **Terminal-Bench 2.1** — harbor IS the terminal-bench task format, so this is near-zero env code: convert with the canonical harbor converter and register a dataset key. The real work is the v1 scope-gate audit (`envs/harbor/convert.py` v1 scope gates: linux-only, no GPU/MCP, no docker-compose, no network restrictions, shared verifier only) against the TB 2.1 task set — decide which gates to lift vs how many tasks they exclude. Grading is the existing in-place `test.sh → reward.json` path.
- **SWE-Bench Pro** — follow the SWE-rebench pattern (`envs/swe_rebench/convert.py` as the template): HF rows → harbor task dirs using the prebuilt per-task images directly. Known image quirks from prior runs: empty `ENTRYPOINT` + no keepalive (the shared sandbox's `sleep infinity` boot already handles this), and broken/poisoned pip indexes + missing curl in some images — so keep the grader self-contained (vendored stdlib test parser, as swerebench does) and provision nothing at episode time. Decide in-place vs fresh-sandbox grading explicitly (in-place is the harbor default; fresh-sandbox is the anti-reward-hack option the native swerebench env used).

### 7.1 Retro re-abstraction (clean layering under `retro/`)

The retro machinery works, but its lifecycle logic is spread out in ways that make it hard to read and hard to reuse for a second family. The redesign principle: a **mechanism vs policy split inside `retro/`** — the generic mechanism ("a rollout can be composed from multiple sample sources whose items have lifecycle-managed leases") in `pool/source/mixed`, the policy ("what a snapshot is, when to take one, what makes a state promising") in `capture/selector/backends`. **Everything stays under `agentic_rl/retro/`** — that the mechanism half *could* one day upstream to slime is a motivation for keeping the boundary clean, not a plan.

**How retro rides slime's async lane today** (the redesign preserves this exactly):

```
trainer (train_async.py) ──① next batch(rollout_id)──▶ RolloutManager (Ray actor)
                                                        │  slime's ONLY manager — retro adds no actor
                                                        ▼
                     ② generate_retro_mixed(args, rollout_id)      [in-process library code]
                     ├─ FRESH leg = slime's _generate_rollout_async continuous pool
                     │    · in-flight groups persist ACROSS steps (capacity = prefetch × batch)
                     │    · each episode runs the capture TAP inline: stage at ~target turn
                     │      ──④ Modal snapshot ──⑤ offer TENTATIVE──▶ ManifestStore (JSONL)
                     │    · batch-accept hook ──⑥ commit→AVAILABLE · reject──▶ INVALID+GC
                     ├─ RETRO leg (concurrent with fresh leg)
                     │    · depth>0: drain RetroPrefetchWorker's ready queue (daemon thread,
                     │      survives steps — its queued groups AGE across weight updates ⑦⑧)
                     │    · depth=0 (default): lease AVAILABLE → generate made-to-order
                     ├─ ③ BOTH legs generate through the same generate_and_rm_group /
                     │    GenerateState semaphore / 16 engines — retro competes for capacity
                     └─ ⑨ returns fresh+retro groups together (RolloutFnTrainOutput)
                                                        ▼
                     trainer GRPO update → weight_version++ → aborts in-flight generations;
                     lag gates (fresh + retro) bound how stale any admitted group can be
```

**The two continuous queues, as pseudocode** (both persist across steps; the per-step call only harvests):

```python
# ========= persistent, cross-step (started once, daemon threads) =========

# QUEUE 1 — fresh lane: AsyncRolloutWorker      agentic_rl/core/fully_async.py
while True:
    # in-flight + completed-but-unshipped ≤ POOL = min(engine_cap, prefetch × batch)  (≈96–128)
    while in_flight + output_q.size < POOL:
        group = data_buffer.get()                  # 8 samples of one prompt
        spawn generate_and_rm_group(group)         # capture tap runs inside each episode:
                                                   #   stage → snapshot → offer TENTATIVE
    completed → output_q                           # ABORTED groups → requeued, never shipped

# QUEUE 2 — retro lane: RetroPrefetchWorker (only if RETRO_PREFETCH_BATCHES > 0)
#                                                agentic_rl/retro/prefetch.py:111
while True:
    if ready_q.size < DEPTH and concurrency_headroom():
        m = pool.lease(event_quota, age ∈ [0,4], newest)   # AVAILABLE → LEASED
        spawn generate_and_rm_group(make_branch_group(m))  # same engines/semaphore as queue 1
    completed → ready_q                            # groups AGE here across weight updates

# ========= per training step t (RolloutManager calls this once) =========
def generate_retro_mixed(args, t):                 # agentic_rl/retro/rollout.py:592
    retro_n, fresh_n = split(32, RETRO_GROUP_RATIO)          # e.g. 8, 24

    fresh_leg:                                     # slime _generate_rollout_async
        while collected < fresh_n:
            g = output_q.pop()                     # ← QUEUE 1
            if lag(g) > 4:    reject_hook(g); reset + requeue   # snapshots → INVALID + delete
            elif not dapo(g): reject_hook(g)
            else:             collect(g)
        accept_hook(each collected g)              # snapshots TENTATIVE → AVAILABLE

    retro_leg (concurrent with fresh_leg):
        taken = drain(ready_q, retro_n)            # ← QUEUE 2: lag gate, quotas, drop aborted
        while taken < retro_n:                     # made-to-order remainder = the DEPTH=0 path
            m = pool.lease(...); g = make_branch_group(m)
            await generate_and_rm_group(g)         # blocking within this step
            gates ok → consume(m) + delete_image   # else release(m) → AVAILABLE retry

    return fresh ∪ retro                           # → trainer → v(t+1) publishes → in-flight aborts
```

With `RETRO_PREFETCH_BATCHES=0` (default) queue 2 doesn't exist and the retro leg is entirely made-to-order inside the step; with depth > 0 both queues run continuously and the step only harvests.

**One prompt's lifecycle** (interactive version: the runbook artifact's *Lifecycle* tab): step t=41 (v42 live) — the pool draws task P → group g of 8 episodes; sibling s₃ stages a candidate at turn 22 of 44 (`cp -a` + ChainCheckpoint), episode ends, the winner becomes a Modal Image + a TENTATIVE manifest carrying the checkpoint and the remaining 53-step budget; g passes lag/DAPO/selection → accept hook → AVAILABLE. Step t=43 (v44 live, policy age 2): the retro leg leases it → 8 branch sandboxes each restore the image into `/app`, wipe the submissions log, take a new AGENT_ID, resume from the masked 22-turn prefix for ≤53 turns under v44 → group passes its gates → CONSUMED + image deleted → trains as one GRPO group (advantages among the 8 branches). Failure paths: parent-group reject → INVALID + delete; branch abort/over-lag → release back to AVAILABLE; aged past 4 → ineligible until the 48 h TTL reaps.

Three corrections to the intuitive mental model: (a) there is **no RetroManager actor and no separate async manager** — slime's `RolloutManager` is the only orchestrator, and retro is library code inside the rollout function it calls (plus one daemon thread); (b) retro does **not locate past rollouts** in storage — capture is a live tap on fresh episodes as they run (the offline-mining path, `phase1.py`, is the dead Phase-1 study); (c) retro does **not send snapshots to the rollout manager for more rollouts** — the retro leg generates the branch rollouts itself through the same engine machinery and hands finished groups back in the same batch.

**Keep (already right):**
- The manifest record + append-only JSONL last-record-wins store — crash-safe, greppable, debuggable with `jq` (buffer.py:50).
- Tentative-until-batch-accept: only trajectories that actually entered training seed the pool, so replay stays tied to the training distribution (rollout.py:365).
- Lease-release-on-reload crash recovery (buffer.py:85); policy-age bounds separate from the behavior-lag gate; hashed event-type assignment.

**Pain points (with the fix each implies):**

| # | Pain | Evidence | Fix |
|---|---|---|---|
| P1 | One env class does both capture and replay, switching on manifest presence, with thread-local episode state | env.py:74, :53 | Split: `CaptureTap` wraps any capturable env; a separate branch runner replays |
| P2 | ✅ FIXED 2026-08-26 (pool.py). ~~Lifecycle transitions written from 4 modules; snapshot deletion decided at ≥2 call sites outside the store~~ | env.py:301; rollout.py:365-383, :321-326; buffer.py; prefetch.py:344 | One `ReplayPool` owns every transition **and** snapshot GC; callers never call `delete_snapshot` |
| P3 | ◐ Leases unified through ReplayPool 2026-08-26; the two generate choreographies (prefetch vs made-to-order) still live in prefetch.py/rollout.py — folding them into one ReplaySource remains | prefetch.py:344 vs rollout.py:305-318 | One `ReplaySource.next_groups(n)`; prefetch depth is an internal detail (depth 0 = synchronous) |
| P4 | Family coupling: selector reads the judge submissions log; capture env subclasses `FrontierCsEnv` | selector.py:16; env.py:47 | `ScoreTrace` protocol — a stream of `(turn, score, ts)` events; harbor adapter = per-step `reward.json` trace |
| P5 | ◐ Pool instantiated per-owner 2026-08-26 (no new globals); prefetch singleton + `_INDEX_BASE` remain | prefetch.py:290, :57; buffer.py:12 | Pool + source instantiated once in the rollout fn's state and passed down; the pool allocates ids |
| P6 | ✅ FIXED 2026-08-26: backends/{modal_snapshot,miniswe_checkpoint}.py behind protocols.py | snapshot.py:206; retro/model.py:9 | `SnapshotBackend` / `AgentCheckpoint` protocols; Modal + mini-swe impls stay project-side |
| P7 | ◐ model.py/snapshot.py renamed into backends/ 2026-08-26 (shims kept); pool.py landed; source.py/mixed.py/capture.py renames wait on P1/P3 | dir listing | Rename by concept: `pool.py`, `source.py`, `mixed.py`, `capture.py`, `checkpoint.py` |
| P8 | ✅ DONE 2026-08-26. The biggest fork file was avoidable: `slime/rollout/fully_async_rollout.py` (+215) was string-loaded via `--rollout-function-path` | §3.10 | Moved to a local copy `core/fully_async.py` (vanilla arms use it too, so not `retro/`); its two CLI flags become `ASYNC_RL_*` knobs. Fork shrinks ~495→~240 lines. The `actor.py` absolute-weight_version fix (+5) cannot move — the lag gate depends on it |
| P9 | Confusing knob names: `ASYNC_RL_RETRO_SNAPSHOT_PATH` is a sandbox-side *source dir* to photograph; `_MANIFEST_PATH` is a volume-side ledger destination | env.py:217; launch_config.py:98 | Rename to `..._SNAPSHOT_SOURCE_DIR`; make `/checkpoints/frontier_retro/<tag>/` the one volume-side retro home. A volume-tarball `SnapshotBackend` (vs Modal Images) becomes a swappable option |

**All-turns capture feasibility (estimate, 2026-08-25).** Goal: keep a snapshot of *every* turn of every kept trajectory so the branch point can be chosen at lease time, not capture time. Two different quantities: the *pipeline need* is only `prefetch_batches × retro_per_step` leased snapshots (e.g. 4 × 28 = 112); the *inventory* is capture-bound: `live ≈ retention_steps × fresh_groups × 8 episodes × T turns` — with age-based GC (delete at policy age > 4) and 4 fresh groups/step, T≈50 → ~8,000 live snapshots. At /app sizes of 1–10 MiB that's 8–80 GiB ≈ $1–7/mo at the volume rate ($0.09/GiB/mo; Modal images aren't separately billed on public pricing today) — **storage cost is a non-issue**. The real constraints: (a) today's GC is consume/invalidate + 48 h TTL only — age-out deletion must be added or inventory grows ~10–20× (TTL-bound, 10⁴–10⁵ image objects); (b) ops, not bytes: `snapshot_directory` costs ~1–5 s per call (logged in `SnapshotResult.latency_seconds`), so T calls/episode adds minutes of wall-clock and ~1,600 image creations/step of registry pressure; (c) consecutive turns are ~ε different but images don't delta-share. **Recommended shape instead: keep ONE image per trajectory and put the turn history inside it** — `git init` the workspace at episode start, commit after every turn (content-addressed deltas ≈ free), snapshot once at episode end with `.git` included; branch-from-turn-t = restore + `git checkout <commit_t>`. That's all-turns choice at today's per-episode image cost, and it slots directly into the `SnapshotBackend` protocol. (Per-turn `ChainCheckpoint`s are the other axis: storing all T is O(T × context) JSON — keep them as a sidecar or in the git repo, not in the manifest JSONL.)

**The one state machine, owned by `ReplayPool`** (statuses unchanged; operations become the API):

```
 capture ──offer(m, batch_key)──▶ TENTATIVE ──commit(batch_key)──▶ AVAILABLE ──lease(filters)──▶ LEASED ──consume()──▶ CONSUMED ─▶ GC
                                     │                                 │                            │
                                     └─abort(batch_key)─▶ INVALID ─▶ GC└─expired/aged─▶ INVALID     └─release() · over-lag · crash-reload─▶ back to AVAILABLE
```

**Lease as a context manager** — the crash-safety idiom becomes impossible to get wrong:

```python
with pool.lease(event_type=quota.next(), max_age=4, order="newest") as lease:
    group = make_branch_group(lease.manifest, k=8)
    out = await generate(group)
    if accept(out):
        lease.consume()        # → CONSUMED, pool GCs the snapshot
    # falling out without consume() auto-releases → back to AVAILABLE
```

**Proposed `retro/` layout** (replaces the current file set):

```
retro/
├── manifest.py     # the record schema — shared vocabulary (keep as-is)
├── pool.py         # ★ ReplayPool + Lease + status machine + JSONL store + snapshot GC   [generic mechanism]
├── source.py       # ★ ReplaySource: yields ready branch groups; prefetch depth internal  [generic mechanism]
├── mixed.py        # ★ MixedRollout composer: fresh + replay legs, split, backfill       [generic mechanism]
├── capture.py      # CaptureTap (wraps any capturable env) + staging + EventSelector wiring
├── selector.py     # event policies over the abstract ScoreTrace                          [family policy]
└── backends/
    ├── modal_snapshot.py      # SnapshotBackend impl (Modal images)                       [infra policy]
    └── miniswe_checkpoint.py  # AgentCheckpoint impl (Chain capture/restore)              [infra policy]
```

**The mechanism/policy split** (all under `retro/`; the left column is what *could* upstream someday — motivation only):

| Generic mechanism (`retro/pool,source,mixed`) | Policy (`retro/capture,selector,backends` + knobs) |
|---|---|
| `MixedRollout` composer (N sample sources + ratios + backfill) | `CaptureTap` + staging + `EventSelector` policies |
| `ReplayPool` + `Lease` (status machine, JSONL store — nothing Modal in it) | `ScoreTrace` adapters (frontier_cs judge submissions; harbor `reward.json`) |
| Related fork delta it builds on: accept/reject hooks, `RolloutFnTrainOutput`, behavior-lag gate + prefetch flags (§3.10) | `ModalSnapshotBackend`, mini-swe `AgentCheckpoint`, launch surface, all `RETRO_*` policy knobs |

Protocols at the boundary (the whole coupling surface): `ScoreTrace` (what "progress" means), `SnapshotBackend` (what a snapshot is), `AgentCheckpoint` (what agent state is). An env is capture-able iff it can provide all three plus a turn callback — that's the `Capturable` contract, and it's what makes retro-on-terminal-bench a per-family adapter rather than a rewrite.

**Migration order** (each step independently shippable; do moves LAST because of contract #10):

1. ✅ **Docs** (2026-08-26): land this runbook; write `SLIME_DELTA.md`; correct the README claim.
2. ✅ **Deletions** (2026-08-26): `async_rl_research/`, dead retro code (#4), legacy evalset path (#8), dead knobs (#10). Fix stale test + CI (#6).
3. ✅ **De-fork the async rollout file (DONE 2026-08-26, P8)**: copied `slime/rollout/fully_async_rollout.py` → `core/fully_async.py`, point `--rollout-function-path` (and retro's imports) at it, convert its two CLI flags to `ASYNC_RL_*` knobs, and revert the slime file + `slime/utils/arguments.py` flags to upstream. Independently shippable; `tests/test_agent/test_behavior_lag.py` guards it.
4. ✅ **Kill the external-repo dependency** (DONE 2026-08-26, #2): in-repo eval entrypoint (`HeldoutEvalSlimeConfig` + `ROLLOUT_MODE=eval` + `post_process_data`); results roll-up script committed (#7). This was the single highest-leverage change for "agent can start without context".
5. ✅ **Knob registry** (DONE 2026-08-26, #3): `agentic_rl/knobs.py` — ~90 knobs with type/default/consumer/scope; launch-time validation with did-you-mean; scan-guarded by `test_knobs.py`. Family-specific knobs keep a family namespace (`FRONTIER_CS_*`).
6. ✅ **Physical moves + retro re-abstraction (both 2026-08-26)** into the layout above — ReplayPool/Lease own every transition + snapshot GC (P2/P5 fixed), both acquisition paths route through the pool (P3), backends/ + protocols.py define the coupling surface (P4/P6 seam), renames done with shims (P7). **Deferred: P1** (splitting env.py's capture/replay dual-mode into a CaptureTap + branch runner — self-contained, do it when adding retro to a second family) — including `verifier_server/ → envs/frontier_cs/judge/` and the `pool/source/mixed/capture` split — updating every string-loaded path (contract #10) in the same commit, with temporary re-export shims (`agentic_rl/generate.py → core/generate.py`) for one deprecation window since old configs and W&B-recorded commands reference the old paths. The retro rewrite is behavior-preserving: same statuses, same JSONL format, same knobs — `tests/test_agent/test_retro.py` + `test_behavior_lag.py` are the harness.
7. ◐ **Onboard Terminal-Bench 2.1, then SWE-Bench Pro** (scaffolded 2026-08-26) — the layering held: both datasets were already published in harbor format (`junlin-modal/terminal-bench-2.1`: 89 tasks; `junlin-modal/swebenchpro`: 731 tasks, both `task_type: harbor` → zero env code), so onboarding = `envs/<family>/` provenance dirs + the one layering fix the runbook predicted (`TRAIN_DATASET` dataset-key knob in launch_config + `envs/datasets.py` registry with deterministic volume-side train/heldout splits — TB 69/20, SWE-Pro 650/81, seed 20260826, sha-pinned in the family READMEs) + smoke arm scripts. **Remaining (Modal):** per family — `::download_data` (pull + materialize split), the harbor oracle check (`python -m agentic_rl.envs.harbor.env … --limit 3`), then the vanilla smoke arm. Retro on these families stays gated at launch until the ScoreTrace adapter + P1 land.
8. **(Deferred — motivation, not a plan.)** If the mechanism half ever proves out and we want it upstream: `MixedRollout` + `ReplayPool`/`Lease` + the fork delta (hooks, `RolloutFnTrainOutput`, behavior-lag gate, prefetch flags) would be the PR, and it would shrink the fork to ~zero. Until then, everything stays under `agentic_rl/retro/` — the clean boundary is its own payoff.
