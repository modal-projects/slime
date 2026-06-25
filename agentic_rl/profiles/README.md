# agentic_rl/profiles

Profiling harnesses for the agentic-RL rollout infrastructure. Results land under
`profiles/<harness>/<run-id>/`.

## `judge_profile.py` — Frontier-CS verifier-server (judge) latency + concurrency

Boots the **production** Frontier-CS judge (the same `vm_runtime` Modal Sandbox that
`environment/verifier_server/autostart.ensure_started()` boots for training — Node +
go-judge, mounting `slime-data` at `/data`, `problemsRoot=/data/frontier_cs/problems`)
and drives it with C++ submissions over its HTTP API to answer:

1. **Per-task oracle eval time** — for each profiled problem, the p50/p90 of the
   judge's verdict latency across 10 uncontended (concurrency=1) submissions of that
   task's *oracle* solution.
2. **Behaviour under load** — fire 50 / 100 / 200 oracle submissions concurrently and
   measure latency (p50/p90/p99), throughput, drain time and error rate as they
   contend for the judge's `nproc` go-judge workers.

### Why "oracle" is calibrated

Only problem `0` ships a verified `reference.cpp`. For every other task the harness
**calibrates** an oracle: it submits the strongest available candidates
(`reference*.cpp` first, then the strongest model solutions under
`Misc/Frontier-CS/algorithmic/solutions/<pid>/`) and keeps the first AC / highest
scoring one, recording the choice and its score. Phase-2 prefers AC oracles so latency
under load reflects judge *queueing*, not solution slowness (real training mixes in
slower TLE submissions, which are worse).

### Run

```bash
cd .../slime
RD=agentic_rl/profiles/frontier_cs_judge/$(date +%Y%m%d-%H%M%S)
export MODAL_ENVIRONMENT=junlin-dev

# 1) boot the production judge (first boot builds the image, ~minutes)
uv run --no-dev python agentic_rl/profiles/judge_profile.py boot  --run-dir $RD
# 2) quick end-to-end sanity on problems 0 & 263
uv run --no-dev python agentic_rl/profiles/judge_profile.py smoke --run-dir $RD
# 3) full run: stratify ~40 problems -> calibrate -> phase1 (10 trials) -> phase2 sweep -> report
uv run --no-dev python agentic_rl/profiles/judge_profile.py run   --run-dir $RD
# 4) free the sandbox
uv run --no-dev python agentic_rl/profiles/judge_profile.py teardown --run-dir $RD
```

Phases are independent and re-runnable (each reads `judge.json` for the live URL and
writes its own artifacts), so a crash never loses prior work. Useful flags:
`--subset-size` (default 40), `--trials` (10), `--concurrency` (`50,100,200`),
`--judge-cpu`/`--judge-mem` (default 8 / 16384, the production values), `--seed`,
`--only <pids>` + `--force` (re-calibrate specific problems).

### Artifacts (per run dir)

| file | contents |
|---|---|
| `judge.json` | judge URL, sandbox id, cpu/mem, boot time, **nproc** (real go-judge worker count), problems visible |
| `stratification.json` | the chosen subset + how it was bucketed |
| `oracles.json` | the calibrated oracle per task (candidate, score, pass/partial) |
| `calibration.jsonl` | every calibration submission |
| `phase1_trials.jsonl` / `phase1_summary.{json,csv}` | every trial + per-task p50/p90 |
| `phase2_requests.jsonl` / `phase2_summary.{json,csv}` | every load submission + per-level aggregates |
| `REPORT.md` | human-readable summary tables |
| `run.log` | full log |

### Notes

- The judge's go-judge `-parallelism` and Node `JUDGE_WORKERS` both default to `nproc`
  inside the sandbox — captured in `judge.json` as the real worker ceiling that
  50/100/200 concurrent submissions queue against.
- Problem metadata + candidate solutions are read from the local mirror
  (`Misc/Frontier-CS/algorithmic/`); the judge grades against slime-data's identical
  problems. The chosen pids are intersected with the judge's `GET /problems`.
