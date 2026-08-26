# Verifier-server integration: Frontier-CS (algorithmic) + Dr. Kernel (KernelGYM)

Design note for two train+eval dataset families graded by a **separate server the
rollout queries during the episode** (not a self-contained in-sandbox grader).
Frontier-CS = competitive programming (CPU judge). Dr. Kernel = Triton kernel
generation (GPU judge, future).

Status: **Frontier-CS implemented** (harbor-execution model, see below). Dr. Kernel
is still planning; it reuses the same verifier-server pattern.

## What changed vs the original (head-side) design

The first design graded **head-side** (read `solution.cpp` out, POST from the head)
with a standalone `judges/` package and a single blind grade. The implemented
design instead makes Frontier-CS a **harbor task family with in-sandbox grading**:

- Tasks are full harbor tasks (per-task `Dockerfile` + `tests/evaluate.py`), sourced
  from `multi-agent-autoresearch/tasks/frontier-cs-algorithm`, run by
  `FrontierCsEnv(HarborEnv)` — not a standalone `RolloutEnv`.
- Grading happens **in the sandbox** (`tests/evaluate.py` POSTs `solution.cpp` to a
  verifier server), like harbor's `test.sh` → `reward.json`. The head doesn't grade.
- The **iterative `submit.sh` loop is kept**: the agent self-grades mid-episode
  against the same server (dense feedback). The **final** `solution.cpp` sets the
  training reward (not best-of-submissions).
- The old `judges/` package is **deleted**; everything lives under `environment/`.

## 0. Why a separate verifier server (still true)

1. **Frontier-CS testdata is 2.5 GB** (`Misc/Frontier-CS/algorithmic/problems/`,
   188 problems; pid 61 alone is 445 MB) — can't bundle per rollout. It lives once
   on the server's Volume; the rollout sends only `pid` + `solution.cpp`.
2. **74/188 problems are interactive** (bidirectional pipe to an `interactor`). The
   vendored Node+go-judge already handles these; rewriting in-sandbox is the hard part.
3. **go-judge needs a real kernel** (namespaces/cgroups), which gVisor can't grant →
   the server runs under `vm_runtime`.
4. (Kernel) grading needs a **GPU**; the agent sandbox is CPU.

---

## 1. Frontier-CS architecture (implemented)

```
   rollout worker (head)                        environment/verifier_server/
        │ FrontierCsEnv.rollout()                 autostart.ensure_started() ──┐ boots once/worker
        ▼                                                                      ▼
 ┌───────────────────────────┐    submit.sh / evaluate.py POST    ┌──────────────────────────┐
 │ per-task harbor sandbox    │ ────────────────────────────────► │ verifier server          │
 │  • mini-swe agent          │  {pid, code} → score (0-100)       │ Node + go-judge :8081    │
 │  • /app/solution.cpp       │ ◄──────────────────────────────── │ vm_runtime Modal Sandbox │
 │  • iterative submit.sh      │                                    │ slime-data Vol (/data)   │
 └───────────────────────────┘                                    └──────────────────────────┘
        │ in-sandbox tests/evaluate.py → /logs/verifier/reward.json (RAW signal)
        ▼
   HarborEnv._verify → rewards.signal_from_reward_dict → rewards.shape(ASYNC_RL_REWARD_SHAPE)
        ▼  RewardResult(reward, is_solved, extra)
```

### Components
- **`environment/frontiercs.py`** — `FrontierCsEnv(HarborEnv)`. On first rollout boots
  the verifier server (`ensure_started()`), stages each task's `environment/` files
  (`statement.txt`, `submit.sh`, `submit.py`, `AGENT.md`, `config.yaml`) into `/app`
  (`_pre_agent_setup`), and passes `JUDGE_URL`/`PROBLEM_ID` to the agent leg
  (`agent_extra_env`) so `submit.sh` reaches the judge. Everything else inherited.
- **`environment/verifier_server/`** — `server/` (vendored go-judge, == a trimmed
  `Misc/Frontier-CS/algorithmic`) + `autostart.py` (`ensure_started()` boots a
  `vm_runtime` Modal Sandbox once/worker, mounts **`slime-data` at `/data`** and points
  the go-judge's `problemsRoot` at `/data/frontier_cs/problems` via `PROBLEMS_ROOT`,
  waits `/health`, exports `FRONTIER_CS_JUDGE_URL`). One server per worker serves all
  problems (caller passes `PROBLEM_ID`); both `submit.py` and `evaluate.py` POST to it.
  No separate problems Volume / populate step — the testdata rides on `slime-data`.
- **`environment/harbor.py`** — generic; two off-by-default hooks added for this:
  `_pre_agent_setup` (stage runtime files) and central reward shaping in `_verify`
  (build `RewardSignal` from `reward.json`, call `rewards.shape`). swe-gym unchanged.
- **`environment/rewards.py`** — central shaper. Verifier emits a RAW signal
  (`score_raw` 0-100, `cases_passed/total`, `is_solved`); `shape()` maps it to a
  scalar via `fractional` (default) / `binary` / `thresholded`, selected by
  `ASYNC_RL_REWARD_SHAPE` (global) or `metadata.reward_shape` (per-row). Legacy
  harbor `{"reward": x}` still reproduces today's behavior.

### Judge contract (`server/judge/src/router.js`)
- `POST /submit` — multipart `code` + form `{pid, lang:"cpp"}` → `{sid}`.
- `GET /result/:sid` → `{status:"queued"|"done"|"error", passed, score(0-100),
  scoreUnbounded, cases:[...]}`; poll until `done`/`error` (`404` = not ready).
- `GET /health`, `GET /problems` (readiness). reward = `score/100` (partial credit).

### Networking (changed from v1)
We **do** use sandbox→judge networking now (both `submit.sh` and `evaluate.py` POST
from inside the sandbox). The agent sandbox already has outbound internet — it dials
the model adapter over a `modal.forward` tunnel — so it reaches the verifier server's
tunnel URL with no extra config. `FRONTIER_CS_JUDGE_URL` (if pre-set) skips the boot.

---

## 2. Data pipeline (implemented; see `convert2slime/README.md`)

HF is the single source of truth; conversion is offline, `download_data` is a dumb pull.

- **Convert** (offline): `convert2slime/frontiercs.py` reads the autoresearch harbor
  tasks, preps them for slime (strip docker-compose; template
  `JUDGE_URL=${FRONTIER_CS_JUDGE_URL}`; clean `[task].name`; swap in the canonical
  slime `tests/evaluate.py` that grades the FINAL `solution.cpp` and emits the raw
  signal), runs the harbor converter, sets `task_type=frontier_cs`, re-roots
  `task_path → frontier_cs/tasks/<id>`, splits 38 eval / rest train.
- **Publish** to HF `junlin-modal/agentic-rl-trainsets`:
  `frontier_cs/{frontier_cs_train.jsonl, frontier_cs_eval.jsonl, tasks/<id>/,
  problems/<pid>/}` (problems = the 2.5 GB testdata).
- **Pull**: the config's `download_data()` snapshot_downloads `frontier_cs/*.jsonl` +
  `frontier_cs/tasks/**` + `frontier_cs/problems/**` (the 2.5 GB) all to `/data` on
  `slime-data`. The verifier server mounts `slime-data` and reads
  `/data/frontier_cs/problems` — one volume, one download, no separate populate step.

---

## 3. Code structure

```
async_rl_research/environment/
  base.py                       # RolloutEnv, RewardResult, load_env; ENVS["frontier_cs"]
  harbor.py                     # HarborEnv (generic; _pre_agent_setup + central shaping hooks)
  frontiercs.py                 # FrontierCsEnv(HarborEnv): verifier-server glue + submit-loop env
  rewards.py                    # central reward shapers (fractional|binary|thresholded)
  verifier_server/
    autostart.py                # ensure_started() → vm_runtime judge sandbox; mounts slime-data, PROBLEMS_ROOT
    server/                     # vendored Node + go-judge (problemsRoot is env-configurable)
  convert2slime/
    harbor.py                   # generic harbor → slime converter
    frontiercs.py               # frontier-cs: prep + harbor convert + task_type + split
    README.md                   # publish/download runbook (single source of truth)

multinode-training-guide/slime/configs/
  w_qwen3_6_frontier_cs_colocate_1n.py   # train; pulls from agentic-rl-trainsets
  w_qwen3_6_frontier_cs_eval.py          # eval-only (num_rollout=0)
```

`generate.py` is task-agnostic (the env owns the verifier-server boot). The non-judge
rollout path (swe-gym/usaco) is unchanged.

---

## 4. Dr. Kernel (later; reuses the verifier-server pattern)

Same idea, different server + rollout shape:
- **Server** — reuse `Misc/KernelGYM/kernelgym/` (FastAPI `:10907`, Redis queue,
  subprocess-isolated per-GPU workers), deployed as a Modal **GPU** verifier server.
- **Rollout** — `environment/kernel.py`, no sandbox; multi-turn generate↔grade
  (`MAX_TURN≈3`): emit `ModelNew`, POST `{reference_code, kernel_code, ...}`, feed
  `{compiled, correctness, speedup}` back; reward = port of
  `kernel_reward.calculate_reward_speedup` (+ decoy/coverage guards) via `rewards.py`.
- **Converter** — `convert2slime/kernel.py` from `hkust-nlp/drkernel-rl-data`.

The kernel server isn't queried from a sandbox (there is none) — the env POSTs from
the head per turn. So it doesn't need the sandbox→judge networking path; it does need
a GPU verifier server (separate from training GPUs).

---

## 5. Open decisions / risks
1. **go-judge under `vm_runtime`** — verify junlin-dev is allowlisted + go-judge boots
   (smoke test: boot, POST a reference `.cpp`, expect a score). Fallback: a plain VM via
   `algorithmic/sky-judge.yaml`, set `FRONTIER_CS_JUDGE_URL`.
2. **Reward hacking via the submit loop** — the agent can POST to the judge mid-episode
   and see its score on the HIDDEN set. We kept this deliberately (dense feedback) and
   mitigate by grading the **final** `solution.cpp` (not best-of), but submit-spam /
   overfitting to the graded set is a real surface to watch. (The autoresearch
   `evaluate.py`'s best-of behavior was dropped for this reason.)
3. **Throughput** — one verifier server per worker under many concurrent rollouts;
   size `JUDGE_WORKERS`/`GJ_PARALLELISM` (≈ nproc). Grade ≈ seconds–minutes
   (TLE-bound by per-case time × n_cases).
4. **Reward-shape ablations** — `ASYNC_RL_REWARD_SHAPE` is the seam; default `fractional`
   (dense, avoids GRPO group collapse on hard problems); `binary`/`thresholded` available.
5. **Eval comparability** — `score_raw/100` (mean per-case ratio) matches the judge's
   native `finalScore`, so eval lines up with the published bench.
6. **(Kernel)** GPU budget for grading separate from training; Dr. Kernel's TRLOO
   advantage trick is out of scope for data integration.
```
