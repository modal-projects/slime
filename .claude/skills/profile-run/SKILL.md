---
name: profile-run
description: Profile/debug a Modal training or eval run for this async-RL project — access Modal app logs, fetch and analyze rollout_*.pt dumps from the slime-checkpoints volume, read the rollout dashboard, and find the W&B run. Use whenever the user reports a bug, shares a log/dashboard link, or asks why a run behaved a certain way.
---

# Profile a training/eval run

How to ground any bug report for this project in **actual observed data** instead of speculation.

## Grounding rule (most important)

When the user reports a bug, shares a log line, or asks "why did X happen":
1. **Pull the real artifact first** — Modal logs, the `rollout_*.pt` dump, and/or W&B — before forming a root-cause claim.
2. **Quote what you actually observe** (counts, token spans, decoded text), not what the code "should" do. This project has burned multiple wrong hypotheses (reasoning-content loss, openai-SDK dropping fields) that the data disproved.
3. If a hypothesis contradicts the data, **say so and revise** — don't defend it.
4. If the needed artifact isn't accessible (e.g. historical logs aged out, W&B entity unknown), **ask the user a specific question** ("what's the app id / W&B entity?") rather than guessing.
5. Verify fixes against the real data path (decode with the real tokenizer / run the actual merge logic), not a synthetic stub.

## Where to look first (efficiency)

Cheapest → most expensive. Don't download a 70–120 MB dump to answer a question W&B already shows.
1. **W&B** (no download): trends over steps — reward/solve (`rollout/raw_reward`), collapse, grad_norm, KL, step_time, cache-hit, eval. Start here for "is it learning / did it collapse / how's throughput."
2. **Modal logs** (stream): live failures and per-sample summaries on the *current* step (tail-only, can't see old steps).
3. **Rollout dump** (download once, reuse): per-sample / token-level forensics — loss-mask spans, turn structure, drift/reset detection, decoded conversations. Only when you need what W&B can't show.

Reuse a single analysis venv across the session (don't reinstall torch/transformers each time); download each dump once.

## Environment facts

- `modal` is **not** on PATH — use **`uvx modal …`** with args **unquoted** (`uvx modal app list --env junlin-dev`, not a single quoted string). Almost everything needs **`--env junlin-dev`**. So-called "modal failures" are nearly always one of: missing `uvx`, missing `--env` (→ empty output or "No such file"), the `app logs` stream auto-disconnecting (~15 min, expected), or arg-quoting — not flaky auth. When correctly invoked it's reliable (verified). First `uvx` call may be slow (downloads modal once, then cached).
- Runs launch from `multinode-training-guide/` via `EXPERIMENT_CONFIG=<cfg> uv run --no-dev modal run -d slime/modal_train.py::train`. Configs live in `multinode-training-guide/slime/configs/`.
- Run tag / W&B group / dump subdir all equal `_RUN_TAG` (default e.g. `qwen3.6-35b-a3b-swe-gym-lite-colocate-1n`). Strip ANSI from CLI output with `sed -E 's/\x1b\[[0-9;]*m//g'`.

## 1. Find the run's app

```bash
cd /Users/junlin/Documents/Research/async-rl/multinode-training-guide
uvx modal app list --env junlin-dev 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' | grep -iE "w_qwen|ephemeral|running"
```
The training run is the `ephemeral` app named after the config (e.g. `w_qwen3_6_…`). Note its `ap-…` id.

## 2. Modal logs

```bash
uvx modal app logs ap-XXXX --env junlin-dev 2>err | sed -E 's/\x1b\[[0-9;]*m//g' > /tmp/logs.txt
```
Caveats (learned the hard way):
- It **streams from ~now**, tail-only — you **cannot scroll back** to an old step's startup. For step-0 history you usually need the dump instead.
- The stream **auto-disconnects ~every 15 min**. For a durable watch, use a Monitor with a self-reconnecting `while true; do uvx modal app logs …; sleep 3; done` loop and a tight `grep` filter (e.g. `adapter_session_empty|aborted:|wall_clock_timeout|\[mini-swe\].*tail:|Traceback`). Don't filter to only the happy path — include failure signatures or a crash looks identical to silence.

Useful greps: `[async_rl] … reward=` (per-sample summaries), `[mini-swe] … exit=N … tail:` (in-sandbox agent failures, nonzero exit only), `[harbor]` (env/grading), `[trajectory] merge prompt base changed` (trajectory drift), `agent budget exhausted before step` (boot/budget).

## 3. Rollout dumps — the main triage tool

Dumps are on the `slime-checkpoints` volume, written per step (config `save_debug_rollout_data`). Train = `rollout_<id>.pt`, eval = `rollout_eval_<id>.pt`. Same `_RUN_TAG` relaunch **overwrites** them.

```bash
uvx modal volume ls slime-checkpoints /swe_rollout_dumps/<RUN_TAG> --env junlin-dev      # list + mtimes
uvx modal volume get slime-checkpoints /swe_rollout_dumps/<RUN_TAG>/rollout_0.pt /tmp/r0.pt --env junlin-dev --force
```
Load (plain dicts — no slime import needed):
```python
import torch
dump = torch.load("/tmp/r0.pt", map_location="cpu", weights_only=False)   # {"rollout_id", "samples":[...]}
s = dump["samples"][0]   # dict: tokens, loss_mask, response_length, response (decoded str),
                         # prompt, rollout_log_probs, metadata{instance_id,is_solved,abort_reason,...},
                         # status, reward, weight_versions, trace
```
Key invariants: `tokens = prompt_ids + response_ids`; `loss_mask`/`rollout_log_probs` align with the **response** portion (last `response_length` tokens). `mask=1` = trained (assistant output), `mask=0` = context (tool results / re-rendered history).

### Analysis recipes (verified)
```python
# trained fraction & turn count
trained = sum(s["loss_mask"]); frac = trained / s["response_length"]
turns = s["response"].count("<|im_start|>assistant") + 1     # +1 for the head turn

# trajectory-merge RESET detector: base task prompt is ~1.0-1.5k tokens; a much larger
# prompt means early turns were dropped into the UNTRAINED prompt (line-107 reset).
prompt_len = len(s["tokens"]) - s["response_length"]
is_reset = prompt_len > 4000

# GRPO signal check: groups with zero reward variance give no advantage
# decode token spans / mask runs with the real tokenizer (see venv below)
```

### Throwaway analysis venv (tokenizer/torch without polluting anything)
```bash
uv venv --python 3.11 /tmp/dbg && \
uv pip install --python /tmp/dbg/bin/python -q transformers jinja2 wandb && \
uv pip install --python /tmp/dbg/bin/python -q torch --index-url https://download.pytorch.org/whl/cpu
# tokenizer-only (Qwen3.6 is public; don't download weights):
/tmp/dbg/bin/python -c "from huggingface_hub import snapshot_download as d; from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained(d('Qwen/Qwen3.6-35B-A3B', allow_patterns=['tokenizer*','*.json']))"
```
`apply_chat_template(..., tokenize=False)` then `tok.encode(...)` to get clean id lists (tokenize=True returns an Encoding in transformers 5.x). The chat template + `reasoning_parser=qwen3` / `tool_call_parser=qwen3_coder` are the source of multi-turn render quirks — see [[trajectory-drift-formaterror]].

## 4. Rollout dashboard (web)

`async_rl_research/dashboard/` serves the same volume. URL pattern:
`https://modal-labs-junlin-dev--swe-rollout-dashboard-dashboard.modal.run/#<RUN_TAG>/<dump>.pt/<sample_idx>`
`convert.py` reconstructs turns by splitting the decoded `response` on `<|im_start|>` and parsing `<think>`/`<tool_call>`/`<tool_response>`. The first (head) turn renders without an opening `<think>` because the prompt prefilled it — that's expected, not a bug.

## 5. W&B (verified — use this FIRST for trends; no big download)

- **Entity `junlinwang`, project `Modal`** (= `WANDB_PROJECT`), run name = group = `_RUN_TAG` (suffix disabled, so one run per relaunch; relaunches make a *new* run with the same name).
- Auth: `wandb` reads `~/.netrc` automatically — no key needed in code (it's already logged in). On a fresh machine: `wandb login` or set `WANDB_API_KEY`. `wandb` isn't installed by default; `uv pip install wandb` into the analysis venv.

```python
import wandb
api = wandb.Api()                                  # loads creds from ~/.netrc
print(api.default_entity)                          # -> junlinwang
runs = list(api.runs("junlinwang/Modal", filters={"group": "<RUN_TAG>"}))
runs.sort(key=lambda r: r.created_at)              # latest = newest relaunch
run = runs[-1]
print(run.id, run.state)                           # e.g. crashed/running/finished
val = run.summary.get("rollout/raw_reward")        # last-logged scalar
h = run.history(keys=["rollout/step","rollout/raw_reward","train/grad_norm","rollout/kl"],
                samples=5000, pandas=True)          # time series (use history, NOT scan_history)
```

**Metric semantics that bite (verified):**
- **`rollout/raw_reward` = the actual mean reward / solve signal** (step 0 ≈ 0.48 matched 122/252 solved in the dump). **`rollout/rewards` is the GRPO-centered advantage ≈ 0 by construction — do NOT use it to judge solve rate.**
- `train/grad_norm`, `train/kl_loss`, `train/ppo_kl`, `perf/step_time`, `sgl_engine/sglang_cache_hit_rate`, `eval/<dataset>/...`. ~193 keys; engine gauges are per-DP-rank means (×8) — see [[swe-rl-perf-profile]].
- `run.history(...)` returns a sampled DataFrame; `_step` is the W&B logging step, use the `rollout/step`/`train/step` columns for the real step. `scan_history(keys=...)` gave all-None here — prefer `history(keys=..., samples=N, pandas=True)`.

W&B alone shows trends without any download: e.g. `rollout/raw_reward` 0.48 → ~0.0 after step 0 is the **policy collapse**, visible instantly.

## Quick interpretation map

- `adapter_session_empty` (0 turns) → in-sandbox agent never completed a turn; check `[mini-swe] … tail:` and `agent budget exhausted before step` (see [[adapter-session-empty-budget-bug]]).
- `exit=-2` → `EXIT_BUDGET_EXCEEDED` (agent ran full `AGENT_TIME_BUDGET_SEC`).
- Large prompt / dropped turns / `merge prompt base changed` → trajectory drift ([[trajectory-drift-formaterror]]).
- `mean turns/sample` collapsing across steps (e.g. 34→1 after one update) → policy collapse; suspect rollout-logprob/EAGLE mismatch, recompute train-side logprobs vs `rollout_log_probs`.
