# agentic_rl — async agentic SWE RL for slime

Fully-async RL on SWE-rebench-V2 coding tasks with the mini-swe-agent harness,
Modal sandboxes for execution, and slime for training. Injected entirely via
slime's hooks — no edits to `slime/`.

## Design (first principles)

The harness **is** the agent loop, so we don't reimplement it. We intercept at
the one thing every harness has — the model API — and let the harness run
unmodified. mini-swe's "model" is a pluggable object, so for a Python harness the
intercept is simply an in-process model that calls sglang and records the exact
prompt/output token ids per call (token-in-token-out — no re-tokenization, no
litellm, no proxy). This is the Polar / Agent Lightning / verl pattern, minus the
HTTP hop (which only a non-Python / out-of-process harness would need).

```
slime fully_async_rollout ──┬─> generate() [one call = one episode]
                            │      ├─ boot Modal sandbox from task.image_name
                            │      ├─ run stock mini-swe-agent (its own loop):
                            │      │     model calls ─→ RecordingModel ─→ sglang   (records exact ids)
                            │      │     bash  calls ─→ sandbox.exec
                            │      ├─ git diff → patch ; grade vs FAIL/PASS_TO_PASS
                            │      └─ prefix-merge recorded calls → Sample(s)
                            └─< trains on the returned Samples
```

| file | role |
| --- | --- |
| `model.py` | mini-swe Model that calls sglang directly and records `(input_ids, output_ids, logprobs, weight_version)` per call |
| `sandbox.py` | Modal sandbox as mini-swe's bash `Environment` |
| `generate.py` | the slime hook: drive one episode (in a worker thread), reconstruct + grade |
| `grade.py` | apply patch + test patch, run tests, parse pytest log (SWE-rebench recipe) |
| `prompts.py` | mini-swe prompts + text protocol constants (pinned) |

Data prep lives with the launcher config (`download_data` writes the prompt_data
jsonl), not here — this package is the generic agent loop. Episode limits
(`agentic_max_steps`, timeouts) come from `args` via slime's `--custom-config-path`.

v1 grades pytest tasks (the launcher filters the dataset to `parse_log_pytest`);
add other frameworks' parsers (SWE-rebench-V2 `lib/agent/log_parsers.py`) to widen.
GLM-4.7-Flash context is ~198k, so generation has headroom; long trajectories are
bounded by `agentic_max_steps` and become large single training micro-batches.

**Token-in-token-out.** Assistant tokens are the exact ids sglang emitted
(`loss_mask=1`); observation tokens come from slicing the next call's prompt
(`loss_mask=0`). Nothing is re-tokenized.

**Reconstruction / compaction.** Calls whose prompts extend the running sequence
merge into one chain (one Sample). A broken prefix — context compaction, a new
sub-agent — starts a new chain (a sibling Sample sharing `rollout_id`).
mini-swe is append-only, so today every episode is one chain → one Sample.

**Off-policy (per-turn versioning).** Each turn carries the `weight_version` live
when it was generated, plus its rollout logprobs — enough for TIS/clipping under
fully-async drift. On a weight update slime aborts in-flight generations; we
discard that episode (fully-async recycles it). Keeping + resuming the partial
trajectory (true partial rollout) is future work and hooks into the same path.

**Failure policy.** An episode that errors mid-run still trains on the turns it
produced (reward 0). An episode with no turns (e.g. sandbox boot failure) or an
aborted turn → `ABORTED` → recycled by fully-async.

## Run (from ~/multinode-training-guide/slime)

```bash
export EXPERIMENT_CONFIG=glm47_flash_agentic_async
modal run slime/modal_train.py::download_data     # SWE-rebench-V2 (Python) → /data
modal run -d slime/modal_train.py::train
```

Episode limits come from env vars (`AGENTIC_MAX_STEPS`, `AGENTIC_EPISODE_TIMEOUT`,
`AGENTIC_EXEC_TIMEOUT`, `AGENTIC_GRADE_TIMEOUT`).
