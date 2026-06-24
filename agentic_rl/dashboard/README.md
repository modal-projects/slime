# SWE rollout-dump dashboard

Browser UI for the rollout debug dumps that training writes to the
`slime-checkpoints` volume (`--save-debug-rollout-data`, see
`configs/w_qwen3_swe_async.py`). Each `rollout_N.pt` is rendered as the agent
conversations it contains — per-group sample chips (solved / reward / aborted),
then a turn-by-turn view of every trajectory: assistant text + `<think>`
blocks, `bash` tool calls, tool outputs with exit codes, plus the sample's
reward/eval metadata and span timings. This is the fast path for seeing what
the agent actually did inside the sandbox.

Eval dumps (`rollout_eval_N.pt`) are first-class: the sidebar splits each run
into train/eval lanes, the eval overview groups samples by dataset instead of
GRPO group, and clicking a run name opens a **run landing page** — a training
curve (solve rate + mean reward per rollout, built from cheap per-dump
summaries) and an eval matrix (step × dataset solve rates), every point/cell
clicking through to the underlying trajectories. Sample pages of eval'd
instances show an "across evals" history strip (solved/reward at every eval
step), built from the same summaries.

Dataset attribution in eval dumps relies on `metadata.eval_dataset`: slime
flattens all eval datasets into one `samples` list when dumping, so each
dataset entry in `eval_config` must stamp itself via
`"metadata_overrides": {"eval_dataset": "<name>"}` (all repo configs do).
Dumps from runs without the stamp render as one `(untagged)` dataset.

**Live app** (Modal env `junlin-dev`):
<https://modal-labs-junlin-dev--swe-rollout-dashboard-dashboard.modal.run>

## Layout

| file | role |
| --- | --- |
| `app.py` | Modal wrapper: bun web server + `slime-checkpoints` volume at `/vol`, volume auto-reload thread (20s) so an in-flight run's new dumps appear |
| `server.ts` | Bun server: static frontend + `/api/runs`, `/api/rollouts`, `/api/rollout` |
| `convert.py` | `.pt` → JSON view-model (runs in-container with CPU torch; parses the Qwen3 chat-template response back into structured turns) |
| `public/` | frontend (vanilla TS, bundled by Bun's HTML imports — no npm deps) |
| `test/smoke.test.ts` | happy-dom render test against a real converted dump |

Converted JSON is cached in the container keyed by file mtime+size, so each
dump converts once (~5 s for a 15 MB dump) and is instant afterwards. The
container scales to zero after 20 min idle; first hit cold-boots it.

## Deploy / develop

```bash
# deploy (volume lives in junlin-dev)
MODAL_ENVIRONMENT=junlin-dev modal deploy async_rl_research/dashboard/app.py

# hot-reload dev against the real volume
MODAL_ENVIRONMENT=junlin-dev modal serve async_rl_research/dashboard/app.py

# run locally against downloaded dumps
mkdir -p /tmp/dump_root/some-run && \
  MODAL_ENVIRONMENT=junlin-dev modal volume get slime-checkpoints \
    /swe_rollout_dumps/rollout_1.pt /tmp/dump_root/some-run/
DUMP_ROOT=/tmp/dump_root PYTHON_BIN=$(command -v python3) bun run dev
# PYTHON_BIN needs torch importable (uv venv + `uv pip install torch` is enough)

# tests (fixture: any convert.py output; defaults to /tmp/api_rollout.json)
python3 convert.py /path/to/rollout_1.pt /tmp/api_rollout.json
bun test test/
```

## Notes

- Dump layout: `swe_rollout_dumps/<run-tag>/rollout_<id>.pt` per launch;
  pre-run-tag dumps sitting at the root show up as the `(root)` pseudo-run.
- The URL is public (anyone with the link). If that ever matters, put
  `requires_proxy_auth=True` on the `@app.function` and hit it with
  `Modal-Key`/`Modal-Secret` headers.
- `convert.py` assumes the Qwen3 chat template (`<|im_start|>` markers). A
  different served model family needs its markers added to `parse_turns`.
