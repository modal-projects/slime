# agentic_rl — fully-async agentic RL for slime

Fully-async RL on agentic coding/competitive-programming tasks with the
mini-swe-agent harness, Modal sandboxes for execution, and slime for training.
Injected entirely via slime's hooks — **no edits to `slime/`**.

## Design (first principles)

The harness **is** the agent loop, so we don't reimplement it. We intercept at
the one thing every harness has — the model API — and run stock mini-swe-agent
unmodified, **in-process on the head node**. Its "model" is a pluggable object,
so the intercept is an in-process model that calls sglang `/generate` with raw
token ids and records the exact prompt/output ids per call (token-in-token-out —
no re-tokenization, no litellm, no HTTP proxy, no second event loop). Only bash
commands cross into the sandbox.

```
slime fully_async_rollout ──> generate() [one call = one episode, in a worker thread]
   │  env = load_env(metadata.task_type)            # harbor | frontier_cs | swerebench
   │  env.rollout(md, model=RecordingModel, limits):
   │     ├─ boot a bash-only Modal Sandbox from the task image
   │     ├─ run stock mini-swe DefaultAgent(model, sandbox) in-process:
   │     │     model.query ─→ RecordingModel ─→ sglang /generate   (records exact ids + weight_version)
   │     │     bash tool   ─→ sandbox.execute
   │     └─ grade -> reward (env-specific)
   └─ build token-faithful Sample(s) with per-turn weight_versions
```

| file | role |
| --- | --- |
| `model.py` | `RecordingModel`: in-process mini-swe Model; calls sglang `/generate`, records `(tokens, loss_mask, logprobs, weight_version)`, parses native tool-calls, splices the new-context delta (no assistant re-render). |
| `sandbox.py` | Modal sandbox as mini-swe's bash `Environment` + grading executor; Dockerfile build, env injection, vm_runtime, boot retries. |
| `generate.py` | the slime hook: dispatch by `task_type`, run one episode in a thread pool, build Sample(s). |
| `metrics.py` | `agentic/*` episode + `async/*` off-policy-health metrics (`--custom-rollout-log-function-path`). |
| `prompts.py` | mini-swe tool-call scaffold + `BASH_TOOL` + submit sentinel (pinned). |
| `environment/` | the task-family abstraction (see below). |
| `environment/convert2slime/` | dataset → slime prompt-jsonl converters. |

## Environment abstraction

A `RolloutEnv` (`environment/base.py`) owns one task family's whole episode —
row validation, sandbox boot, driving the agent leg(s), and grading — while the
`RecordingModel` + `Sandbox` are the shared tools it composes. Rows pick their
env by `metadata.task_type`:

- **`harbor`** — harbor-format tasks (USACO, SWE-rebench-V2-as-harbor, ...).
  Multi-step episodes, **in-place** grading (`test.sh` → `reward.json`; the
  deliverable is the sandbox state, so there is no patch to transplant), reward
  shaping via `rewards.py`, optional per-step `min_reward` gates.
- **`frontier_cs`** — Frontier-CS competitive programming on top of `harbor`:
  a per-worker Node + go-judge verifier server, judge-env injected into the
  sandbox so the agent's iterative `submit.sh` can self-grade mid-episode.
- **`swerebench`** — SWE-rebench-native single-shot tasks: capture the git diff,
  grade in a **fresh** sandbox (anti reward-hack) with pytest.

`rewards.py` is the one place to design/A-B reward shapes (fractional | binary |
thresholded) without touching envs.

## Fully-async / off-policy

Each turn records the `weight_version` live at generation
(`Sample.weight_versions`), so a multi-turn episode that straddles weight updates
carries enough for TIS/clipping and the `async/version_span|version_lag|
sample_age` metrics. A weight update aborts in-flight generations → the turn's
`finish_reason == "abort"` → `Sample.Status.ABORTED` → slime recycles the
episode. Blocking episodes are offloaded to a wide dedicated thread pool (sized
to `sglang_server_concurrency × num_engines`), since `asyncio.to_thread` caps at
~32.

## Qwen

Native bash tool-calls (set `--sglang-tool-call-parser qwen3_coder
--sglang-reasoning-parser qwen3`). Because the recorder keeps one running token
sequence and only ever renders the *new observation* delta — never re-rendering a
prior assistant turn — Qwen's `<think>` history-stripping and the tool-call-arg
whitespace rstrip cannot drift the prompt from the training target. The old
chat-completions adapter's `_dictify_tool_arguments` fix is therefore obsolete
here. `agentic_max_empty_turns` / `finish_reason == "length"` end an episode
cleanly instead of format-erroring to the step limit.

## Run

```
--custom-generate-function-path    agentic_rl.generate.generate
--custom-rollout-log-function-path agentic_rl.metrics.log_rollout_data
--custom-config-path               agentic_rl/config_example.yaml
--sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3
```

with `train_async.py` + `fully_async_rollout` (non-colocated). `ASYNC_RL_TASK_ROOT`
points at the dir that `metadata.task_path`s resolve against (harbor/frontier_cs).

Oracle check (reference solution through the exact rollout path, reward → 1.0):

```
python -m agentic_rl.environment.harbor out/usaco.jsonl --task-root out --limit 3
```
