# async_rl_research

Agentic-RL rollout package for slime. It runs an in-sandbox coding agent
(default: mini-swe-agent) against tasks on Modal, records exact SGLang tokens
via an HTTP adapter, and grades the result into a reward. Task families are
pluggable **envs**: SWE-Gym (git-diff grading in a clean sandbox) and harbor
datasets like USACO (in-place `test.sh` verification, multi-step aware).

| Module | Role |
| --- | --- |
| `generate.py` | Per-sample rollout entrypoint (`--custom-generate-function-path async_rl_research.generate.generate`); orchestrates `runtime × env` |
| `agent/base.py` | `AgentRuntime` contract + shared launch/provision machinery + runtime registry |
| `agent/mini_swe_agent.py` | Default runtime (`mini-swe`): adapter choice, venv provisioning, headless runner |
| `env/base.py` | `RolloutEnv` contract (row schema, sandbox lifecycle, grading) + env registry; rows pick their env via `metadata.task_type` |
| `env/swe_gym.py` | SWE-Gym env: prebuilt image boot / pre_commands / git diff / clean-sandbox eval |
| `env/harbor.py` | Harbor env: Dockerfile boot, step loop, in-place verify (+ oracle-check CLI) |
| `env/convert2slime/` | Dataset converters, paired with their env by filename (see `data/README.md`) |
| `evalset.py` | Eval-set builder: spec YAML → subsampled per-subset jsonl + manifest + ready `--eval-config` (see `data/README.md`) |
| `modal_sandbox.py` | Modal backend (boot concurrency, create retry; registry refs + Dockerfile builds) |
| `dashboard/` | Modal web app (Bun/TS) for browsing the rollout debug dumps as agent conversations (see `dashboard/README.md`) |
| `profiles/PERF.md` | Measured rollout-time attribution, ranked fixes, and a step-by-step profiling guide |
| `profiles/profiling.py` | In-rollout instrumentation: env phase timers + adapter middleware (per-session turn count / gen time) → `sample.metadata["timing"]` → dumps |
| `profiles/profile.py` | Offline analyzer: W&B run + rollout dump → one attribution row in `profiles/runs.jsonl` + regenerated `profiles/ATTRIBUTION.md` |

## Setup

Harbor datasets need two things at rollout time: `ASYNC_RL_TASK_ROOT` pointing at
the converter's out dir (on the slime-data volume), and ideally an oracle pass
first (`python -m async_rl_research.environment.harbor <jsonl> --limit 3`, expect
reward=1.0) -- see `data/README.md` for the full flow.

The rollout boot honors these env vars:

| Env var | Purpose |
| --- | --- |
| `MODAL_REGISTRY_SECRET` | Modal secret for authenticated Docker Hub pulls (`dockerhub-creds`) |
| `MODAL_ENVIRONMENT` | Modal environment the images are cached in |
| `SLIME_AGENT_SANDBOX_ADD_PYTHON` | Add python to the image (must match rollout) |

## Eval

Eval reuses the exact same `generate()` → `runtime × env` stack as training:
slime's eval path (`slime/rollout/sglang_rollout.py::eval_rollout`) iterates
`--eval-config` datasets and calls the custom generate function with
`evaluation=True` per sample. Mean reward per dataset lands in W&B as
`eval/{name}` (plus `-truncated_ratio`, response-len stats).

Three pieces, in order:

1. **Build an eval set** (subsampled, versioned, pinned by manifest):
   `python -m async_rl_research.evalset spec.yaml --out-dir /data/evalsets/v0`
   — see `data/README.md`. Oracle-check harbor subsets before burning GPU time.
2. **Wire it into the training config** as an inline `eval_config` dict (the
   launcher materializes it to a temp YAML → `--eval-config`); set
   `eval_interval`. train_async.py evals every `eval_interval` rollouts
   (first at rollout `eval_interval` — no step-0 baseline in async mode; get
   the base-model baseline from an eval-only run instead). Eval blocks the
   train loop and shares the sglang engines — size subsets accordingly.
3. **Eval-only runs**: `num_rollout = 0` with `eval_interval` set routes
   through `train.py`'s stock eval-only branch — one eval pass, then exit
   (set `load` to a saved checkpoint to eval a trained model; use
   `async_mode = False` in the experiment config so train.py is the
   entrypoint).

Per-dataset eval sampling overrides (`temperature`, `top_p`, `top_k`) flow
through `generate.py::_sampling_params` into the adapter's session defaults;
per-turn `max_new_tokens` stays adapter-governed regardless of
`eval_max_response_len`.
