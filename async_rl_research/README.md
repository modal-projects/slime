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
| `modal_sandbox.py` | Modal backend (boot concurrency, create retry; registry refs + Dockerfile builds) |
| `dashboard/` | Modal web app (Bun/TS) for browsing the rollout debug dumps as agent conversations (see `dashboard/README.md`) |

## Setup

Harbor datasets need two things at rollout time: `ASYNC_RL_TASK_ROOT` pointing at
the converter's out dir (on the slime-data volume), and ideally an oracle pass
first (`python -m async_rl_research.env.harbor <jsonl> --limit 3`, expect
reward=1.0) -- see `data/README.md` for the full flow.

The rollout boot honors these env vars:

| Env var | Purpose |
| --- | --- |
| `MODAL_REGISTRY_SECRET` | Modal secret for authenticated Docker Hub pulls (`dockerhub-creds`) |
| `MODAL_ENVIRONMENT` | Modal environment the images are cached in |
| `SLIME_AGENT_SANDBOX_ADD_PYTHON` | Add python to the image (must match rollout) |
