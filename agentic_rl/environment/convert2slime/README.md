# Data pipeline — convert & publish runbook

How raw task datasets become slime train/eval data, and how a run pulls them onto
Modal. **Two principles:**

> **1. One HF dataset repo per dataset.** Each repo is self-contained: `train.jsonl`
> + `eval.jsonl` + one shared `tasks/` tree (+ `problems/` for frontier_cs).
>
> **2. HuggingFace is the source of truth.** Conversion is OFFLINE; each config's
> `download_data()` is a dumb `snapshot_download` into `/data` — no converting,
> no `task_path` rewriting at run time.

The contract that ties it together is the dataset **key** — one string that is the
`/data/<key>/` subdir, the `task_path` prefix (`<key>/tasks/<id>`), and the
registry key. The registry [`configs/datasets.py`](../../../../multinode-training-guide/slime/configs/datasets.py)
maps `key → HF repo`; everything (download path, prompt/eval paths, `eval_config`)
derives from it.

## Per-repo layout

Each repo stores its content nested under `<key>/` and is published with
`path_in_repo=<key>`, so a single `snapshot_download(repo, local_dir=/data)` lands
it at `/data/<key>/` and `task_path=<key>/tasks/<id>` is self-consistent:

```
<key>/train.jsonl          # train datasets only
<key>/eval.jsonl           # held-out; eval-only datasets ship just this
<key>/tasks/<id>/...        # ONE shared task tree, indexed by both jsonls (disjoint ids)
<key>/problems/<pid>/...    # frontier_cs only: judge testdata (verifier server reads it)
```

Slices are **not** committed — `download_data` subsamples `eval.jsonl` in-config
(`subsample`), so a smaller eval is `(key, n)` in the config, not a file in the repo.

## Publish a harbor dataset (swe_gym_lite, usaco, terminal_bench, …)

```bash
# fetch raw harbor tasks (example: swegym-lite from harbor-datasets)
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/harbor-framework/harbor-datasets.git /tmp/hd
git -C /tmp/hd sparse-checkout set datasets/swegym-lite

# convert: bake task_path=<key>/tasks/<id>, split train/eval (or --eval-only)
python -m agentic_rl.environment.convert2slime.harbor \
    --tasks-dir /tmp/hd/datasets/swegym-lite --key swe_gym_lite --eval-n 100 \
    --out-dir /tmp/pub/swe_gym_lite
#   eval-only dataset (no train.jsonl):  --eval-only   (instead of --eval-n)
```
```python
# publish to its OWN repo, nested under the key
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("junlin-modal/swe-gym-lite", repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id="junlin-modal/swe-gym-lite", repo_type="dataset",
                  folder_path="/tmp/pub/swe_gym_lite", path_in_repo="swe_gym_lite")
```

## Publish SWE-rebench-V2 (SWE-bench-style → harbor)

`nebius/SWE-rebench-V2` is SWE-bench-style (not harbor): each row carries a
prebuilt `image_name` + an `install_config` (`test_cmd` + a named `log_parser`)
+ `test_patch` + `FAIL_TO_PASS`/`PASS_TO_PASS`. `swerebench.py` renders each row
into a harbor task dir (the prebuilt `image_name` used **directly** as
`docker_image`, reset+apply test_patch, run `test_cmd`, then a **vendored stdlib
pytest grader** → F2P/P2P resolution) and hands off to `harbor.convert`.

> **Image conventions (verified by oracle check, NOT swebench's).** Nebius's
> published images check the repo out at `/<repo-basename>` (e.g.
> `/scikit-build-core`) — also the image's default `WORKDIR`, so we leave
> `workdir` unset and harbor detects it via `pwd`; scripts never hardcode a cd.
> They are plain **system-python** images (repo pip-installed system-wide, pytest
> on PATH) — NOT `/testbed` + conda. (An earlier `/testbed` Dockerfile-wrapper
> broke `git apply`; using `image_name` directly is also build-free.)

> **Scope: Python only.** The public `SWE-rebench/SWE-bench-fork` registers only
> the Python `parse_log_pytest*` parsers by name; the multilingual registry
> (`parse_log_gotest`, `parse_log_cargo`, …) is not public. `--language` defaults
> to `python` and refuses other values until a matching grader is added.

```bash
# Convert a quality-filtered Python pilot (meta.llm_metadata.code=='A', no issues).
# --scan-limit caps rows examined; --limit caps accepted tasks; --eval-n holds out eval.
python -m agentic_rl.environment.convert2slime.swerebench \
    --out-dir /tmp/pub/swe_rebench_v2 --eval-n 100 --limit 300 --scan-limit 4000
#   widen later: --min-grade none (no quality filter), bigger --limit/--scan-limit.

# PUBLISHED 2026-06-22 (public): the FULL Python set, ALL tasks -> train (no holdout):
#   swerebench ... --out-dir <dir>/swe_rebench_v2 --language python --min-grade none --eval-n 0
# -> 7,243 tasks; live at https://huggingface.co/datasets/junlin-modal/swe-rebench-v2
```
```bash
# ORACLE CHECK before scaling/publishing (runs gold solve.sh → tests on a real
# Modal sandbox / the actual swerebenchv2 image; reward should be 1.0). This is
# the one validation that can't run locally — confirms conda 'testbed' activates
# in the image and the test patch applies. Needs Modal env vars (MODAL_ENVIRONMENT…).
ASYNC_RL_TASK_ROOT=/tmp/pub MODAL_ENVIRONMENT=junlin-dev \
    python -m agentic_rl.environment.harbor \
    /tmp/pub/swe_rebench_v2/train.jsonl --task-root /tmp/pub --limit 3
```
```python
# Publish to its OWN repo, nested under the key (matches configs/datasets.py).
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("junlin-modal/swe-rebench-v2", repo_type="dataset", private=False, exist_ok=True)
# Large tree (~43k files): upload_large_folder is resumable; point it at the PARENT so
# swe_rebench_v2/ lands at the repo root (no path_in_repo arg in upload_large_folder).
api.upload_large_folder(repo_id="junlin-modal/swe-rebench-v2", repo_type="dataset",
                        folder_path="/tmp/pub")  # /tmp/pub contains only swe_rebench_v2/
```

Already registered (`configs/datasets.py`: `swe_rebench_v2 →
junlin-modal/swe-rebench-v2`) and a training config exists
(`configs/w_qwen3_6_swe_rebench_v2_noncolocate_3n.py`, the SWE-Gym 3n recipe
repointed at this dataset).

## Publish frontier-cs (adds `problems/`)

```bash
python -m agentic_rl.environment.convert2slime.frontiercs \
    --tasks-dir /Users/junlin/Documents/Research/multi-agent-autoresearch/tasks/frontier-cs-algorithm \
    --out-dir /tmp/pub/frontier_cs --eval-n 38 --seed 0
cp -r /Users/junlin/Documents/Research/Misc/Frontier-CS/algorithmic/problems /tmp/pub/frontier_cs/problems
```
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("junlin-modal/frontier-cs", repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id="junlin-modal/frontier-cs", repo_type="dataset",
                  folder_path="/tmp/pub/frontier_cs", path_in_repo="frontier_cs")  # incl. the 2.5 GB problems/ (LFS)
```

Register every published dataset in `configs/datasets.py` (`DATASETS[key] = Dataset(repo, has_train=…)`).

## How configs consume it (registry-driven)

Configs declare keys; `download_data` loops; `eval_config` is derived. No paths hardcoded:
```python
from configs.datasets import eval_datasets, pull, subsample, train_path
_TRAIN = "swe_gym_lite"
_EVAL = [("swe_gym_lite", 100), ("usaco", 50)]   # (key, subsample n | None)

prompt_data = train_path(_TRAIN)                 # /data/swe_gym_lite/train.jsonl
eval_config = {"defaults": {...}, "datasets": eval_datasets(_EVAL)}

def download_data(self):
    for k in {_TRAIN, *(k for k, _ in _EVAL)}:
        pull(k)                                  # whole repo -> /data/<key>/
    for k, n in _EVAL:
        if n is not None:
            subsample(k, n)                      # /data/<key>/eval.<n>.0.jsonl
```
`task_path=<key>/tasks/<id>` resolves against `ASYNC_RL_TASK_ROOT=/data` →
`/data/<key>/tasks/<id>`. `HarborEnv` fail-fasts if a resolved task dir is missing.

## Migration / cleanup (your HF token)

1. Publish each dataset to its own repo (above) + register in `configs/datasets.py`.
2. **Migrate the remaining configs** that still import the legacy `HF_EVAL_REPO`
   shim + read `evalsets/v0` (`w_qwen3_6_openthoughts_agent_2n`,
   `w_qwen3_6_usaco_colocate_1n`, and the qwen3-legacy `w_qwen3_swe_eval` /
   `w_qwen3_usaco_1n` / `w_qwen3_swe_async`) to the registry pattern above.
3. Only **after** (2) delete the monolithic repos:
   ```python
   from huggingface_hub import HfApi
   HfApi().delete_repo("junlin-modal/agentic-rl-evalsets", repo_type="dataset")
   HfApi().delete_repo("junlin-modal/agentic-rl-trainsets", repo_type="dataset")
   ```
   (Deleting earlier breaks any config still pointing at them.)

## frontier-cs execution (unchanged by this; see notes_remote_judge_integration.md)

`FrontierCsEnv` boots the verifier server once/worker (mounts slime-data, reads
`/data/frontier_cs/problems`), stages each task's `environment/` files into `/app`,
passes `JUDGE_URL`/`PROBLEM_ID` to the agent leg for `submit.sh`, and grades the
final `solution.cpp` in-sandbox → raw signal → `rewards.py` shaping
(`ASYNC_RL_REWARD_SHAPE`, default `fractional`).
