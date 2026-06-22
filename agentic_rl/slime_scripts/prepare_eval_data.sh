#!/usr/bin/env bash
# Build the agentic-RL eval set and publish it to HuggingFace — the PRODUCER
# pipeline. Consumers run qwen3_swe_eval.sh, which pulls the published dataset
# onto the volume (modal_train.py::download_data) and launches the eval.
#
#   ./prepare_eval_data.sh   # maintainer; needs MODAL_HF_TOKEN in ~/.global_env
#
# Steps: clone convert screen evalset oracle publish
#
# Publishes to https://huggingface.co/datasets/junlin-modal/agentic-rl-evalsets
# (public). The repo mirrors the slime-data volume root (evalsets/v0/*.jsonl +
# exactly the harbor task dirs those subsets reference), so after download every
# metadata.task_path resolves under ASYNC_RL_TASK_ROOT=/data. Bump v0 -> v1 here
# and in configs/w_qwen3_swe_eval.py to change the eval set -- never mutate a
# version a past run used.
set -euo pipefail

SLIME=${SLIME:-$HOME/Documents/Research/async-rl/slime}
GUIDE=${GUIDE:-$HOME/Documents/Research/async-rl/multinode-training-guide}
STAGE=${STAGE:-$HOME/slime-data-stage}   # local mirror of slime-data volume root (/data)
# Which published version to (re)build. Bump here AND set EVAL_VERSION in the
# consumer configs (w_qwen3_swe_eval.py / w_qwen3_6_swe_eval.py) to switch the
# eval set. Both versions coexist on HF; never mutate a version a past run used.
#   v0 = swe_gym_lite_100 + swebench_verified_100 + openthoughts_tblite(92) + usaco_50
#   v1 = swebench_verified(500) + swebenchpro(731) + swebench_multilingual(300)
#        + terminal_bench(89, Terminal-Bench 2.1)
EVALSET=${EVALSET:-evalsets/v1}
VERSION=$(basename "$EVALSET")
HF_REPO=${HF_REPO:-junlin-modal/agentic-rl-evalsets}
# Terminal-Bench 2.1 isn't in the harbor-datasets git repo; it's a hub dataset
# pulled with `harbor datasets download <slug>` (export layout:
# <dir>/<dataset-name>/<task>/), then converted via --tasks-dir like the others.
TBENCH_SLUG=${TBENCH_SLUG:-terminal-bench/terminal-bench-2-1}
# Default skips `screen` (the ~45-min, 100-sandbox oracle screen): tblite
# selection now uses an explicit broken-task exclude-list (BROKEN_TBLITE_IDS /
# the `include` step) instead of "reference passes on gVisor", which dropped
# 50/100 though only 8 are truly broken. Run STEPS=screen on demand.
STEPS=${STEPS:-clone,convert,include,evalset,oracle,publish}

# openthoughts-tblite tasks genuinely unrunnable on Modal/gVisor (not just
# oracle-flaky); the other 92/100 are kept. Edit this list — NOT the screen —
# to change tblite membership.
BROKEN_TBLITE_IDS=(
  openthoughts-tblite__acl-permissions-inheritance
  openthoughts-tblite__california-housing-api
  openthoughts-tblite__container-registry-optimization
  openthoughts-tblite__etl_checkpoint_resume_bug
  openthoughts-tblite__log-summary
  openthoughts-tblite__malicious-package-forensics
  openthoughts-tblite__security-breach-incident-response
  openthoughts-tblite__security-incident-log-analysis
)

export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-junlin-dev}

# Per-version dataset wiring. SPARSE = harbor-datasets dirs to sparse-checkout;
# LOCAL = those converted from the local clone via --tasks-dir; TBENCH=1 also
# converts terminal-bench from the harbor registry (not in the git repo).
case "$VERSION" in
  v0)
    SPARSE_DATASETS=(datasets/swegym-lite datasets/swebench-verified datasets/openthoughts-tblite datasets/usaco)
    LOCAL_DATASETS=(swegym-lite swebench-verified openthoughts-tblite usaco)
    ORACLE_SUBSETS=(swe_gym_lite_100 swebench_verified_100 openthoughts_tblite usaco_50)
    TBENCH=0
    ;;
  v1)
    SPARSE_DATASETS=(datasets/swebench-verified datasets/swebenchpro datasets/swebench_multilingual)
    LOCAL_DATASETS=(swebench-verified swebenchpro swebench_multilingual)
    ORACLE_SUBSETS=(swebench_verified swebenchpro swebench_multilingual terminal_bench)
    TBENCH=1
    ;;
  *) echo "unknown EVALSET version '$VERSION' (expected v0|v1)" >&2; exit 1 ;;
esac

has_step() { [[ ",$STEPS," == *",$1,"* ]]; }
py() { (cd "$SLIME" && PYTHONPATH="$SLIME" uv run --no-project --with modal,pyyaml python "$@"); }

# ── 1. Sparse-clone the source datasets from harbor-datasets ──────────────────
if has_step clone; then
  mkdir -p "$STAGE"
  if [[ ! -d "$STAGE/harbor-datasets/.git" ]]; then
    git clone --filter=blob:none --sparse --depth 1 \
      https://github.com/harbor-framework/harbor-datasets.git "$STAGE/harbor-datasets"
  fi
  git -C "$STAGE/harbor-datasets" sparse-checkout set "${SPARSE_DATASETS[@]}"
fi

# ── 2. Convert each harbor task tree to slime prompt data + task dirs ─────────
if has_step convert; then
  for ds in "${LOCAL_DATASETS[@]}"; do
    name=${ds//-/_}
    py -m agentic_rl.environment.convert2slime.harbor \
      --tasks-dir "$STAGE/harbor-datasets/datasets/$ds" \
      --out-dir "$STAGE/$name" --name "$name"
  done
  # Terminal-Bench 2.1: download from the harbor hub (needs `harbor` in the env;
  # py() only has modal,pyyaml), then convert the exported task tree via
  # --tasks-dir. Export layout is <out>/<dataset-name>/<task>/, so point the
  # converter at the single dataset subdir.
  if [[ "$TBENCH" == 1 ]]; then
    tb_dl="$STAGE/terminal_bench_download"
    rm -rf "$tb_dl"
    uv run --no-project --with harbor harbor datasets download "$TBENCH_SLUG" \
      --export --overwrite -o "$tb_dl"
    tb_src=$(find "$tb_dl" -mindepth 1 -maxdepth 1 -type d | head -1)
    py -m agentic_rl.environment.convert2slime.harbor \
      --tasks-dir "$tb_src" --out-dir "$STAGE/terminal_bench" --name terminal_bench
  fi
fi

# ── 3. Screen openthoughts-tblite against the sandbox backend ────────────────
# tb-style system tasks can be impossible on Modal/gVisor regardless of model
# (e.g. setfacl: Operation not supported). Run every reference solution and keep
# only those that pass, so eval measures the model not the infra. ~100
# sandboxes, 10-way parallel, ~30-45 min.
if has_step screen; then
  src="$STAGE/openthoughts_tblite/openthoughts_tblite.jsonl"
  log="$STAGE/openthoughts_tblite/oracle_screen.log"
  n=$(wc -l < "$src" | tr -d ' ')
  : > "$log"
  export SLIME STAGE  # for the xargs subshells (MODAL_ENVIRONMENT already exported)
  # NB: the SOURCE jsonl's task_paths are relative to the converter out-dir
  # (the evalset builder re-roots them to $STAGE later).
  seq 0 $((n - 1)) | xargs -P 10 -n 1 bash -c '
    cd "$SLIME" && PYTHONPATH="$SLIME" \
    uv run --no-project --with modal,pyyaml python -m agentic_rl.environment.harbor \
      "$STAGE/openthoughts_tblite/openthoughts_tblite.jsonl" --index "$0" \
      --task-root "$STAGE/openthoughts_tblite" --solve-timeout 900
  ' >> "$log" 2>&1 || true
  grep -E '^\[OK \]' "$log" | sed -E 's/^\[OK \] ([^:]+):.*/\1/' | sort -u \
    > "$STAGE/openthoughts_tblite/oracle_pass_ids.txt"
  python3 - "$src" "$STAGE/openthoughts_tblite/oracle_pass_ids.txt" <<'PYEOF'
import json, sys
src, ids_path = sys.argv[1], sys.argv[2]
ids = set(open(ids_path).read().split())
out = src.replace(".jsonl", "_oraclepass.jsonl")
total = kept = 0
with open(out, "w") as f:
    for line in open(src):
        total += 1
        if json.loads(line)["metadata"]["instance_id"] in ids:
            f.write(line); kept += 1
print(f"oracle screen: kept {kept}/{total} -> {out}")
PYEOF
fi

# ── 3b. Build the tblite "included" set: converted 100 minus BROKEN_TBLITE_IDS ─
# Replaces the oracle screen for selection — keeps every task not on the broken
# list (92/100). Reads the freshly converted jsonl, so the output inherits the
# current prompt schema.
if has_step include && [[ "$VERSION" == v0 ]]; then
  src="$STAGE/openthoughts_tblite/openthoughts_tblite.jsonl"
  out="$STAGE/openthoughts_tblite/openthoughts_tblite_included.jsonl"
  printf '%s\n' "${BROKEN_TBLITE_IDS[@]}" > "$STAGE/openthoughts_tblite/broken_ids.txt"
  python3 - "$src" "$STAGE/openthoughts_tblite/broken_ids.txt" "$out" <<'PYEOF'
import json, sys
src, broken_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
broken = set(open(broken_path).read().split())
total = kept = 0
with open(out, "w") as f:
    for line in open(src):
        total += 1
        if json.loads(line)["metadata"]["instance_id"] not in broken:
            f.write(line); kept += 1
print(f"tblite include: kept {kept}/{total} (excluded {len(broken)} broken) -> {out}")
PYEOF
fi

# ── 4. Build the versioned eval set (deterministic subsamples) ────────────────
# v1 subsets keep every converted row (no `n:` cap) — full held-out benchmarks.
if has_step evalset; then
  spec="$STAGE/evalset_${VERSION}_spec.yaml"
  case "$VERSION" in
    v0)
      cat > "$spec" <<EOF
task_root: $STAGE
subsets:
  - name: swe_gym_lite_100
    source: $STAGE/swegym_lite/swegym_lite.jsonl
    n: 100
    seed: 0
  - name: swebench_verified_100
    source: $STAGE/swebench_verified/swebench_verified.jsonl
    n: 100
    seed: 0
  - name: openthoughts_tblite
    source: $STAGE/openthoughts_tblite/openthoughts_tblite_included.jsonl
    n: 92
    seed: 0
  - name: usaco_50
    source: $STAGE/usaco/usaco.jsonl
    n: 50
    seed: 0
EOF
      ;;
    v1)
      cat > "$spec" <<EOF
task_root: $STAGE
subsets:
  - name: swebench_verified
    source: $STAGE/swebench_verified/swebench_verified.jsonl
  - name: swebenchpro
    source: $STAGE/swebenchpro/swebenchpro.jsonl
  - name: swebench_multilingual
    source: $STAGE/swebench_multilingual/swebench_multilingual.jsonl
  - name: terminal_bench
    source: $STAGE/terminal_bench/terminal_bench.jsonl
EOF
      ;;
  esac
  py -m agentic_rl.evalset "$spec" --out-dir "$STAGE/$EVALSET"
fi

# ── 5. Oracle-check the subsets (reference solutions; expect reward=1.0) ──────
# Plumbing check, not a model run: boots Modal sandboxes. Do NOT publish if it fails.
if has_step oracle; then
  for subset in "${ORACLE_SUBSETS[@]}"; do
    echo "── oracle: $subset"
    ASYNC_RL_TASK_ROOT="$STAGE" py -m agentic_rl.environment.harbor \
      "$STAGE/$EVALSET/$subset.jsonl" --limit 2
  done
fi

# ── 6. Publish to HuggingFace (public) ────────────────────────────────────────
# Bundle = eval-set files + exactly the task dirs they reference, laid out as
# the data root, so a download is a verbatim mirror onto the volume.
if has_step publish; then
  set -a; source "$HOME/.global_env"; set +a
  : "${MODAL_HF_TOKEN:?MODAL_HF_TOKEN not found in ~/.global_env}"
  STAGE="$STAGE" EVALSET="$EVALSET" HF_REPO="$HF_REPO" HF_TOKEN="$MODAL_HF_TOKEN" \
  uv run --no-project --with huggingface_hub python - <<'PYEOF'
import json, os, shutil, sys
from pathlib import Path

stage, ver, repo = Path(os.environ["STAGE"]), os.environ["EVALSET"], os.environ["HF_REPO"]
bundle = stage / "hf_bundle"
shutil.rmtree(bundle, ignore_errors=True)
(bundle / ver).mkdir(parents=True)

task_paths = set()
for f in sorted((stage / ver).iterdir()):
    shutil.copy2(f, bundle / ver / f.name)
    if f.suffix == ".jsonl":
        for line in f.read_text().splitlines():
            md = json.loads(line).get("metadata") or {}
            if md.get("task_type") == "harbor" and md.get("task_path"):
                task_paths.add(md["task_path"])

for tp in sorted(task_paths):
    src = stage / tp
    if not src.is_dir():
        sys.exit(f"referenced task dir missing locally: {src}")
    shutil.copytree(src, bundle / tp, dirs_exist_ok=True)

(bundle / "README.md").write_text(f"""# agentic-rl-evalsets

Versioned eval sets for the slime agentic-RL setup (mini-swe-agent on Modal
sandboxes). Built from [harbor-framework/harbor-datasets](https://github.com/harbor-framework/harbor-datasets)
by `multinode-training-guide/scripts_agenticRL/prepare_eval_data.sh`:
converted to slime prompt rows, deterministically subsampled (seeds pinned in
`{ver}/manifest.json`), and oracle-checked (reference solutions pass through
the exact rollout path; openthoughts-tblite is additionally screened to the
tasks whose solutions pass on Modal/gVisor).

Layout mirrors the slime-data volume root: `{ver}/*.jsonl` are the eval
datasets; `metadata.task_path` in each row resolves against the repo root
(`ASYNC_RL_TASK_ROOT`). Install:

```bash
EXPERIMENT_CONFIG=w_qwen3_swe_eval modal run slime/modal_train.py::download_data
```
(or `huggingface_hub.snapshot_download` straight into your data root)
""")

from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(folder_path=str(bundle), repo_id=repo, repo_type="dataset",
                  commit_message=f"publish {ver}: {len(task_paths)} task dirs")
print(f"published {ver} + {len(task_paths)} task dirs -> https://huggingface.co/datasets/{repo}")
PYEOF
fi

