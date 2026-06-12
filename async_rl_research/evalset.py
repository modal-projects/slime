"""Build a versioned eval set by subsampling converted slime datasets.

An *eval set* is a directory of per-subset JSONL files drawn from
already-converted datasets (the outputs of ``env/convert2slime/*``), plus a
manifest pinning exactly which instances were chosen. Build it once onto the
slime-data volume, then point the training config's inline ``eval_config``
at the subset files. The rows are ordinary slime prompt rows, so eval drives
the same ``runtime x env`` stack as training.

Spec YAML::

    task_root: /data            # optional: harbor metadata.task_path values are
                                # rewritten relative to this dir, so ONE
                                # ASYNC_RL_TASK_ROOT covers train + eval rows.
                                # Omit to inline absolute task paths instead.
    subsets:
      - name: swebench_verified_50
        source: /data/swebench_verified/swebench_verified.jsonl
        n: 50                   # omit -> keep all rows
        seed: 0                 # deterministic subsample (default 0)
      - name: usaco_hard
        source: /data/usaco/usaco.jsonl
        ids: [usaco_829, ...]   # optional instance_id allowlist, applied before n

Usage::

    python -m async_rl_research.evalset spec.yaml --out-dir /data/evalsets/v0

Outputs ``<out-dir>/<name>.jsonl`` per subset, ``manifest.json`` (spec +
chosen instance ids), and ``eval_config.yaml`` (a ready ``--eval-config``
file). It also prints the equivalent inline ``eval_config`` dict to paste
into a training config. Paths inside the spec should be the paths as seen at
*runtime* (e.g. ``/data/...`` on the cluster); run the builder where those
paths resolve (a Modal shell / function with the volume mounted) so the
harbor task-dir checks mean something. After building, oracle-check a subset:
``ASYNC_RL_TASK_ROOT=<task_root> python -m async_rl_research.environment.harbor
<out-dir>/<name>.jsonl --limit 3``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def _load_spec(path: Path) -> dict[str, Any]:
    import yaml

    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("subsets"), list) or not spec["subsets"]:
        raise SystemExit(f"{path}: spec must be a mapping with a non-empty `subsets` list")
    names = [s.get("name") for s in spec["subsets"]]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise SystemExit(f"{path}: every subset needs a unique `name` (got {names})")
    for s in spec["subsets"]:
        if not s.get("source"):
            raise SystemExit(f"{path}: subset {s.get('name')!r} is missing `source`")
        unknown = set(s) - {"name", "source", "n", "seed", "ids"}
        if unknown:
            raise SystemExit(f"{path}: subset {s['name']!r} has unknown keys {sorted(unknown)}")
    return spec


def _instance_id(row: dict[str, Any], index: int) -> str:
    return (row.get("metadata") or {}).get("instance_id") or row.get("label") or f"row-{index}"


def _rewrite_task_path(row: dict[str, Any], source_dir: Path, task_root: Path | None, problems: list[str]) -> None:
    """Re-root a harbor row's relative task_path so it stays resolvable.

    Converted harbor rows carry task_path relative to their converter's
    out dir; a subsampled copy lives elsewhere, so pin the path down: relative
    to ``task_root`` when given (matching the run's single ASYNC_RL_TASK_ROOT),
    absolute otherwise (env/harbor accepts absolute paths as-is).
    """
    md = row.get("metadata") or {}
    if md.get("task_type") != "harbor" or not md.get("task_path"):
        return
    task_dir = Path(md["task_path"])
    if not task_dir.is_absolute():
        task_dir = source_dir / task_dir
    if task_root is not None:
        try:
            md["task_path"] = str(task_dir.relative_to(task_root))
        except ValueError:
            problems.append(f"task dir {task_dir} is outside task_root {task_root}; kept absolute")
            md["task_path"] = str(task_dir)
    else:
        md["task_path"] = str(task_dir)
    if not task_dir.is_dir():
        problems.append(f"task dir not found: {task_dir}")


def _build_subset(subset: dict[str, Any], out_dir: Path, task_root: Path | None, strict: bool) -> dict[str, Any]:
    source = Path(subset["source"])
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    ids = [_instance_id(row, i) for i, row in enumerate(rows)]

    if allow := subset.get("ids"):
        missing = set(allow) - set(ids)
        if missing:
            raise SystemExit(f"subset {subset['name']!r}: ids not in {source}: {sorted(missing)}")
        keep = [i for i, iid in enumerate(ids) if iid in set(allow)]
    else:
        keep = list(range(len(rows)))

    n = subset.get("n")
    if n is not None and n < len(keep):
        keep = sorted(random.Random(subset.get("seed", 0)).sample(keep, n))

    problems: list[str] = []
    chosen = []
    for i in keep:
        row = json.loads(json.dumps(rows[i]))  # deep copy; never mutate the source rows
        _rewrite_task_path(row, source.parent, task_root, problems)
        chosen.append(row)

    if problems:
        for p in problems:
            print(f"  [{subset['name']}] WARNING: {p}", file=sys.stderr)
        if strict:
            raise SystemExit(f"subset {subset['name']!r}: {len(problems)} problem(s) with --strict")

    out_path = out_dir / f"{subset['name']}.jsonl"
    with out_path.open("w") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {subset['name']}: {len(chosen)}/{len(rows)} rows from {source} -> {out_path}")
    return {
        "name": subset["name"],
        "source": str(source),
        "n_source_rows": len(rows),
        "n_rows": len(chosen),
        "seed": subset.get("seed", 0),
        "path": str(out_path),
        "instance_ids": [ids[i] for i in keep],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="eval-set spec YAML (see module docstring)")
    parser.add_argument("--out-dir", type=Path, required=True, help="eval-set output dir (one dir per version)")
    parser.add_argument("--strict", action="store_true", help="fail on missing task dirs instead of warning")
    args = parser.parse_args()

    spec = _load_spec(args.spec)
    task_root = Path(spec["task_root"]) if spec.get("task_root") else None
    # Resolve so the paths baked into the manifest / eval_config are absolute.
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    built = [_build_subset(s, args.out_dir, task_root, args.strict) for s in spec["subsets"]]

    manifest = {"spec": spec, "subsets": built}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    datasets = [{"name": b["name"], "path": b["path"]} for b in built]
    import yaml

    (args.out_dir / "eval_config.yaml").write_text(yaml.dump({"eval": {"datasets": datasets}}, sort_keys=False))

    print(f"\nwrote {args.out_dir}/manifest.json and {args.out_dir}/eval_config.yaml")
    if task_root is not None:
        print(f"run with ASYNC_RL_TASK_ROOT={task_root} (harbor task_paths are relative to it)")
    print("\ninline eval_config for the training config:\n")
    print("    eval_config = {")
    print('        "defaults": {"n_samples_per_eval_prompt": 1},')
    print('        "datasets": [')
    for d in datasets:
        print(f'            {{"name": "{d["name"]}", "path": "{d["path"]}"}},')
    print("        ],")
    print("    }")


if __name__ == "__main__":
    main()
