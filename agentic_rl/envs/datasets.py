"""In-repo dataset registry: the "dataset key" half of family onboarding.

RUNBOOK §7 step 7: a new task family needs only its ``envs/<family>/`` dir, a
row here, and an arm script. Each spec names the published harbor-format HF
repo (mirroring the ``/data`` volume layout: ``<key>/{train,eval}.jsonl`` +
``<key>/tasks/<id>/…``) and how training rows are obtained:

* ``has_train=True`` — the repo ships ``train.jsonl`` (frontier_cs,
  swe_rebench_v2); training reads it directly.
* ``split=SplitSpec(...)`` — the repo is eval-only; ``materialize_split``
  derives a deterministic train/heldout partition ON THE VOLUME at
  ``download_data`` time (the published repo is never mutated). Seeded and
  recorded (counts + sha256 sidecar), so it is exactly reproducible.

``group_regex`` makes the split group-disjoint (e.g. whole repos held out for
SWE-Bench Pro) instead of row-random; None = random by row. Note the leakage
trade-off documented in each family README.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitSpec:
    eval_count: int  # rows held out of training
    seed: int = 20260826
    group_regex: str | None = None  # instance_id → group key; None = per-row split


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # volume dir under /data and TRAIN_DATASET value
    hf_repo: str
    family: str  # envs/<family> that owns conversion + provenance
    has_train: bool = False
    needs_judge: bool = False  # frontier_cs verifier server
    supports_retro: bool = False  # retro capture env exists for this family
    split: SplitSpec | None = None


DATASETS: dict[str, DatasetSpec] = {
    "frontier_cs": DatasetSpec(
        key="frontier_cs",
        hf_repo="junlin-modal/frontier-cs",
        family="frontier_cs",
        has_train=True,
        needs_judge=True,
        supports_retro=True,
    ),
    "swe_rebench_v2": DatasetSpec(
        key="swe_rebench_v2",
        hf_repo="junlin-modal/swe-rebench-v2",
        family="swe_rebench",
        has_train=True,
    ),
    "terminal_bench_2_1": DatasetSpec(
        key="terminal_bench_2_1",
        hf_repo="junlin-modal/terminal-bench-2.1",
        family="terminal_bench",
        # 89 tasks total — primarily a transfer-eval set; training on it uses
        # the derived 69/20 split below and should be a deliberate choice.
        split=SplitSpec(eval_count=20),
    ),
    "swebenchpro": DatasetSpec(
        key="swebenchpro",
        hf_repo="junlin-modal/swebenchpro",
        family="swebenchpro",
        # 731 tasks / only 11 distinct repos. Default split is per-row random
        # (same-repo issues can appear on both sides — the standard SWE-bench
        # framing); see envs/swebenchpro/README.md for the group-disjoint
        # alternative and why it is high-variance here.
        split=SplitSpec(eval_count=81),
    ),
}


def train_jsonl_path(spec: DatasetSpec, data_root: str | Path) -> Path:
    """The jsonl an arm trains on (published train.jsonl or the derived split)."""

    base = Path(data_root) / spec.key
    if spec.has_train:
        return base / "train.jsonl"
    if spec.split is None:
        raise ValueError(f"dataset {spec.key!r} has no train split (eval-only)")
    return base / f"train.split-{spec.split.seed}.jsonl"


def heldout_jsonl_path(spec: DatasetSpec, data_root: str | Path) -> Path:
    base = Path(data_root) / spec.key
    if spec.has_train or spec.split is None:
        return base / "eval.jsonl"
    return base / f"eval.split-{spec.split.seed}.jsonl"


def make_split(
    rows: list[dict],
    *,
    eval_count: int,
    seed: int,
    group_regex: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Deterministic (train_rows, eval_rows) partition of ``rows``.

    Row order within each side preserves the input order. With ``group_regex``
    whole groups are held out until ``eval_count`` rows are covered (the last
    group may overshoot — held-out size is then >= eval_count).
    """

    if not 0 < eval_count < len(rows):
        raise ValueError(f"eval_count must be in (0, {len(rows)}), got {eval_count}")

    def _instance_id(row: dict) -> str:
        instance = (row.get("metadata") or {}).get("instance_id") or row.get("label")
        if not isinstance(instance, str) or not instance:
            raise ValueError("row has no metadata.instance_id or label to split on")
        return instance

    rng = random.Random(seed)
    if group_regex is None:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        heldout = set(indices[:eval_count])
    else:
        pattern = re.compile(group_regex)

        def _group(row: dict) -> str:
            instance = _instance_id(row)
            match = pattern.search(instance)
            return match.group(0) if match else instance

        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(_group(row), []).append(index)
        order = sorted(groups)
        rng.shuffle(order)
        heldout = set()
        for name in order:
            if len(heldout) >= eval_count:
                break
            heldout.update(groups[name])
    train = [row for index, row in enumerate(rows) if index not in heldout]
    evaluation = [row for index, row in enumerate(rows) if index in heldout]
    return train, evaluation


def materialize_split(spec: DatasetSpec, data_root: str | Path) -> dict:
    """Write the derived train/eval split next to the pulled dataset.

    Idempotent: re-running with the same seed rewrites byte-identical files.
    Returns the audit record (also written to ``split-<seed>.json``).
    """

    if spec.split is None:
        raise ValueError(f"dataset {spec.key!r} declares no split")
    base = Path(data_root) / spec.key
    source = base / "eval.jsonl"
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    train, evaluation = make_split(
        rows,
        eval_count=spec.split.eval_count,
        seed=spec.split.seed,
        group_regex=spec.split.group_regex,
    )

    def _write(path: Path, subset: list[dict]) -> str:
        text = "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in subset)
        path.write_text(text, encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    train_path = train_jsonl_path(spec, data_root)
    eval_path = heldout_jsonl_path(spec, data_root)
    record = {
        "dataset": spec.key,
        "source": str(source),
        "source_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        "seed": spec.split.seed,
        "group_regex": spec.split.group_regex,
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "train_sha256": _write(train_path, train),
        "eval_sha256": _write(eval_path, evaluation),
    }
    (base / f"split-{spec.split.seed}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
