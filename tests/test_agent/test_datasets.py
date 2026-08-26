"""The dataset registry + deterministic volume-side splits (envs/datasets.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentic_rl.envs.datasets import (
    DATASETS,
    DatasetSpec,
    SplitSpec,
    heldout_jsonl_path,
    make_split,
    materialize_split,
    train_jsonl_path,
)


def _rows(n: int, groups: int = 4) -> list[dict]:
    return [
        {
            "prompt": f"p{i}",
            "label": f"task-{i}",
            "metadata": {"instance_id": f"instance_repo{i % groups}__name-{'0' * 39}{i % 10}"},
        }
        for i in range(n)
    ]


def test_registry_covers_the_onboarded_families():
    assert {"frontier_cs", "swe_rebench_v2", "terminal_bench_2_1", "swebenchpro"} <= set(DATASETS)
    assert DATASETS["frontier_cs"].supports_retro
    assert DATASETS["frontier_cs"].needs_judge
    for key in ("terminal_bench_2_1", "swebenchpro"):
        spec = DATASETS[key]
        assert not spec.supports_retro and not spec.needs_judge
        assert not spec.has_train and spec.split is not None


def test_train_path_published_vs_derived():
    assert str(train_jsonl_path(DATASETS["frontier_cs"], "/data")) == "/data/frontier_cs/train.jsonl"
    derived = str(train_jsonl_path(DATASETS["swebenchpro"], "/data"))
    assert derived == "/data/swebenchpro/train.split-20260826.jsonl"
    eval_only = DatasetSpec(key="x", hf_repo="r", family="f")
    with pytest.raises(ValueError, match="no train split"):
        train_jsonl_path(eval_only, "/data")


def test_make_split_is_deterministic_disjoint_and_order_preserving():
    rows = _rows(50)
    one = make_split(rows, eval_count=10, seed=7)
    two = make_split(rows, eval_count=10, seed=7)
    assert one == two
    train, evaluation = one
    assert len(train) == 40 and len(evaluation) == 10
    train_ids = {r["label"] for r in train}
    eval_ids = {r["label"] for r in evaluation}
    assert not train_ids & eval_ids
    assert train_ids | eval_ids == {r["label"] for r in rows}
    # different seed → different partition
    assert make_split(rows, eval_count=10, seed=8)[1] != evaluation


def test_make_split_group_mode_holds_out_whole_groups():
    rows = _rows(40, groups=5)
    pattern = r"^instance_repo\d+"
    _, evaluation = make_split(rows, eval_count=8, seed=3, group_regex=pattern)
    import re

    eval_groups = {re.match(pattern, r["metadata"]["instance_id"]).group(0) for r in evaluation}
    train_groups = {
        re.match(pattern, r["metadata"]["instance_id"]).group(0)
        for r in make_split(rows, eval_count=8, seed=3, group_regex=pattern)[0]
    }
    assert not eval_groups & train_groups
    assert len(evaluation) >= 8  # last group may overshoot


def test_materialize_split_is_idempotent_and_audited(tmp_path):
    spec = DatasetSpec(
        key="toy", hf_repo="r", family="f", split=SplitSpec(eval_count=3, seed=11)
    )
    base = tmp_path / "toy"
    base.mkdir()
    (base / "eval.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in _rows(12))
    )
    record = materialize_split(spec, tmp_path)
    assert record["train_rows"] == 9 and record["eval_rows"] == 3
    train_path = train_jsonl_path(spec, tmp_path)
    eval_path = heldout_jsonl_path(spec, tmp_path)
    first = (train_path.read_bytes(), eval_path.read_bytes())
    again = materialize_split(spec, tmp_path)
    assert again == record
    assert (train_path.read_bytes(), eval_path.read_bytes()) == first
    audit = json.loads((base / "split-11.json").read_text())
    assert audit["train_sha256"] == record["train_sha256"]
    assert len(audit["train_sha256"]) == 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
