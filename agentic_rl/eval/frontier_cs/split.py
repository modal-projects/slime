"""Validate that Frontier-CS train and held-out evaluation rows are disjoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _instance_id(row: dict[str, Any], index: int) -> str:
    metadata = row.get("metadata") or {}
    value = metadata.get("instance_id") or row.get("label")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {index} has no metadata.instance_id or label")
    return value


def _problem_id(row: dict[str, Any]) -> str | None:
    verifier = ((row.get("metadata") or {}).get("verifier") or {}).get("env") or {}
    value = verifier.get("PROBLEM_ID")
    return str(value) if value is not None else None


def _index_rows(rows: Iterable[dict[str, Any]], split: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    problem_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{split} row {index} is not a JSON object")
        instance_id = _instance_id(row, index)
        if instance_id in indexed:
            raise ValueError(f"{split} contains duplicate instance_id {instance_id!r}")
        indexed[instance_id] = row
        if (problem_id := _problem_id(row)) is not None:
            if problem_id in problem_ids:
                raise ValueError(f"{split} contains duplicate PROBLEM_ID {problem_id!r}")
            problem_ids.add(problem_id)
    return indexed, problem_ids


def validate_split_rows(
    train_rows: Iterable[dict[str, Any]],
    eval_rows: Iterable[dict[str, Any]],
    *,
    expected_eval_tasks: int = 38,
) -> dict[str, Any]:
    """Validate split cardinality, uniqueness, and train/eval disjointness."""

    train, train_problem_ids = _index_rows(train_rows, "train")
    evaluation, eval_problem_ids = _index_rows(eval_rows, "eval")
    if len(evaluation) != expected_eval_tasks:
        raise ValueError(f"expected {expected_eval_tasks} eval tasks, found {len(evaluation)}")
    if overlap := set(train) & set(evaluation):
        raise ValueError(f"train/eval instance_id overlap: {sorted(overlap)}")
    if problem_overlap := train_problem_ids & eval_problem_ids:
        raise ValueError(f"train/eval PROBLEM_ID overlap: {sorted(problem_overlap)}")
    return {
        "train_tasks": len(train),
        "eval_tasks": len(evaluation),
        "instance_overlap": 0,
        "problem_id_overlap": 0,
        "train_instance_ids": sorted(train),
        "eval_instance_ids": sorted(evaluation),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_files(
    train_path: Path,
    eval_path: Path,
    *,
    expected_eval_tasks: int = 38,
) -> dict[str, Any]:
    """Validate JSONL files and return a reproducibility record with hashes."""

    summary = validate_split_rows(
        _read_jsonl(train_path),
        _read_jsonl(eval_path),
        expected_eval_tasks=expected_eval_tasks,
    )
    return {
        **summary,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_sha256": _sha256(train_path),
        "eval_sha256": _sha256(eval_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("/data/frontier_cs/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("/data/frontier_cs/eval.jsonl"))
    parser.add_argument("--expected-eval-tasks", type=int, default=38)
    args = parser.parse_args()
    summary = validate_split_files(
        args.train,
        args.eval,
        expected_eval_tasks=args.expected_eval_tasks,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
