"""Roll per-arm held-out ``summary.json`` files into one results document.

Reproduces the schema of ``results/heldout-avg3-*.json`` (RUNBOOK §6 debt #7):
per-arm avg@3 with a task-level bootstrap standard error, plus paired
task-bootstrap comparisons between named arm pairs. Deterministic given
``--seed`` (a fresh generator per arm/pair, so adding arms or pairs never
shifts existing numbers).

    python -m agentic_rl.eval.frontier_cs.rollup \
        --summary p50=/path/to/p50/summary.json \
        --summary base=/path/to/base/summary.json \
        --pair "p50 - base" \
        --notes "..." --output results/heldout-avg3-YYYYMMDD.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date as _date
from pathlib import Path
from typing import Any

DEFAULT_REPLICATES = 50_000
DEFAULT_SEED = 20260817


def _task_means(summary: dict[str, Any]) -> dict[str, float]:
    per_task = summary.get("per_task")
    if not isinstance(per_task, dict) or not per_task:
        raise ValueError("summary has no per_task block; rerun aggregate on the dump")
    return {task: float(entry["mean_reward"]) for task, entry in per_task.items()}


def _rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def bootstrap_se(values: list[float], *, replicates: int, seed: int) -> float:
    """Task-level bootstrap SE of the mean (resample tasks with replacement)."""

    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    idx = _rng(seed).integers(0, len(arr), size=(replicates, len(arr)))
    return float(arr[idx].mean(axis=1).std(ddof=1))


def paired_comparison(
    a: dict[str, float],
    b: dict[str, float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired task-bootstrap of mean(a - b) over the shared task set."""

    import numpy as np

    if set(a) != set(b):
        raise ValueError(
            f"paired arms cover different tasks ({len(set(a) ^ set(b))} mismatched)"
        )
    tasks = sorted(a)
    deltas = np.asarray([a[t] - b[t] for t in tasks], dtype=np.float64)
    idx = _rng(seed).integers(0, len(deltas), size=(replicates, len(deltas)))
    replicate_means = deltas[idx].mean(axis=1)
    return {
        "mean_difference": float(deltas.mean()),
        "paired_standard_error": float(replicate_means.std(ddof=1)),
        "bootstrap_95_interval": [
            float(np.percentile(replicate_means, 2.5)),
            float(np.percentile(replicate_means, 97.5)),
        ],
        "wins_ties_losses": [
            int((deltas > 0).sum()),
            int((deltas == 0).sum()),
            int((deltas < 0).sum()),
        ],
    }


def rollup(
    summaries: dict[str, dict[str, Any]],
    pairs: list[tuple[str, str]],
    *,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    notes: str = "",
    date: str | None = None,
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for key, summary in summaries.items():
        means = _task_means(summary)
        metadata = summary.get("metadata") or {}
        arms[key] = {
            "avg_at_3": summary["avg_at_k"],
            "se": bootstrap_se(list(means.values()), replicates=replicates, seed=seed),
            "pass_at_3": summary["pass_at_k"],
            "solve": summary["sample_solve_rate"],
            "eval_id": metadata.get("eval_id"),
            "checkpoint_step": metadata.get("checkpoint_step"),
        }
    paired = []
    for left, right in pairs:
        for key in (left, right):
            if key not in summaries:
                raise ValueError(f"pair references unknown arm {key!r}")
        comparison = paired_comparison(
            _task_means(summaries[left]),
            _task_means(summaries[right]),
            replicates=replicates,
            seed=seed,
        )
        paired.append({"pair": f"{left} - {right}", **comparison})
    return {
        "evaluation": "frontier-cs-heldout-avg3",
        "date": date or _date.today().isoformat(),
        "notes": notes
        + (" " if notes else "")
        + f"Bootstrap: {replicates} replicates, seed {seed}, unit=task.",
        "arms": arms,
        "paired": paired,
    }


def _parse_summary_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ARM=path/to/summary.json")
    key, _, path = value.partition("=")
    return key.strip(), Path(path)


def _parse_pair_arg(value: str) -> tuple[str, str]:
    if "-" not in value:
        raise argparse.ArgumentTypeError('expected "left - right"')
    left, _, right = value.partition("-")
    return left.strip(), right.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        type=_parse_summary_arg,
        help="ARM=path/to/summary.json (repeatable)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        type=_parse_pair_arg,
        help='"left - right" paired comparison (repeatable)',
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--notes", default="")
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = {key: json.loads(path.read_text()) for key, path in args.summary}
    document = rollup(
        summaries,
        args.pair,
        replicates=args.replicates,
        seed=args.seed,
        notes=args.notes,
        date=args.date,
    )
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
