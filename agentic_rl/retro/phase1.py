"""Analyze a debug rollout dump produced by the Phase-1 branchability survey.

Usage:
    python -m agentic_rl.retro.phase1 /checkpoints/.../rollout_0.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from .survey import BranchOutcome, BranchSurveyGroup, summarize_groups


def groups_from_samples(samples: list[Sample]) -> list[BranchSurveyGroup]:
    by_group: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        retro = (sample.metadata or {}).get("agentic", {}).get("retro_branch")
        if isinstance(retro, dict) and sample.group_index is not None:
            by_group[int(sample.group_index)].append(sample)

    groups: list[BranchSurveyGroup] = []
    for siblings in by_group.values():
        if len(siblings) != 8:
            continue
        first_retro = siblings[0].metadata["agentic"]["retro_branch"]
        outcomes = tuple(_outcome(sample) for sample in siblings)
        groups.append(
            BranchSurveyGroup(
                snapshot_id=str(first_retro["snapshot_id"]),
                selector=str(first_retro.get("event_type") or "unknown"),
                inherited_score=float(first_retro.get("inherited_score") or 0.0),
                inherited_best=float(first_retro.get("inherited_best") or 0.0),
                outcomes=outcomes,
            )
        )
    return groups


def analyze_dump(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime environment owns Torch
        raise RuntimeError("import torch: Phase-1 dump analysis requires the Slime runtime") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = [Sample.from_dict(value) for value in payload.get("samples", [])]
    groups = groups_from_samples(samples)
    return {
        "rollout_id": payload.get("rollout_id"),
        "source": str(path),
        "summary": summarize_groups(groups),
        "groups": [group.metrics() for group in groups],
    }


def _outcome(sample: Sample) -> BranchOutcome:
    agentic = (sample.metadata or {}).get("agentic") or {}
    outcome = agentic.get("outcome") or {}
    fallback_reward = sample.reward if isinstance(sample.reward, (int, float)) else 0.0
    final = float(outcome.get("final") if outcome.get("final") is not None else fallback_reward)
    best = outcome.get("best_submitted")
    best = final if best is None else max(final, float(best))
    timing = agentic.get("timing") or {}
    submissions = agentic.get("submission_summary") or {}
    return BranchOutcome(
        reward=final,
        best_reward=best,
        output_tokens=int(agentic.get("output_tokens") or sum(sample.loss_mask or ())),
        sandbox_seconds=float(timing.get("agent") or agentic.get("elapsed_sec") or 0.0),
        judge_calls=int(submissions.get("n") or 0) + 1,
        weight_versions=tuple(str(value) for value in sample.weight_versions),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_dump(args.dump)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
