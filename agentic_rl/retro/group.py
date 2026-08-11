"""Construct canonical eight-slot GRPO groups from one retro manifest."""

from __future__ import annotations

import copy

from slime.utils.types import Sample

from .manifest import RetroSnapshotManifest, SnapshotStatus

RETRO_ENV_SPEC = "agentic_rl.retro.env:RetroFrontierCsEnv"


def template_from_manifest(manifest: RetroSnapshotManifest) -> Sample:
    metadata = dict(manifest.sample_metadata)
    prompt = metadata.pop("_retro_prompt", "")
    label = metadata.pop("_retro_label", None)
    metadata.pop("task_type", None)
    return Sample(prompt=prompt, label=label, metadata=metadata)


def make_branch_group(
    template: Sample,
    manifest: RetroSnapshotManifest,
    *,
    group_index: int,
    first_sample_index: int,
    width: int = 8,
) -> list[Sample]:
    if width != 8:
        raise ValueError(f"initial retro design requires width=8, got {width}")
    if manifest.is_expired() or manifest.status not in (
        SnapshotStatus.AVAILABLE,
        SnapshotStatus.LEASED,
    ):
        raise ValueError(f"retro snapshot {manifest.snapshot_id} is not eligible")

    group: list[Sample] = []
    for offset in range(width):
        sample = copy.deepcopy(template)
        sample.group_index = group_index
        sample.index = first_sample_index + offset
        sample.rollout_id = sample.index
        sample.session_id = None
        sample.tokens = []
        sample.response = ""
        sample.response_length = 0
        sample.reward = None
        sample.loss_mask = None
        sample.weight_versions = []
        sample.rollout_log_probs = None
        sample.remove_sample = False
        sample.status = Sample.Status.PENDING
        sample.metadata = {
            **manifest.sample_metadata,
            **(template.metadata or {}),
            "task_type": RETRO_ENV_SPEC,
            "retro_manifest": manifest.to_dict(),
            "retro_sibling": offset,
        }
        group.append(sample)
    return group
