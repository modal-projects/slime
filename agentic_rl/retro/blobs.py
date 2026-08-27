"""Volume-side checkpoint blobs — keep the manifest JSONL small.

A manifest row used to inline the full ChainCheckpoint (raw token ids + all
messages, ~0.3–1.5 MB per row), which makes every pool reload parse token
arrays and becomes untenable once manifests also carry per-turn records
(all-turns capture). The checkpoint now lives in a sibling file next to the
ledger:

    <manifest dir>/blobs/<snapshot_id>.checkpoint.json

and ``manifest.agent_state`` carries ``{"checkpoint_ref": <relative path>,
"sha256": <content hash>}``. Rollout workers resolve refs against
``ASYNC_RL_RETRO_MANIFEST_PATH`` (the same checkpoints-volume file they append
manifests to). ``load_checkpoint`` also accepts the legacy inline
``{"checkpoint": {...}}`` shape so unit tests without a volume and mixed
mid-run ledgers keep working; capture falls back to inline when no manifest
path is configured.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentic_rl.retro.blobs")

_BLOB_DIR = "blobs"


def write_checkpoint_blob(manifest_path: str | Path, snapshot_id: str, checkpoint: dict[str, Any]) -> dict[str, str]:
    """Write the checkpoint payload atomically; return the agent_state ref dict."""

    from .manifest import _sanitize  # same secret-stripping the inline path had

    ref = f"{_BLOB_DIR}/{snapshot_id}.checkpoint.json"
    blob_path = Path(manifest_path).parent / ref
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_sanitize(checkpoint), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    tmp_path = blob_path.with_suffix(f".tmp-{os.getpid()}")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, blob_path)
    return {"checkpoint_ref": ref, "sha256": digest}


def load_checkpoint(agent_state: dict[str, Any], manifest_path: str | Path | None) -> dict[str, Any]:
    """Resolve an agent_state to its checkpoint dict (ref or legacy inline)."""

    inline = agent_state.get("checkpoint")
    if isinstance(inline, dict):
        return inline
    ref = agent_state.get("checkpoint_ref")
    if not ref:
        raise ValueError("agent_state carries neither 'checkpoint' nor 'checkpoint_ref'")
    if manifest_path is None:
        raise ValueError(
            "agent_state uses a checkpoint_ref but no manifest path is configured "
            "(ASYNC_RL_RETRO_MANIFEST_PATH) to resolve it against"
        )
    blob_path = Path(manifest_path).parent / str(ref)
    data = blob_path.read_bytes()
    expected = str(agent_state.get("sha256") or "")
    if expected:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"checkpoint blob {ref} is corrupt: sha {actual} != recorded {expected}")
    return json.loads(data)


def delete_checkpoint_blob(agent_state: dict[str, Any], manifest_path: str | Path | None) -> None:
    """Best-effort blob removal at snapshot GC time; failures only warn."""

    ref = (agent_state or {}).get("checkpoint_ref")
    if not ref or manifest_path is None:
        return
    try:
        (Path(manifest_path).parent / str(ref)).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("checkpoint blob cleanup %s: %s", ref, exc)
