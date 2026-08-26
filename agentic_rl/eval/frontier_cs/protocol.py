"""Typed registry for the common Frontier-CS held-out evaluation protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path(__file__).with_name("arms.json")
# In-repo launch target since RUNBOOK §7 step 4 (was the guide repo's
# frontier_cs.w_qwen3_6_27b_frontier_cs_heldout_avg3 EXPERIMENT_CONFIG).
CONFIG_MODULE = "agentic_rl.launch.modal_train (ROLLOUT_MODE=eval)"
_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class EvalProtocol:
    dataset: str
    training_dataset: str
    eval_sha256: str
    train_sha256: str
    expected_tasks: int
    samples_per_task: int
    temperature: float
    top_p: float
    top_k: int
    max_response_len: int
    max_context_len: int
    max_steps: int
    episode_timeout_seconds: int
    verifier_timeout_seconds: int
    exec_timeout_seconds: int
    think_closure: bool
    rollout_seed: int


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    source_run_tag: str
    checkpoint_step: int | None = None
    # Absolute checkpoint root overriding the swe_ckpts/<source_run_tag>
    # convention — needed for arms that are not training runs (e.g. the
    # vanilla base-model conversion, which lives at
    # /checkpoints/Qwen3.6-27B_torch_dist with a `release` layout).
    load_path: str | None = None

    @property
    def checkpoint_path(self) -> str:
        return self.load_path or f"/checkpoints/swe_ckpts/{self.source_run_tag}"

    def environment(self, eval_id: str) -> dict[str, str]:
        values = {
            "ROLLOUT_MODE": "eval",
            "FRONTIER_CS_EVAL_ARM": self.key,
            "FRONTIER_CS_EVAL_RUN_TAG": self.source_run_tag,
            "FRONTIER_CS_EVAL_ID": eval_id,
        }
        if self.checkpoint_step is not None:
            values["FRONTIER_CS_EVAL_CKPT_STEP"] = str(self.checkpoint_step)
        if self.load_path is not None:
            values["FRONTIER_CS_EVAL_LOAD"] = self.load_path
        return values


def _require_int(data: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"protocol.{key} must be an integer >= {minimum}, got {value!r}")
    return value


def _load_protocol(data: dict[str, Any]) -> EvalProtocol:
    required_strings = ("dataset", "training_dataset")
    for key in required_strings:
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"protocol.{key} must be a non-empty string")
    hashes = {}
    for key in ("eval_sha256", "train_sha256"):
        value = data.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"protocol.{key} must be a lowercase SHA-256 digest")
        hashes[key] = value
    return EvalProtocol(
        dataset=data["dataset"],
        training_dataset=data["training_dataset"],
        eval_sha256=hashes["eval_sha256"],
        train_sha256=hashes["train_sha256"],
        expected_tasks=_require_int(data, "expected_tasks", minimum=1),
        samples_per_task=_require_int(data, "samples_per_task", minimum=1),
        temperature=float(data["temperature"]),
        top_p=float(data["top_p"]),
        top_k=_require_int(data, "top_k", minimum=-1),
        max_response_len=_require_int(data, "max_response_len", minimum=1),
        max_context_len=_require_int(data, "max_context_len", minimum=1),
        max_steps=_require_int(data, "max_steps", minimum=1),
        episode_timeout_seconds=_require_int(data, "episode_timeout_seconds", minimum=1),
        verifier_timeout_seconds=_require_int(data, "verifier_timeout_seconds", minimum=1),
        exec_timeout_seconds=_require_int(data, "exec_timeout_seconds", minimum=1),
        think_closure=bool(data["think_closure"]),
        rollout_seed=_require_int(data, "rollout_seed"),
    )


def load_registry(path: Path = DEFAULT_REGISTRY) -> tuple[EvalProtocol, dict[str, ArmSpec]]:
    """Load and validate the common protocol and checkpoint registry."""

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("protocol"), dict):
        raise ValueError(f"{path}: expected a protocol mapping")
    if not isinstance(raw.get("arms"), dict) or not raw["arms"]:
        raise ValueError(f"{path}: expected a non-empty arms mapping")

    protocol = _load_protocol(raw["protocol"])
    if protocol.max_response_len > protocol.max_context_len:
        raise ValueError("max_response_len cannot exceed max_context_len")
    if not 0 < protocol.top_p <= 1:
        raise ValueError(f"protocol.top_p must be in (0, 1], got {protocol.top_p}")

    arms: dict[str, ArmSpec] = {}
    for key, value in raw["arms"].items():
        if not isinstance(key, str) or not _SAFE_TAG.fullmatch(key):
            raise ValueError(f"unsafe arm key {key!r}")
        if not isinstance(value, dict):
            raise ValueError(f"arm {key!r} must be a mapping")
        label = value.get("label")
        source_run_tag = value.get("source_run_tag")
        if not isinstance(label, str) or not label:
            raise ValueError(f"arm {key!r} needs a non-empty label")
        if not isinstance(source_run_tag, str) or not _SAFE_TAG.fullmatch(source_run_tag):
            raise ValueError(f"arm {key!r} has unsafe source_run_tag {source_run_tag!r}")
        checkpoint_step = value.get("checkpoint_step")
        if checkpoint_step is not None and (
            not isinstance(checkpoint_step, int) or isinstance(checkpoint_step, bool) or checkpoint_step < 0
        ):
            raise ValueError(f"arm {key!r} checkpoint_step must be null or a non-negative integer")
        load_path = value.get("load_path")
        if load_path is not None and (
            not isinstance(load_path, str)
            or not load_path.startswith("/")
            or not re.fullmatch(r"/[A-Za-z0-9._/-]+", load_path)
        ):
            raise ValueError(f"arm {key!r} load_path must be null or a safe absolute path")
        arms[key] = ArmSpec(
            key=key,
            label=label,
            source_run_tag=source_run_tag,
            checkpoint_step=checkpoint_step,
            load_path=load_path,
        )
    return protocol, arms
