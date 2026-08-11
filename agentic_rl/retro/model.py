"""Exact head-process checkpointing for suffix-only retro continuations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from agentic_rl.model import Chain


@dataclass(frozen=True)
class ChainCheckpoint:
    """State needed for the next ``RecordingModel.query(messages)`` call.

    ``tokens`` ends immediately after the last assistant action.  The latest
    tool observation is present in ``messages`` but has not yet been rendered
    into tokens; this mirrors the live model seam at a post-tool turn boundary.
    """

    tokens: list[int]
    seen_msgs: int
    msg_hashes: list[str]
    messages: list[dict[str, Any]]
    n_calls: int
    cost: float
    extra_template_vars: dict[str, Any]
    source_weight_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "seen_msgs": self.seen_msgs,
            "msg_hashes": list(self.msg_hashes),
            "messages": copy.deepcopy(self.messages),
            "n_calls": self.n_calls,
            "cost": self.cost,
            "extra_template_vars": copy.deepcopy(self.extra_template_vars),
            "source_weight_version": self.source_weight_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChainCheckpoint:
        return cls(
            tokens=[int(token) for token in value["tokens"]],
            seen_msgs=int(value["seen_msgs"]),
            msg_hashes=[str(item) for item in value["msg_hashes"]],
            messages=copy.deepcopy(value["messages"]),
            n_calls=int(value.get("n_calls", 0)),
            cost=float(value.get("cost", 0.0)),
            extra_template_vars=copy.deepcopy(value.get("extra_template_vars", {})),
            source_weight_version=str(value.get("source_weight_version") or ""),
        )


def capture_checkpoint(agent: Any, model: Any) -> ChainCheckpoint:
    if not getattr(model, "chains", None):
        raise ValueError("cannot snapshot retro state before the first model turn")
    chain = model.cur
    if not chain.tokens or not chain.seen_msgs:
        raise ValueError("cannot snapshot incomplete RecordingModel chain")
    versions = [version for version in getattr(chain, "versions", ()) if version is not None]
    return ChainCheckpoint(
        tokens=list(chain.tokens),
        seen_msgs=int(chain.seen_msgs),
        msg_hashes=list(chain.msg_hashes),
        messages=copy.deepcopy(agent.messages),
        n_calls=int(agent.n_calls),
        cost=float(agent.cost),
        extra_template_vars=copy.deepcopy(agent.extra_template_vars),
        source_weight_version=str(versions[-1]) if versions else "",
    )


def restore_recording_model(model: Any, checkpoint: ChainCheckpoint) -> None:
    """Seed a fresh recorder with an exact, fully masked inherited prefix."""

    chain = Chain()
    chain.tokens = list(checkpoint.tokens)
    chain.loss_mask = [0] * len(chain.tokens)
    chain.logprobs = [0.0] * len(chain.tokens)
    chain.versions = []
    chain.prompt_len = len(chain.tokens)
    chain.seen_msgs = checkpoint.seen_msgs
    chain.msg_hashes = list(checkpoint.msg_hashes)
    chain.full_prompt = model.tokenizer.decode(chain.tokens, skip_special_tokens=False)
    if hasattr(chain, "turn_spans"):
        chain.turn_spans = []
    if hasattr(chain, "turn_ts"):
        chain.turn_ts = []
    model.chains = [chain]


def restore_agent(agent: Any, checkpoint: ChainCheckpoint) -> None:
    agent.messages = copy.deepcopy(checkpoint.messages)
    agent.n_calls = checkpoint.n_calls
    agent.cost = checkpoint.cost
    agent.extra_template_vars = copy.deepcopy(checkpoint.extra_template_vars)
