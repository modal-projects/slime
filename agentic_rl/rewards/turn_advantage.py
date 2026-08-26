"""Turn-level advantage painting — TRAIN side.

Wire-up: ``--custom-advantage-function-path agentic_rl.rewards.turn_advantage.compute_turn_advantages``
(slime calls it from ``compute_advantages_and_returns`` after KL, and it must
populate ``rollout_data["advantages"]`` and ``rollout_data["returns"]``).

Per token:

    A_token = A_outcome + omega * A_turn(turn(token))

``A_outcome`` is ``rollout_data["rewards"][i]`` — the group-normalized episode
scalar, already computed rollout-side (broadcast to every token exactly like
slime's stock GRPO branch, so with the turn component absent or omega=0 this is
bit-identical to the builtin estimator). ``A_turn`` comes pre-normalized in
``rollout_data["metadata"][i]`` (see ``turn_reward.post_process_rewards``):
``turn_adv[j]`` is painted over the response-relative token span
``turn_spans[j]``; tokens outside every span (observation deltas, injected
think-closure ids) keep the pure outcome value — they are loss-masked anyway.

``omega`` is read from ``ASYNC_RL_TURN_MIX_WEIGHT`` (default 0.3; baked into
the run image by the O3 config, so rollout and train nodes agree).

Context parallelism: the per-sample tensors here must match ``kl[i]``'s shape,
which under CP>1 is the CP-local zigzag slice (rollout_log_probs are sliced in
``actor._get_rollout_data``). The turn values are painted in FULL response
space and re-sliced with ``slice_log_prob_with_cp`` — the same recipe
``get_reinforce_plus_plus_returns`` uses. Any shape surprise degrades that
sample to the pure outcome broadcast (never crashes the step).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("agentic_rl")

TURN_MIX_ENV = "ASYNC_RL_TURN_MIX_WEIGHT"
DEFAULT_TURN_MIX = 0.3


def _mix_weight() -> float:
    try:
        return float(os.environ.get(TURN_MIX_ENV, DEFAULT_TURN_MIX))
    except ValueError:
        return DEFAULT_TURN_MIX


def _paint_turns(response_length: int, spans, values, dtype, device) -> torch.Tensor:
    full = torch.zeros(response_length, dtype=dtype, device=device)
    for (start, end), v in zip(spans, values, strict=True):
        s = max(0, min(int(start), response_length))
        e = max(s, min(int(end), response_length))
        full[s:e] = float(v)
    return full


def _slice_to_local(full: torch.Tensor, total_length: int, response_length: int, local_len: int) -> torch.Tensor | None:
    """Bring a full-response-space tensor into kl[i]'s representation: identity
    when lengths already match (CP=1), zigzag CP slice otherwise."""
    if local_len == response_length:
        return full
    try:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        sliced = slice_log_prob_with_cp(full, total_length, response_length)
    except Exception:  # noqa: BLE001 — mpu not initialized / unexpected layout
        return None
    return sliced if len(sliced) == local_len else None


def compute_turn_advantages(args, rollout_data) -> None:
    """slime ``--custom-advantage-function-path`` hook."""
    kl = rollout_data["kl"]
    rewards = rollout_data["rewards"]
    metadata = rollout_data.get("metadata") or [None] * len(kl)
    response_lengths = rollout_data.get("response_lengths") or [len(k) for k in kl]
    total_lengths = rollout_data.get("total_lengths") or response_lengths
    omega = _mix_weight()

    returns: list[torch.Tensor] = []
    n_painted = 0
    for i in range(len(kl)):
        base = torch.ones_like(kl[i]) * float(rewards[i])
        md = metadata[i] if isinstance(metadata[i], dict) else None
        spans = (md or {}).get("turn_spans")
        values = (md or {}).get("turn_adv")
        if omega and spans and values and len(spans) == len(values):
            full = _paint_turns(int(response_lengths[i]), spans, values, base.dtype, base.device)
            local = _slice_to_local(full, int(total_lengths[i]), int(response_lengths[i]), len(kl[i]))
            if local is not None:
                base = base + omega * local.to(device=base.device, dtype=base.dtype)
                n_painted += 1
            else:
                logger.warning(
                    "[turn_advantage] sample %d: cannot map %d turn values onto local len %d "
                    "(response_len %d); using pure outcome",
                    i, len(values), len(kl[i]), int(response_lengths[i]),
                )
        returns.append(base)

    if n_painted:
        logger.info("[turn_advantage] painted turn advantages on %d/%d samples (omega=%.3g)", n_painted, len(kl), omega)
    rollout_data["returns"] = returns
    rollout_data["advantages"] = list(returns)
