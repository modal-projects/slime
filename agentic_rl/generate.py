"""Per-sample agentic-RL rollout hook for slime.

Wire-up: ``--custom-generate-function-path agentic_rl.generate.generate``.

One call = one episode: pick the env by ``metadata.task_type``, run the
in-process mini-swe loop (the token-recording ``RecordingModel`` + a bash-only
Modal ``Sandbox``) in a worker thread, and build token-faithful training
Sample(s). Each turn carries the ``weight_version`` live at generation, so
fully-async off-policy correction (TIS/clipping) and the async-health metrics
work. A weight-update abort mid-generation -> ``Status.ABORTED`` -> slime
recycles the episode. Reward is computed inline so slime's reward-model step is
skipped. Train and eval take the same path.

Episode limits come from ``args`` (``--custom-config-path``): ``agentic_max_steps``,
``agentic_episode_timeout``, ``agentic_exec_timeout``, ``agentic_grade_timeout``,
``agentic_eval_timeout``, ``agentic_query_timeout``, ``agentic_max_empty_turns``,
``agentic_max_boot_retries``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import logging
import threading
import time
import uuid
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from .environment.base import EnvMetadataError, EpisodeLimits, RewardResult, load_env
from .model import RecordingModel

logger = logging.getLogger("agentic_rl")

# Per-instance boot-failure tally: a task whose image won't build returns
# ABORTED and slime requeues the same group forever; after the cap we ship a
# masked reward-0 sample to drop it.
_boot_fails: dict[str, int] = {}

# Episodes are long I/O-bound chains; asyncio.to_thread caps at min(32, cpu+4)
# threads, which throttles in-flight episodes regardless of sglang concurrency.
# Use a wide dedicated pool sized to the rollout's serving concurrency.
_episode_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _episode_executor(args) -> concurrent.futures.ThreadPoolExecutor:
    global _episode_pool
    if _episode_pool is None:
        with _pool_lock:
            if _episode_pool is None:
                from slime.utils.http_utils import get_rollout_num_engines

                n = getattr(args, "sglang_server_concurrency", 128) * get_rollout_num_engines(args)
                n = min(max(n, 32), 2048)
                _episode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=n, thread_name_prefix="episode")
                logger.info("[agentic_rl] episode thread pool: max_workers=%d", n)
    return _episode_pool


def _limits(args) -> EpisodeLimits:
    return EpisodeLimits(
        max_steps=getattr(args, "agentic_max_steps", 30),
        episode_timeout=getattr(args, "agentic_episode_timeout", 1800),
        exec_timeout=getattr(args, "agentic_exec_timeout", 120),
        grade_timeout=getattr(args, "agentic_grade_timeout", 1800),
        eval_timeout=getattr(args, "agentic_eval_timeout", 600),
    )


def _sampling_params(state: GenerateState, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Base on the rollout's sampling defaults; let slime's per-call params (eval
    temperature/top_p, max_new_tokens) override."""
    params = dict(state.sampling_params)
    for k in ("temperature", "top_p", "top_k", "max_new_tokens", "stop", "stop_token_ids"):
        if overrides and overrides.get(k) is not None:
            params[k] = overrides[k]
    return params


def _session_id(sample: Sample, md: dict[str, Any]) -> str:
    if sample.index is not None and sample.group_index is not None:
        return f"agent-{md['instance_id']}-{sample.index}-{sample.group_index}"
    return f"agent-{md['instance_id']}-{uuid.uuid4().hex[:8]}"


def _run_episode(env, md: dict[str, Any], model: RecordingModel, limits: EpisodeLimits) -> RewardResult:
    """Synchronous episode body run in a worker thread. Never raises: an episode
    that errors mid-run still trains on the turns it produced (reward 0)."""
    try:
        return env.rollout(md, model=model, limits=limits)
    except Exception:  # noqa: BLE001
        logger.exception("[agentic_rl] episode failed (instance=%s)", md.get("instance_id"))
        return RewardResult(reward=0.0, is_solved=False, extra={"error": "episode_exception"})


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    state = GenerateState(args)
    if state.aborted:
        sample.status = Sample.Status.ABORTED
        return sample

    try:
        env = load_env((sample.metadata or {}).get("task_type"))
        md = env.normalize_metadata(sample)
    except EnvMetadataError as e:
        return _abort(sample, str(e))
    except (ValueError, TypeError) as e:
        return _abort(sample, f"env_dispatch:{type(e).__name__}:{e}")

    limits = _limits(args)
    sample.metadata = {**(sample.metadata or {}), "budgets": dataclasses.asdict(limits)}
    session_id = sample.session_id or _session_id(sample, md)
    sample.session_id = session_id

    model = RecordingModel(
        state.tokenizer,
        _sampling_params(state, sampling_params),
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        session_id,
        tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
        reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
        max_context_len=int(getattr(args, "rollout_max_context_len", 0) or 0),
        query_timeout=getattr(args, "agentic_query_timeout", 600),
        max_empty_turns=getattr(args, "agentic_max_empty_turns", 3),
    )

    t0 = time.time()
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_episode_executor(args), _run_episode, env, md, model, limits)
    except asyncio.CancelledError:
        if asyncio.current_task().cancelling():
            raise
        return _abort(sample, "exception:CancelledError")
    except Exception as e:  # noqa: BLE001
        logger.error("[agentic_rl] %s: rollout failed: %s", md["instance_id"], e)
        return _abort(sample, f"exception:{type(e).__name__}")

    if model.aborted:  # weight update aborted a turn mid-flight -> recycle
        sample.status = Sample.Status.ABORTED
        return sample

    return _build_samples(sample, model, result, state.tokenizer, md, args, elapsed=time.time() - t0)


def _build_samples(sample, model, result, tokenizer, md, args, *, elapsed: float):
    key = sample.label or md.get("instance_id") or ""
    usable = [c for c in model.chains if c.has_response]
    if not usable:
        _boot_fails[key] = _boot_fails.get(key, 0) + 1
        if _boot_fails[key] <= getattr(args, "agentic_max_boot_retries", 3):
            sample.status = Sample.Status.ABORTED
            return sample
        logger.warning("[agentic_rl] %s unusable after %d boot failures; dropping", key, _boot_fails[key])
        return _ship_null(sample, tokenizer, md, "ImageUnusable")
    _boot_fails.pop(key, None)

    k = len(usable)
    stats = {
        "turns": sum(len(c.versions) for c in usable),
        "format_errors": model.n_format_errors,
        "gen_time": round(model.gen_time, 1),
        "output_tokens": sum(sum(c.loss_mask) for c in usable),
        "response_tokens": sum(len(c.tokens) - c.prompt_len for c in usable),
        "chains": k,
        "gen_timestamp": time.time(),
        "is_solved": result.is_solved,
        **{key_: val for key_, val in result.extra.items() if key_ != "harbor_step_results"},
    }
    logger.info(
        "[agentic_rl] %s: reward=%.2f solved=%s turns=%d chains=%d elapsed=%.1fs",
        md["instance_id"], result.reward, result.is_solved, stats["turns"], k, elapsed,
    )

    samples = []
    for i, c in enumerate(usable):
        s = sample if i == 0 else copy.copy(sample)
        s.tokens = c.tokens
        s.response_length = len(c.tokens) - c.prompt_len
        s.loss_mask = c.loss_mask[c.prompt_len :]
        s.rollout_log_probs = c.logprobs[c.prompt_len :]
        s.weight_versions = [v for v in c.versions if v is not None]
        s.reward = result.reward / k
        s.response = tokenizer.decode(c.tokens[c.prompt_len :], skip_special_tokens=False)
        s.status = Sample.Status.COMPLETED
        s.rollout_id = sample.index  # siblings share rollout_id so reducers don't over-count
        s.metadata = {**(sample.metadata or {}), "instance_id": md["instance_id"], "agentic": stats}
        samples.append(s)

    # prefix-cache info is per episode; attribute it to the primary sample.
    samples[0].prefix_cache_info.cached_tokens = model.cached_tokens
    samples[0].prefix_cache_info.total_prompt_tokens = model.input_tokens
    return samples[0] if k == 1 else samples


def _ship_null(sample: Sample, tokenizer, md: dict[str, Any], reason: str) -> Sample:
    """A valid but fully-masked reward-0 sample for a permanently-failing task:
    ships to leave the buffer, contributes no gradient (remove_sample=True)."""
    ptoks = tokenizer.encode(md.get("problem_statement", "") or reason, add_special_tokens=False)[:512]
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (ptoks[-1] if ptoks else 0)
    sample.tokens = ptoks + [eos]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.weight_versions = []
    sample.reward = 0.0
    sample.status = Sample.Status.COMPLETED
    sample.remove_sample = True
    sample.rollout_id = sample.index
    sample.metadata = {**(sample.metadata or {}), "agentic": {"exit_status": reason, "turns": 0, "gen_timestamp": time.time()}}
    return sample


def _abort(sample: Sample, reason: str) -> Sample:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.rollout_id = sample.index
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[agentic_rl] aborted: %s", reason)
    return sample
