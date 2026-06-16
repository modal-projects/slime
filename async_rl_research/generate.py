"""Generic agentic-RL rollout entrypoint for slime (design A: HTTP adapter).

Wire-up: ``--custom-generate-function-path async_rl_research.generate.generate``.

Agent- and task-agnostic per-sample orchestrator. Owns the parts identical for
any in-sandbox agent on any task family (adapter/HTTP lifecycle, session
management, trajectory merge, abort/timeout isolation) and delegates the rest to
a runtime (``agent/base.py``) and an env (``env/base.py``, picked per row by
metadata.task_type). Per sample: ``_State`` serves the runtime's adapter on a bg
thread, a session is opened keyed by session_id, ``env.rollout`` runs the agent
(which dials back to the adapter per model call), and the recorded token
segments merge into Sample(s). Reward is computed inline so slime's reward-model
step is skipped.

Env knobs
---------
    SLIME_HEAD_HOST        public IP sandboxes use to reach the adapter
                           (REQUIRED unless MODAL_EXPOSE_ADAPTER=1)
    MODAL_EXPOSE_ADAPTER   1 to publish the adapter through a modal.forward
                           tunnel (required on a Modal cluster)
    SHIM_BIND_HOST         0.0.0.0   adapter bind host on the head node
    SHIM_PORT              18002     adapter bind port
    ASYNC_RL_AGENT_RUNTIME agent runtime spec (default "mini-swe")
    ASYNC_RL_AGENT_DRIVER  legacy alias for ASYNC_RL_AGENT_RUNTIME
    ASYNC_RL_TASK_ROOT     root dir relative metadata.task_path resolve against
    AGENT_TIME_BUDGET_SEC  1800  total agent wallclock budget per sample
    AGENT_EVAL_TIMEOUT_SEC 600   wallclock cap per grading command
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import traceback
from typing import Any

from slime.agent.trajectory import TokenSegment, fan_out_sample_segments
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .profiles import profiling
from .agent.base import AgentRuntime, load_runtime
from .aiohttp_threaded import run_app_in_thread
from .environment.base import EnvMetadataError, RewardResult, load_env
from .modal_sandbox import SandboxBootTimeout

logger = logging.getLogger(__name__)


AGENT_TIME_BUDGET_SEC = int(os.environ.get("AGENT_TIME_BUDGET_SEC", "1800"))
AGENT_EVAL_TIMEOUT_SEC = int(os.environ.get("AGENT_EVAL_TIMEOUT_SEC", "600"))
SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "18002"))
# On a Modal cluster sandboxes are network-isolated and reach the adapter only
# via a public modal.forward tunnel.
MODAL_EXPOSE_ADAPTER = os.environ.get("MODAL_EXPOSE_ADAPTER", "0").strip().lower() in ("1", "true", "yes")


def _load_runtime(args) -> AgentRuntime:
    """Resolve + instantiate the agent runtime (env > arg > registry default).

    Validation is eager so a misdeclared runtime fails the worker boot loudly.
    """
    spec = (
        os.environ.get("ASYNC_RL_AGENT_RUNTIME")
        or os.environ.get("ASYNC_RL_AGENT_DRIVER")  # legacy alias
        or getattr(args, "agent_runtime", None)
        or getattr(args, "agent_driver", None)
    )
    return load_runtime(spec)


# Singleton per worker process: tokenizer + adapter + bg HTTP server.
class _State(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.args = args
        self.runtime = _load_runtime(args)
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        # Reuse the served model's SGLang parsers so tool-call / reasoning parse.
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None

        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        public_host = os.environ.get("SLIME_HEAD_HOST")
        if not public_host and not MODAL_EXPOSE_ADAPTER:
            raise RuntimeError(
                "SLIME_HEAD_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for the reverse-connection to the adapter, "
                "or set MODAL_EXPOSE_ADAPTER=1 to publish the adapter through a "
                "modal.forward tunnel (required on a Modal cluster). Without "
                "either the in-sandbox agent cannot dial back and the rollout "
                "will silently abort."
            )

        self.adapter = self.runtime.adapter_cls(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
        )
        # Per-turn timing by session; install before the app starts.
        profiling.install(self.adapter.app)
        # handler_cancellation=True: a client disconnect cancels the handler and
        # arms /abort_request, else an inflight /generate trips sglang's idle assert.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=SHIM_BIND_HOST,
            port=SHIM_PORT,
            thread_name="agent-adapter",
            runner_kwargs={"handler_cancellation": True},
        )
        # Work past the bind can still fail (e.g. modal.forward); tear down so
        # the orphaned daemon thread doesn't hold SHIM_PORT against retries.
        try:
            self._tunnel_cm = None
            self.adapter_url = self._resolve_adapter_url(public_host)
        except BaseException:
            self.app_handle.stop()
            raise
        logger.info(
            "[async_rl] runtime=%s adapter=%s tokenizer=%s tool_parser=%s reasoning_parser=%s",
            self.runtime.name,
            self.adapter_url,
            args.hf_checkpoint,
            self.tool_parser,
            self.reasoning_parser,
        )

    def _resolve_adapter_url(self, public_host: str | None) -> str:
        """Pick the URL the in-sandbox agent dials back on.

        On a Modal cluster: a per-process ``modal.forward`` tunnel (one static
        env can't cover multiple data-parallel workers), held on ``self`` for
        the process lifetime.
        """
        if not MODAL_EXPOSE_ADAPTER:
            return f"http://{public_host}:{self.app_handle.port}"

        import modal

        # Blocking CM, never exited -- the process owns the tunnel until death.
        self._tunnel_cm = modal.forward(self.app_handle.port)
        tunnel = self._tunnel_cm.__enter__()
        logger.info("[async_rl] modal.forward tunnel for adapter port %d -> %s", self.app_handle.port, tunnel.url)
        return tunnel.url


# ---------------------------------------------------------------------------
# Trajectory -> Sample
# ---------------------------------------------------------------------------
def _start_session(state: _State, sample: Sample, md: dict[str, Any], sampling_params: dict[str, Any]) -> str:
    """Register the adapter session BEFORE the agent starts (it sends
    ``session_id`` as its bearer token to group its turns)."""
    if sample.session_id:
        session_id = sample.session_id
    elif sample.index is not None and sample.group_index is not None:
        session_id = f"agent-{md['instance_id']}-{sample.index}-{sample.group_index}"
    else:
        session_id = f"agent-{md['instance_id']}-{secrets.token_hex(8)}"
    sample.session_id = session_id
    # Adapter applies the request body OVER sampling_defaults; runtimes must
    # strip the agent's own sampling knobs to stay on-policy.
    state.adapter.open_session(
        session_id,
        sampling_defaults=_sampling_params(state.args, sampling_params),
        max_context_tokens=state.max_context_len,
    )
    return session_id


def _sampling_params(args, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    # Pin the knobs that must match training; the adapter fills the rest. These
    # become the session defaults the adapter applies UNDER each request.
    # ``overrides`` (slime's sampling_params) carries the per-call values:
    # the eval temperature/top_p AND ``max_new_tokens`` -- which slime sets to
    # rollout_max_response_len for train and eval_max_response_len for eval.
    # Forwarding it makes that the adapter's per-turn generation cap (still
    # further clamped to the remaining rollout_max_context_len budget); dropping
    # it would silently fall back to the adapter's hardcoded per-turn default.
    params = (
        {}
        if args is None
        else {
            k: v
            for k, v in (
                ("temperature", getattr(args, "rollout_temperature", None)),
                ("top_p", getattr(args, "rollout_top_p", None)),
                ("top_k", getattr(args, "rollout_top_k", None)),
                ("max_new_tokens", getattr(args, "rollout_max_response_len", None)),
            )
            if v is not None
        }
    )
    for k in ("temperature", "top_p", "top_k", "max_new_tokens"):
        if overrides and overrides.get(k) is not None:
            params[k] = overrides[k]
    return params


def _merge_samples(
    *,
    sample: Sample,
    state: _State,
    segments: list[TokenSegment],
    reward_result: RewardResult,
    elapsed_sec: float,
    instance_id: str,
):
    """Fan TokenSegments + reward out into Sample(s).

    A linear chain yields one Sample; routing through ``fan_out_sample_segments``
    stays correct if an agent later adds context-compaction "wipe" segments
    (reward split reward/K, siblings share ``rollout_id``).
    """
    if not segments:
        # Carry the agent's exit code + failure tail (set by the env in
        # reward_result.extra) onto the abort, so a zero-turn rollout self-
        # explains in the dump instead of needing tail-only Modal logs.
        diag = {k: reward_result.extra[k] for k in ("agent_exit_code", "agent_tail") if k in reward_result.extra}
        return _abort_result(sample, "adapter_session_empty", extra=diag)

    trajectory_metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "is_solved": reward_result.is_solved,
        "elapsed_sec": elapsed_sec,
        # Env-specific diagnostics (swe_gym: applied_cleanly; harbor: per-step).
        **reward_result.extra,
    }
    fanned = fan_out_sample_segments(
        sample, segments, reward_result.reward, state.tokenizer, metadata=trajectory_metadata
    )
    if not fanned:
        raise ValueError("fan-out produced no samples")
    logger.info(
        "[async_rl] %s: reward=%.2f solved=%s elapsed=%.1fs segments=%d extra=%s",
        instance_id,
        reward_result.reward,
        reward_result.is_solved,
        elapsed_sec,
        len(fanned),
        {k: v for k, v in reward_result.extra.items() if k != "agent_tail" and not isinstance(v, (list, dict))},
    )
    return fanned


# ---------------------------------------------------------------------------
# Main per-sample function (the --custom-generate-function-path target)
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Per-sample agent rollout with a wall-clock guard.

    Treats train and eval identically (run the agent + grade).
    """
    state = _State(args)
    # Row -> env dispatch (lazy import). A bad row aborts THAT sample; an env
    # module that won't import still raises loudly.
    try:
        env = load_env((sample.metadata or {}).get("task_type"))
        md = env.normalize_metadata(sample)
    except EnvMetadataError as e:
        return _abort_result(sample, str(e))
    except (ValueError, TypeError) as e:
        return _abort_result(sample, f"env_dispatch_failed:{type(e).__name__}:{e}")

    instance_id = md["instance_id"]
    # Enforced budgets (env defaults, or task.toml under override): recorded up
    # front so the dump self-reports them even when the sample aborts.
    sample.metadata = {
        **(sample.metadata or {}),
        "budgets": env.effective_budgets(
            md, agent_time_budget_sec=AGENT_TIME_BUDGET_SEC, eval_timeout_sec=AGENT_EVAL_TIMEOUT_SEC
        ),
    }
    session_id = _start_session(state, sample, md, sampling_params)
    t0 = time.time()
    try:
        reward_result: RewardResult = await env.rollout(
            md,
            runtime=state.runtime,
            session_id=session_id,
            adapter_url=state.adapter_url,
            agent_time_budget_sec=AGENT_TIME_BUDGET_SEC,
            eval_timeout_sec=AGENT_EVAL_TIMEOUT_SEC,
        )
        # Fold adapter per-turn stats into the env's phase timing.
        turn_stats = profiling.pop_session_stats(session_id)
        if turn_stats:
            reward_result.extra.setdefault("timing", {}).update(turn_stats)
        segments = await state.adapter.finish_session(session_id)
        return _merge_samples(
            sample=sample,
            state=state,
            segments=segments,
            reward_result=reward_result,
            elapsed_sec=time.time() - t0,
            instance_id=instance_id,
        )

    except SandboxBootTimeout as e:
        _attach_partial_timing(sample, session_id, t0)
        return _abort_result(sample, f"boot_timeout:{e.timeout_sec}s")
    except asyncio.CancelledError:
        # A stray CancelledError from inside the rollout (e.g. Modal's
        # synchronicity bridge) would crash the whole training step. Only a
        # genuine external cancel leaves the task cancelling -- re-raise then.
        if asyncio.current_task().cancelling():
            raise
        logger.error("[async_rl] %s: stray CancelledError; aborting sample", instance_id)
        _attach_partial_timing(sample, session_id, t0)
        return _abort_result(sample, "exception:CancelledError")
    except Exception as e:
        logger.error("[async_rl] %s: rollout failed: %s\n%s", instance_id, e, traceback.format_exc())
        _attach_partial_timing(sample, session_id, t0)
        return _abort_result(sample, f"exception:{type(e).__name__}")
    finally:
        # Close the sid before the next step's release_memory_occupation, else
        # stragglers race its idle assert.
        await state.adapter.finish_session(session_id)  # idempotent


def _attach_partial_timing(sample: Sample, session_id: str, t0: float) -> None:
    """On abort, keep accrued turn stats (distinguishes 'alive but slow' from
    'never dialed in')."""
    stats = profiling.pop_session_stats(session_id) or {}
    stats["elapsed_at_abort"] = round(time.time() - t0, 1)
    sample.metadata = {**(sample.metadata or {}), "timing": stats}


def _abort(sample: Sample, reason: str, extra: dict[str, Any] | None = None) -> Sample:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    # Shape-consistent with response_length: the train actor slices
    # rollout_log_probs for every sample, so a None here crashes the step.
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    # Mirror fan_out_sample_segments: build_dp_schedule groups by rollout_id, so
    # a None collapses all aborts into one group and drops below global_batch_size.
    sample.rollout_id = sample.index
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason, **(extra or {})}
    logger.warning("[async_rl] aborted: %s", reason)
    return sample


def _abort_result(sample: Sample, reason: str, extra: dict[str, Any] | None = None):
    """Uniform list shape for this (potentially fan-out) generate function."""
    return [_abort(sample, reason, extra)]
