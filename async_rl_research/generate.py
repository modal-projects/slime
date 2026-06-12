"""Generic agentic-RL rollout entrypoint for slime (design A: HTTP adapter).

Wire-up::

    --custom-generate-function-path async_rl_research.generate.generate

This is the **agent- and task-agnostic** per-sample orchestrator. It owns the
parts that are identical for any in-sandbox agent on any task family, and
delegates everything else to two orthogonal collaborators:

    generate.py  (this file)   adapter/HTTP lifecycle + session management +
                               trajectory merge + abort/timeout isolation.
    agent/base.py              the AgentRuntime contract + shared launch and
                               provisioning machinery + the runtime registry
                               (load_runtime). One subclass per agent
                               framework (default: mini_swe_agent).
    env/base.py                the RolloutEnv contract + the env registry
                               (load_env). One subclass per task family --
                               row schema, sandbox boot/prep, agent-leg
                               sequencing, grading. env/swe_gym.py grades a
                               captured git diff in a CLEAN sandbox;
                               env/harbor.py verifies harbor tasks in-place
                               (multi-step aware). Rows pick their env via
                               metadata.task_type (absent -> swe_gym).

Topology (design A -- "in-sandbox subprocess + HTTP adapter"):

    host generate():
      1. _State (once/worker): build the runtime's adapter (an aiohttp app that
         speaks the agent's wire API and records exact SGLang tokens) and serve
         it on a bg thread; expose adapter_url = http://$SLIME_HEAD_HOST:$PORT.
      2. open an adapter session keyed by session_id.
      3. env.rollout(): boot the task sandbox, prep the workspace, let the
         runtime launch the agent inside it for each task step. The agent
         dials BACK to adapter_url for every model call; the adapter renders
         messages -> input_ids, calls SGLang /generate (return_logprob), and
         records (prompt_ids, output_ids, logprobs). The env grades the
         result into a RewardResult.
      4. finish_session() drains the recorded token segments; merge -> Sample.

Reward is computed inline (env.rollout) and written onto the sample, so
slime's default reward-model step is skipped (generate_and_rm only calls
async_rm when sample.reward is None).

The full contracts live on ``agent.base.AgentRuntime`` and
``env.base.RolloutEnv`` -- single sources of truth. This module only relies
on: ``runtime.adapter_cls`` (constructed as adapter_cls(tokenizer=,
sglang_url=, tool_parser=, reasoning_parser=)), ``env.normalize_metadata``,
and ``env.rollout``.

Env knobs
---------
    SLIME_HEAD_HOST        public IP sandboxes use to reach the adapter
                           (REQUIRED unless MODAL_EXPOSE_ADAPTER=1)
    MODAL_EXPOSE_ADAPTER   1 to publish the adapter through a modal.forward
                           tunnel (required on a Modal cluster: sandboxes are
                           network-isolated and can only dial a public URL)
    SHIM_BIND_HOST         0.0.0.0   adapter bind host on the head node
    SHIM_PORT              18002     adapter bind port
    ASYNC_RL_AGENT_RUNTIME which agent runtime to use: a registry short name
                           ("mini-swe"), "module:Class", or a module path
                           exposing RUNTIME (default "mini-swe"; see
                           agent.base.load_runtime)
    ASYNC_RL_AGENT_DRIVER  legacy alias for ASYNC_RL_AGENT_RUNTIME
    ASYNC_RL_TASK_ROOT     root dir that relative metadata.task_path values
                           resolve against (harbor env; the converter's
                           --out-dir on the slime-data volume)
    AGENT_TIME_BUDGET_SEC  1800      total agent wallclock budget per sample
                                     (multi-step episodes share it)
    AGENT_EVAL_TIMEOUT_SEC 600       wallclock cap per grading command
    AGENT_GENERATE_GUARD_SEC         full generate() guard; default budget+eval+180
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

logger = logging.getLogger(__name__)


AGENT_TIME_BUDGET_SEC = int(os.environ.get("AGENT_TIME_BUDGET_SEC", "1800"))
AGENT_EVAL_TIMEOUT_SEC = int(os.environ.get("AGENT_EVAL_TIMEOUT_SEC", "600"))
# Wall-clock guard for the entire generate() call. When exceeded, the in-flight
# sample is aborted (`wall_clock_timeout`) and the rest of the rollout
# continues -- isolates one hung trajectory from the whole training step.
AGENT_GENERATE_GUARD_SEC = int(os.environ.get("AGENT_GENERATE_GUARD_SEC", "0") or 0) or (
    AGENT_TIME_BUDGET_SEC + AGENT_EVAL_TIMEOUT_SEC + 180
)
SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "18002"))
# On a Modal cluster the head is itself a Modal container and the sandboxes are
# network-isolated (no i6pn/cluster routing), so they can only reach the
# adapter via a public modal.forward tunnel rather than a private cluster IP.
MODAL_EXPOSE_ADAPTER = os.environ.get("MODAL_EXPOSE_ADAPTER", "0").strip().lower() in ("1", "true", "yes")


def _load_runtime(args) -> AgentRuntime:
    """Resolve + instantiate the agent runtime (env > arg > registry default).

    Validation is eager (load_runtime / AgentRuntime.__init_subclass__), so a
    misdeclared runtime fails the worker boot loudly instead of
    AttributeError-ing mid-sample.
    """
    spec = (
        os.environ.get("ASYNC_RL_AGENT_RUNTIME")
        or os.environ.get("ASYNC_RL_AGENT_DRIVER")  # legacy alias
        or getattr(args, "agent_runtime", None)
        or getattr(args, "agent_driver", None)
    )
    return load_runtime(spec)


# ---------------------------------------------------------------------------
# Singleton: tokenizer + runtime-selected adapter + background HTTP server.
# SingletonMeta keys per class, so there is exactly one runtime + adapter +
# server per rollout worker process; trajectories stay isolated by session_id.
# ---------------------------------------------------------------------------
class _State(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.args = args
        self.runtime = _load_runtime(args)
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        # Adapter reuses the SGLang parsers configured for the served model so
        # tool-call bash / reasoning are parsed correctly (e.g.
        # --sglang-tool-call-parser qwen3_coder, --sglang-reasoning-parser qwen3).
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
        # Per-turn timing by session (bearer token == session_id); must be
        # installed before the app starts. Feeds metadata["timing"] below.
        profiling.install(self.adapter.app)
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the adapter's fire-and-forget /abort_request. Without
        # it a cancelled client leaves an inflight sglang /generate that races
        # the next release_memory_occupation and trips sglang's idle assertion.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=SHIM_BIND_HOST,
            port=SHIM_PORT,
            thread_name="agent-adapter",
            runner_kwargs={"handler_cancellation": True},
        )
        # Everything past the bind can still fail (e.g. modal.forward). The
        # server thread is a daemon that holds SHIM_PORT for the process
        # lifetime, and SingletonMeta only caches on a clean __init__, so a
        # failure here would orphan the thread and make every later _State()
        # collide on the port. Tear the server down on any failure so the bind
        # is releasable and the next sample can retry.
        try:
            # Base URL (no /v1). The runtime appends whatever its wire API needs.
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

        On a Modal cluster the adapter must be reached through a per-process
        ``modal.forward`` tunnel (the head is a Modal container; sandboxes are
        network-isolated). Each rollout-worker process is its own ``_State``
        singleton, so it opens its OWN tunnel -- a single static env can't cover
        multiple data-parallel workers. The tunnel context is held on ``self``
        for the process lifetime; litellm speaks HTTPS, so the encrypted
        ``https://<random>.modal.host`` URL needs no port juggling.
        """
        if not MODAL_EXPOSE_ADAPTER:
            return f"http://{public_host}:{self.app_handle.port}"

        import modal

        # Blocking context manager (we're in sync __init__); never exited --
        # the process owns the tunnel until it dies.
        self._tunnel_cm = modal.forward(self.app_handle.port)
        tunnel = self._tunnel_cm.__enter__()
        logger.info("[async_rl] modal.forward tunnel for adapter port %d -> %s", self.app_handle.port, tunnel.url)
        return tunnel.url


# ---------------------------------------------------------------------------
# Trajectory -> Sample
# ---------------------------------------------------------------------------
def _start_session(state: _State, sample: Sample, md: dict[str, Any]) -> str:
    """Register the adapter session BEFORE the agent starts.

    The in-sandbox agent sends ``session_id`` as its auth/bearer token so the
    adapter groups all of its turns under one chain.
    """
    if sample.session_id:
        session_id = sample.session_id
    elif sample.index is not None and sample.group_index is not None:
        session_id = f"agent-{md['instance_id']}-{sample.index}-{sample.group_index}"
    else:
        session_id = f"agent-{md['instance_id']}-{secrets.token_hex(8)}"
    sample.session_id = session_id
    # sampling_defaults are the rollout's baseline, but the adapter applies the
    # request body OVER them (adapters/common._sampling_params): an agent that
    # sends its own temperature/top_p would silently override training's.
    # Runtimes must strip sampling knobs from agent requests to stay on-policy.
    state.adapter.open_session(
        session_id,
        sampling_defaults=_sampling_params(state.args),
        max_context_tokens=state.max_context_len,
    )
    return session_id


def _sampling_params(args) -> dict[str, Any]:
    # Kept tiny on purpose: the adapter fills the rest of its defaults. We only
    # pin the knobs that must match training. Extend as needed.
    if args is None:
        return {}
    return {
        k: v
        for k, v in (
            ("temperature", getattr(args, "rollout_temperature", None)),
            ("top_p", getattr(args, "rollout_top_p", None)),
            ("top_k", getattr(args, "rollout_top_k", None)),
        )
        if v is not None
    }


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

    A single linear agent chain yields one ("final") segment -> K=1 -> one
    Sample. Routing through ``fan_out_sample_segments`` (which handles K==1)
    keeps it correct if an agent later adds context compaction ("wipe"
    segments): reward is split reward/K and siblings share ``rollout_id`` so
    the per-rollout loss reducer counts the trajectory once.
    """
    if not segments:
        return _abort_result(sample, "adapter_session_empty")

    trajectory_metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "is_solved": reward_result.is_solved,
        "elapsed_sec": elapsed_sec,
        # Env-specific diagnostics (swe_gym: applied_cleanly; harbor: per-step
        # rewards). Keys are env-namespaced or unambiguous by construction.
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
        {k: v for k, v in reward_result.extra.items() if not isinstance(v, (list, dict))},
    )
    return fanned


# ---------------------------------------------------------------------------
# Main per-sample function (the --custom-generate-function-path target)
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Per-sample agent rollout with a wall-clock guard.

    Accepts ``evaluation`` (slime passes it when present in the signature) but
    treats train and eval identically -- running the agent + grading the
    result is what eval wants too.
    """
    state = _State(args)
    # Row -> env dispatch. load_env imports the env module lazily, so this
    # module stays importable on nodes that never boot a sandbox. A bad row
    # (unknown task_type, unusable metadata) aborts THAT sample; systemic
    # failures (an env module that doesn't import) still raise loudly.
    try:
        env = load_env((sample.metadata or {}).get("task_type"))
        md = env.normalize_metadata(sample)
    except EnvMetadataError as e:
        return _abort_result(sample, str(e))
    except (ValueError, TypeError) as e:
        return _abort_result(sample, f"env_dispatch_failed:{type(e).__name__}:{e}")

    instance_id = md["instance_id"]
    session_id = _start_session(state, sample, md)
    t0 = time.time()
    try:
        async with asyncio.timeout(AGENT_GENERATE_GUARD_SEC):
            reward_result: RewardResult = await env.rollout(
                md,
                runtime=state.runtime,
                session_id=session_id,
                adapter_url=state.adapter_url,
                agent_time_budget_sec=AGENT_TIME_BUDGET_SEC,
                eval_timeout_sec=AGENT_EVAL_TIMEOUT_SEC,
            )
            # Fold the adapter's per-turn stats into the env's phase timing
            # (one "timing" dict per sample -> dumps -> profile.py).
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

    except asyncio.TimeoutError:
        _log_timeout_diagnostic(t0)
        return _abort_result(sample, "wall_clock_timeout")
    except Exception as e:
        logger.error("[async_rl] %s: rollout failed: %s\n%s", instance_id, e, traceback.format_exc())
        return _abort_result(sample, f"exception:{type(e).__name__}")
    finally:
        # Close the sid before the next train step's release_memory_occupation;
        # stragglers from this trajectory would otherwise race its idle assert.
        await state.adapter.finish_session(session_id)  # idempotent


def _log_timeout_diagnostic(t0: float) -> None:
    """Dump pending-task names when the wall-clock guard fires. Never crashes."""
    try:
        elapsed = time.time() - t0
        pending = [t for t in asyncio.all_tasks() if not t.done()]
        stuck = []
        for t in pending[:5]:
            coro = getattr(t, "_coro", None)
            stuck.append(getattr(coro, "__qualname__", repr(coro)))
        logger.warning(
            "[async_rl] generate() wall_clock_timeout after %.1fs (guard=%ds); %d tasks pending; stuck: %s",
            elapsed,
            AGENT_GENERATE_GUARD_SEC,
            len(pending),
            stuck,
        )
    except Exception:  # pragma: no cover
        pass


def _abort(sample: Sample, reason: str) -> Sample:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    # Per-token fields must stay shape-consistent with response_length: when
    # any sample in the batch carries rollout_log_probs, the train actor
    # slices it for EVERY sample (actor._get_rollout_data), and a None here
    # crashes the whole train step. loss_mask is 0 so the value never trains.
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    # Mirror fan_out_sample_segments' convention (rollout_id = sample.index):
    # build_dp_schedule groups samples by rollout_id, so a None here collapses
    # every aborted sample in the batch into ONE rollout group and the unique
    # rollout count drops below global_batch_size, crashing the train step.
    sample.rollout_id = sample.index
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[async_rl] aborted: %s", reason)
    return sample


def _abort_result(sample: Sample, reason: str):
    """Uniform list shape for this (potentially fan-out) generate function."""
    return [_abort(sample, reason)]
