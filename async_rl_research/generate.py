"""Generic agentic-RL rollout entrypoint for slime (design A: HTTP adapter).

Wire-up::

    --custom-generate-function-path async_rl_research.generate.generate

This is the **agent-agnostic** per-sample orchestrator. It owns the parts that
are identical for any in-sandbox agent and delegates the agent-specific and
sandbox-specific work to two collaborators:

    generate.py  (this file)   the rollout recipe + adapter/HTTP lifecycle +
                               trajectory merge + abort/timeout isolation.
    agent/<driver>.py          everything specific to one agent (which adapter,
                               how to launch it in the sandbox, its prompt /
                               env wiring). Default: agent/mini_swe_agent.py.
    sandbox.py                 sandbox backend + SWE eval (boot / git_diff /
                               evaluate). NOT built yet -- contract below.

Topology (design A -- "in-sandbox subprocess + HTTP adapter"):

    host generate():
      1. _State (once/worker): build the driver's adapter (an aiohttp app that
         speaks the agent's wire API and records exact SGLang tokens) and serve
         it on a bg thread; expose adapter_url = http://$SLIME_HEAD_HOST:$PORT.
      2. open an adapter session keyed by session_id.
      3. boot a sandbox; the driver launches the agent inside it as a
         subprocess. The agent dials BACK to adapter_url for every model call;
         the adapter renders messages -> input_ids, calls SGLang /generate
         (return_logprob), and records (prompt_ids, output_ids, logprobs).
      4. capture git diff; score it in a CLEAN sandbox (no test-cheating).
      5. finish_session() drains the recorded token segments; merge -> Sample.

Reward is computed inline (sandbox.evaluate) and written onto the sample, so
slime's default reward-model step is skipped (generate_and_rm only calls
async_rm when sample.reward is None).

----------------------------------------------------------------------------
Driver contract (a driver is a *module*; default async_rl_research.agent.mini_swe_agent)
----------------------------------------------------------------------------
    ADAPTER_CLS : type[BaseAdapter]
        The slime adapter class for this agent's wire protocol
        (OpenAIAdapter for mini-swe-agent / litellm, AnthropicAdapter for
        claude-code). Constructed as
        ADAPTER_CLS(tokenizer=, sglang_url=, tool_parser=, reasoning_parser=).

    async def run_agent(sb, *, md, session_id, adapter_url, time_budget_sec) -> int
        Provision + launch the agent inside the already-booted sandbox `sb`,
        wait for it to finish, return an exit code. The agent must send
        `session_id` as its auth/bearer so the adapter groups its turns, and
        must target `adapter_url` for model calls. `md` is the normalized
        dataset row (see _metadata).

----------------------------------------------------------------------------
sandbox.py contract (async_rl_research.sandbox -- NOT built yet)
----------------------------------------------------------------------------
    @asynccontextmanager
    async def boot_agent_sandbox(image: str) -> AsyncIterator[Sandbox]: ...

    async def git_diff(sb, workdir: str) -> str: ...

    async def evaluate(*, image, workdir, diff_text, swepro=None, eval_cmd=None,
                       pre_commands=None, timeout_sec=600) -> tuple[float, bool, bool]:
        # (reward, solved, applied_cleanly); applies diff in a CLEAN sandbox.

Env knobs
---------
    SLIME_HEAD_HOST        public IP sandboxes use to reach the adapter (REQUIRED)
    SHIM_BIND_HOST         0.0.0.0   adapter bind host on the head node
    SHIM_PORT              18002     adapter bind port
    ASYNC_RL_AGENT_DRIVER  dotted module path of the driver
                           (default async_rl_research.agent.mini_swe_agent)
    AGENT_TIME_BUDGET_SEC  1800      wallclock budget for one agent run
    AGENT_EVAL_TIMEOUT_SEC 600       wallclock cap on the evaluator sandbox
    AGENT_GENERATE_GUARD_SEC         full generate() guard; default budget+eval+180
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import logging
import os
import secrets
import time
import traceback
from dataclasses import dataclass
from typing import Any

from slime.agent.trajectory import TokenSegment, fan_out_sample_segments
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .aiohttp_threaded import run_app_in_thread

logger = logging.getLogger(__name__)


DEFAULT_DRIVER = "async_rl_research.agent.mini_swe_agent"

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


def _load_driver(args):
    """Resolve the agent driver *module* (env > arg > default)."""
    path = (
        os.environ.get("ASYNC_RL_AGENT_DRIVER")
        or getattr(args, "agent_driver", None)
        or DEFAULT_DRIVER
    )
    return importlib.import_module(path)


# ---------------------------------------------------------------------------
# Singleton: tokenizer + driver-selected adapter + background HTTP server.
# SingletonMeta keys per class, so there is exactly one adapter + server per
# rollout worker process; trajectories stay isolated by session_id.
# ---------------------------------------------------------------------------
class _State(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.args = args
        self.driver = _load_driver(args)
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        # Adapter reuses the SGLang parsers configured for the served model so
        # tool-call bash / reasoning are parsed correctly (e.g.
        # --sglang-tool-call-parser qwen3_coder, --sglang-reasoning-parser qwen3).
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None

        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        public_host = os.environ.get("SLIME_HEAD_HOST")
        if not public_host:
            raise RuntimeError(
                "SLIME_HEAD_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for the reverse-connection to the adapter. "
                "Without it the in-sandbox agent cannot dial back and the "
                "rollout will silently abort."
            )

        self.adapter = self.driver.ADAPTER_CLS(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
        )
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
        # Base URL (no /v1). The driver appends whatever its wire API needs.
        self.adapter_url = f"http://{public_host}:{self.app_handle.port}"
        logger.info(
            "[async_rl] driver=%s adapter=%s tokenizer=%s tool_parser=%s reasoning_parser=%s",
            self.driver.__name__,
            self.adapter_url,
            args.hf_checkpoint,
            self.tool_parser,
            self.reasoning_parser,
        )


# ---------------------------------------------------------------------------
# Trajectory -> Sample
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardResult:
    reward: float
    is_solved: bool
    applied_cleanly: bool


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
    # sampling_defaults win over anything the agent sends, keeping the rollout
    # on-policy (the adapter merges request body OVER these defaults).
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
        "applied_cleanly": reward_result.applied_cleanly,
        "elapsed_sec": elapsed_sec,
    }
    fanned = fan_out_sample_segments(
        sample, segments, reward_result.reward, state.tokenizer, metadata=trajectory_metadata
    )
    if not fanned:
        raise ValueError("fan-out produced no samples")
    logger.info(
        "[async_rl] %s: reward=%.2f solved=%s applied=%s elapsed=%.1fs segments=%d",
        instance_id,
        reward_result.reward,
        reward_result.is_solved,
        reward_result.applied_cleanly,
        elapsed_sec,
        len(fanned),
    )
    return fanned


# ---------------------------------------------------------------------------
# Main per-sample function (the --custom-generate-function-path target)
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Per-sample agent rollout with a wall-clock guard.

    Accepts ``evaluation`` (slime passes it when present in the signature) but
    treats train and eval identically -- running the agent + grading its diff
    is what eval wants too.
    """
    # `sandbox` is intentionally lazy-imported: it is not built yet (its
    # contract is documented above). Everything else in this module imports and
    # runs without it.
    from . import sandbox

    state = _State(args)
    md = _metadata(sample)
    if not md["image"] or not md["workdir"]:
        return _abort_result(sample, "missing_image_or_workdir")

    instance_id = md["instance_id"]
    session_id = _start_session(state, sample, md)
    t0 = time.time()
    try:
        async with asyncio.timeout(AGENT_GENERATE_GUARD_SEC):
            async with sandbox.boot_agent_sandbox(md["image"]) as sb:
                await state.driver.run_agent(
                    sb,
                    md=md,
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=AGENT_TIME_BUDGET_SEC,
                )
                diff_text = await sandbox.git_diff(sb, md["workdir"])

            reward, is_solved, applied_cleanly = await sandbox.evaluate(
                image=md["image"],
                workdir=md["workdir"],
                diff_text=diff_text,
                swepro=md["swepro"],
                eval_cmd=md["eval_cmd"],
                pre_commands=md["pre_commands"],
                timeout_sec=AGENT_EVAL_TIMEOUT_SEC,
            )
            reward_result = RewardResult(
                reward=float(reward), is_solved=bool(is_solved), applied_cleanly=bool(applied_cleanly)
            )
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


# ---------------------------------------------------------------------------
# Dataset-row normalization (agent-agnostic; SWE schema shared with the example)
# ---------------------------------------------------------------------------
def _wrap_f2p_script(script: str | None) -> str | None:
    if not script:
        return None
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return f"echo {b64} | base64 -d > /tmp/slime_f2p.py && python /tmp/slime_f2p.py"


def _metadata(sample: Sample) -> dict[str, Any]:
    """Normalize the two dataset schemas (flat vs ``remote_env_info``)."""
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    return {
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "swepro": m.get("swepro"),
        "eval_cmd": m.get("eval_cmd") or _wrap_f2p_script(rem.get("f2p_script")),
        "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
    }


def _coerce_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _abort(sample: Sample, reason: str) -> Sample:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[async_rl] aborted: %s", reason)
    return sample


def _abort_result(sample: Sample, reason: str):
    """Uniform list shape for this (potentially fan-out) generate function."""
    return [_abort(sample, reason)]
