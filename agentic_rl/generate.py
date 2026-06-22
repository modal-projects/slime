"""slime per-sample rollout hook (--custom-generate-function-path). One call = one episode:
boot the task sandbox, run the unmodified mini-swe harness through a token-recording model,
build a token-faithful training Sample, and grade it. Per-episode stats land in
sample.metadata['agentic'] for metrics.py; episode limits come from args (--custom-config-path).
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from . import prompts
from .grade import grade
from .model import RecordingModel
from .sandbox import Sandbox

logger = logging.getLogger("agentic_rl")

# Per-instance boot-failure tally. A task whose image won't build returns ABORTED and slime
# requeues the same group forever; after the cap we ship a masked reward-0 sample to drop it.
_boot_fails: dict[str, int] = {}

# Episodes are long I/O-bound chains; asyncio.to_thread caps at min(32, cpu+4) threads, which
# throttles in-flight episodes regardless of sglang_server_concurrency. Use a wide dedicated pool.
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
                logger.info("agentic episode thread pool: max_workers=%d", n)
    return _episode_pool


def _run_episode(
    task: dict, tokenizer, sampling_params, router_url: str, limits: dict, session_id: str
) -> tuple[float, RecordingModel, dict]:
    """Run stock mini-swe in a sandbox. Never raises; returns (reward, model, stats)."""
    from minisweagent.agents.default import DefaultAgent

    model = RecordingModel(
        tokenizer,
        sampling_params,
        router_url,
        prompts.OBSERVATION_TEMPLATE,
        session_id,
        query_timeout=limits["query_timeout"],
    )
    workdir = "/" + task["repo"].split("/")[1]
    patch, reward, exit_status, grade_time = None, 0.0, "none", 0.0
    sandbox = None
    t0 = time.perf_counter()
    try:
        # lifetime covers episode + grade: grading runs tests in a fresh sandbox, but a long agent
        # loop near episode_timeout would otherwise see this one reaped mid-run ("already shut down").
        sandbox = Sandbox(
            task["image_name"],
            cwd=workdir,
            lifetime=limits["episode_timeout"] + limits["grade_timeout"] + 300,
            exec_timeout=limits["exec_timeout"],
        )
        agent = DefaultAgent(
            model,
            sandbox,
            system_template=prompts.SYSTEM_TEMPLATE,
            instance_template=prompts.INSTANCE_TEMPLATE,
            step_limit=limits["max_steps"],
            cost_limit=0.0,
            wall_time_limit_seconds=limits["episode_timeout"],
        )
        exit_info = agent.run(task=task["problem_statement"])
        exit_status = (exit_info or {}).get("exit_status", "?")
        _, patch = sandbox.exec("git add -A && git diff --cached HEAD", cwd=workdir, timeout=120)
    except Exception:
        logger.exception("episode failed (instance=%s)", task.get("instance_id"))
    finally:
        if sandbox is not None:
            sandbox.terminate()  # grading runs in a fresh, clean sandbox

    if patch is not None and not model.aborted:
        g0 = time.perf_counter()
        try:
            reward = grade(task, patch, timeout=limits["grade_timeout"])
        except Exception:
            logger.exception("grading failed (instance=%s)", task.get("instance_id"))
        grade_time = time.perf_counter() - g0

    stats = {
        "turns": len(model.versions),
        "format_errors": model.n_format_errors,
        "exit_status": exit_status,
        "gen_time": round(model.gen_time, 1),
        "exec_time": round(sandbox.exec_time if sandbox is not None else 0.0, 1),
        "boot_time": round(sandbox.boot_time if sandbox is not None else 0.0, 1),
        "grade_time": round(grade_time, 1),
        "episode_time": round(time.perf_counter() - t0, 1),
        "output_tokens": sum(model.loss_mask),
        "response_tokens": len(model.tokens) - (model.prompt_len or len(model.tokens)),
        "gen_timestamp": time.time(),  # for sample-age (staleness) at train time
    }
    return reward, model, stats


def _ship_null(sample: Sample, tokenizer, problem_statement: str, reason: str) -> Sample:
    """A valid but fully-masked reward-0 sample for a permanently-failing task: ships to leave
    the buffer, contributes no gradient (slime zeros the loss mask via remove_sample)."""
    ptoks = tokenizer.encode(problem_statement or "", add_special_tokens=False)[:512]
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (ptoks[-1] if ptoks else 0)
    sample.tokens = ptoks + [eos]
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.weight_versions = []
    sample.reward = 0.0
    sample.status = Sample.Status.COMPLETED
    sample.remove_sample = True
    sample.metadata = {**sample.metadata, "agentic": {"exit_status": reason, "turns": 0, "gen_timestamp": time.time()}}
    return sample


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False):
    state = GenerateState(args)
    if state.aborted:
        sample.status = Sample.Status.ABORTED
        return sample

    task = dict(sample.metadata)
    task["problem_statement"] = sample.prompt if isinstance(sample.prompt, str) else task["problem_statement"]
    limits = {
        "max_steps": getattr(args, "agentic_max_steps", 20),
        "episode_timeout": getattr(args, "agentic_episode_timeout", 1800),
        "exec_timeout": getattr(args, "agentic_exec_timeout", 120),
        "grade_timeout": getattr(args, "agentic_grade_timeout", 1800),
        "query_timeout": getattr(args, "agentic_query_timeout", 600),  # per-turn cap; bounds hung generations
    }
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    session_id = sample.session_id or str(uuid.uuid4())  # pin an episode's turns to one worker (prefix cache)

    loop = asyncio.get_running_loop()
    reward, model, stats = await loop.run_in_executor(
        _episode_executor(args),
        _run_episode,
        task,
        state.tokenizer,
        state.sampling_params,
        router_url,
        limits,
        session_id,
    )

    key = sample.label or task.get("instance_id") or ""

    if model.aborted:  # weight update aborted a turn mid-flight → recycle
        sample.status = Sample.Status.ABORTED
        return sample

    # No usable trajectory (boot failed / every turn rolled back): retry a few times for transient
    # failures, then ship a masked sample so the group leaves the buffer instead of recycling forever.
    if model.prompt_len is None or len(model.tokens) <= model.prompt_len:
        _boot_fails[key] = _boot_fails.get(key, 0) + 1
        if _boot_fails[key] <= getattr(args, "agentic_max_boot_retries", 3):
            sample.status = Sample.Status.ABORTED
            return sample
        logger.warning("instance %s unusable after %d boot failures; dropping", key, _boot_fails[key])
        return _ship_null(sample, state.tokenizer, task.get("problem_statement", ""), "ImageUnusable")

    _boot_fails.pop(key, None)
    sample.tokens = model.tokens
    sample.response_length = len(model.tokens) - model.prompt_len
    sample.loss_mask = model.loss_mask[model.prompt_len :]
    sample.rollout_log_probs = model.logprobs[model.prompt_len :]
    sample.weight_versions = [v for v in model.versions if v is not None]
    sample.reward = reward
    sample.status = Sample.Status.COMPLETED
    sample.prefix_cache_info.cached_tokens = model.cached_tokens
    sample.prefix_cache_info.total_prompt_tokens = model.input_tokens
    sample.metadata = {**sample.metadata, "agentic": stats}
    return sample
