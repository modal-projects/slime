"""slime per-sample rollout hook (--custom-generate-function-path). One call = one episode:
boot the task sandbox, run the unmodified mini-swe harness through a token-recording model,
build a token-faithful training Sample, and grade it. Per-episode stats land in
sample.metadata['agentic'] for metrics.py; episode limits come from args (--custom-config-path).
"""

import asyncio
import collections
import concurrent.futures
import json
import logging
import random
import threading
import time
import uuid

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from . import prompts
from .grade import grade_detailed
from .model import RecordingModel
from .sandbox import Sandbox

logger = logging.getLogger("agentic_rl")

# Per-instance boot-failure tally. A task whose image won't build returns ABORTED and slime
# requeues the same group forever; after the cap we ship a masked reward-0 sample to drop it.
_boot_fails: dict[str, int] = {}

# A small ring of recent transcripts, drained by metrics.log_rollout_data, to inspect trajectory
# quality offline. Bounded: last 8, each truncated (~4KB).
_recent_trajectories: collections.deque = collections.deque(maxlen=8)


def _capture_transcript(agent, task: dict, reward: float, stats: dict) -> None:
    try:
        text = "\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in agent.messages)
        if len(text) > 4000:
            text = text[:2000] + "\n...[truncated]...\n" + text[-2000:]
        _recent_trajectories.append(
            {
                "instance": task.get("instance_id", "?"),
                "reward": reward,
                "turns": stats.get("turns", 0),
                "exit_status": stats.get("exit_status", "?"),
                "transcript": text,
            }
        )
    except Exception:
        pass


def _overlong_penalty(total_len: int, args) -> float:
    """Soft overlong penalty as a CONTEXT-CAP GUARD (DAPO ramp shape): 0 until the trajectory nears the
    served window, linear to -1 between (l_max - l_cache) and l_max, clamped beyond. Keyed on total_length
    — the quantity that actually hits the context cap and ends the episode (ContextExceeded), NOT generated
    tokens — so a long-but-fine episode well inside the budget is untouched. The caller floors this for
    CORRECT episodes so a solve is never zeroed. Disabled when agentic_overlong_max is 0/unset."""
    l_max = getattr(args, "agentic_overlong_max", 0) or 0
    if not l_max:
        return 0.0
    l_cache = getattr(args, "agentic_overlong_cache", 40960)
    if total_len <= l_max - l_cache:
        return 0.0
    if total_len >= l_max:
        return -1.0
    return (l_max - l_cache - total_len) / l_cache


def _diag_dump(task: dict, patch: str, reward: float, exit_status: str, reward_source: str, detail: dict) -> None:
    """Diagnostic only (--agentic-diag-dump): emit one JSON log line per episode so the submitted
    patch + grading outcome can be grepped from the run log and zero-reward episodes classified as
    real-fix-rejected (harness leak) vs wrong-patch (true difficulty). Off in production."""
    try:
        out = detail.get("output", "")
        rec = {
            "instance": task.get("instance_id", "?"),
            "reward": reward,
            "exit_status": exit_status,
            "reward_source": reward_source,
            "n_required": len(detail.get("required", [])),
            "n_passed": len(detail.get("passed", [])),
            "missing": detail.get("missing", [])[:8],
            "patch": patch[:6000],
            "grade_head": out[:1200],  # apply result (set -e aborts here on apply failure)
            "grade_tail": out[-1800:],  # test summary when apply succeeded
        }
        logger.info("AGENTIC_DIAG %s", json.dumps(rec))
    except Exception:
        pass


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

    # Cold-start ramp: jitter each episode's start so the thundering herd of sandbox-boots + first queries
    # doesn't burst-overwhelm Modal's sandbox control plane (exec HTTPErrors ~1024 concurrent) or the sgl-
    # router (503s). Spreading the start over a window lets us run HIGH steady-state concurrency instead of
    # retreating to a low cap (GLM's fix). Held inside the sglang semaphore, so it also throttles the burst.
    if (ramp := limits.get("ramp_window", 0.0)) > 0:
        time.sleep(random.uniform(0.0, ramp))

    model = RecordingModel(
        tokenizer,
        sampling_params,
        router_url,
        prompts.OBSERVATION_TEMPLATE,
        session_id,
        query_timeout=limits["query_timeout"],
        max_context_len=limits["max_context_len"],
        action_format=limits["action_format"],
    )
    is_r2e = "expected_output_json" in task  # R2E-Gym task (vs SWE-rebench)
    workdir = "/testbed" if is_r2e else "/" + task["repo"].split("/")[1]
    patch, reward, solved, exit_status, grade_time, reward_source = None, 0.0, 0.0, "none", 0.0, "none"
    sandbox = None
    agent = None
    t0 = time.perf_counter()
    try:
        # lifetime must cover the full agent run: the wall limit (episode_timeout) is checked BETWEEN steps,
        # so a slow FINAL turn can run another query_timeout past it → max episode ≈ episode_timeout +
        # query_timeout. Grading runs in its own fresh sandbox, so it's not this sandbox's concern. Using
        # grade_timeout (600) here under-sized it, reaping long episodes mid-final-turn ("already shut down").
        sandbox = Sandbox(
            task["image_name"],
            cwd=workdir,
            lifetime=limits["episode_timeout"] + limits["query_timeout"] + 300,
            exec_timeout=limits["exec_timeout"],
        )
        if is_r2e:
            # Remove the held-out tests + runner (/r2e_tests, run_tests.sh) FIRST — anti-reward-hacking (they're
            # world-readable in the raw image), and it MUST precede the commit: otherwise `git add -A` tracks
            # run_tests.sh, and the agent's later diff would carry its DELETION → grade applies that → the runner
            # vanishes → 0 tests run → false reward 0. Then commit R2E's setup-dirt as the base so the agent's
            # submission is a clean diff of ITS changes only (grade re-commits the same base symmetrically; it
            # keeps its own fresh copy of the tests/runner for scoring).
            sandbox.exec(
                "rm -rf /r2e_tests; rm -f run_tests.sh; "
                "git config user.email r2e@local && git config user.name r2e && "
                "git add -A && git commit -q -m r2e-base || true",
                cwd=workdir,
                timeout=120,
            )
        agent = DefaultAgent(
            model,
            sandbox,
            system_template=prompts.SYSTEM_TEMPLATE,
            instance_template=prompts.INSTANCE_TEMPLATE,
            step_limit=limits["max_steps"],
            cost_limit=0.0,
            wall_time_limit_seconds=limits["episode_timeout"],
            # mini-swe defaults to 3 consecutive format errors → episode death. Qwen's occasional tool-call
            # miss shouldn't kill a long episode; raise it well above the per-episode miss rate (Tmax uses 64).
            max_consecutive_format_errors=limits["max_format_errors"],
        )
        exit_info = agent.run(task=task["problem_statement"]) or {}
        exit_status = exit_info.get("exit_status", "?")
        patch = exit_info.get("submission") or ""  # agent's curated source-only patch (on Submitted)
        reward_source = "submission" if patch.strip() else "fallback_diff"
        if not patch.strip():  # no clean submission → grade the working-tree diff (grade.py resets test files)
            _, patch = sandbox.exec("git add -A && git diff --cached HEAD", cwd=workdir, timeout=120)
    except Exception:
        logger.exception("episode failed (instance=%s)", task.get("instance_id"))
    finally:
        if sandbox is not None:
            sandbox.terminate()  # grading runs in a fresh, clean sandbox

    if patch is not None and not model.aborted:
        g0 = time.perf_counter()
        try:
            detail = grade_detailed(task, patch, timeout=limits["grade_timeout"])
            # TRAIN REWARD only on SUBMITTED episodes. Giving the non-submit git-diff fallback dense credit
            # reinforced rambling (the gradient learned "ramble for partial credit" over "submit-and-solve"),
            # so non-submit → 0 enforces submit discipline; dynamic sampling then filters the all-fail groups
            # so every kept group's winner is a real submitter. solved (metric) stays the strict working-tree
            # binary for diagnostics (the capability-vs-submission gap).
            reward = detail["dense"] if reward_source == "submission" else 0.0
            solved = detail["reward"]  # strict binary → solved_frac (working-tree; informative)
            if limits.get("diag_dump"):
                _diag_dump(task, patch, solved, exit_status, reward_source, detail)
        except Exception:
            logger.exception("grading failed (instance=%s)", task.get("instance_id"))
        grade_time = time.perf_counter() - g0

    stats = {
        "turns": len(model.versions),  # productive (non-format-error) turns
        "hit_step_cap": float(len(model.versions) >= limits["max_steps"]),  # ran out of turns (vs other exits)
        "n_calls": model.n_calls,  # total model calls; n_calls - turns = the format-error tax
        "format_errors": model.n_format_errors,
        "length_truncations": model.n_length_truncations,  # turns the model's output was cut at the per-turn cap
        "exit_status": exit_status,
        "solved": float(solved),  # strict binary (all required pass) → solved_frac; reward is dense
        "reward_source": reward_source,  # "submission" (curated) vs "fallback_diff" (git add -A) vs "none"
        "gen_time": round(model.gen_time, 1),
        "exec_time": round(sandbox.exec_time if sandbox is not None else 0.0, 1),
        "boot_time": round(sandbox.boot_time if sandbox is not None else 0.0, 1),
        "grade_time": round(grade_time, 1),
        "episode_time": round(time.perf_counter() - t0, 1),
        "output_tokens": sum(model.loss_mask),
        "response_tokens": len(model.tokens) - (model.prompt_len or len(model.tokens)),
        "reasoning_tokens": model.reasoning_tokens,
        "total_length": len(model.tokens),  # full trajectory length → context utilization vs the 128k window
        "prefix_cache_hit": round(model.cached_tokens / model.input_tokens, 3) if model.input_tokens else 0.0,
        "gen_timestamp": time.time(),  # for sample-age (staleness) at train time
    }
    if agent is not None:
        _capture_transcript(agent, task, reward, stats)
    return reward, model, stats


def _ship_masked_sample(sample: Sample, tokenizer, problem_statement: str, reason: str) -> Sample:
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


def _recycle_or_drop(args, sample, tokenizer, problem_statement, reason):
    """A status=ABORTED sample means 'requeue me'. The fully-async orchestrator honors that; the SYNC
    generate_rollout ships EVERY returned sample straight to training, where a status=ABORTED sample's
    reward=None crashes _post_process_rewards (torch.tensor(None)). So: async → ABORTED (recycle);
    sync → a masked reward-0 sample (valid reward, remove_sample=True so it contributes no gradient)."""
    if "fully_async" in getattr(args, "rollout_function_path", ""):
        sample.status = Sample.Status.ABORTED
        return sample
    return _ship_masked_sample(sample, tokenizer, problem_statement, reason)


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False):
    state = GenerateState(args)
    if state.aborted:
        return _recycle_or_drop(
            args, sample, state.tokenizer, sample.prompt if isinstance(sample.prompt, str) else "", "StateAborted"
        )

    task = dict(sample.metadata)
    task["problem_statement"] = sample.prompt if isinstance(sample.prompt, str) else task["problem_statement"]
    limits = {
        "max_steps": getattr(args, "agentic_max_steps", 20),
        "episode_timeout": getattr(args, "agentic_episode_timeout", 1800),
        "exec_timeout": getattr(args, "agentic_exec_timeout", 120),
        "grade_timeout": getattr(args, "agentic_grade_timeout", 1800),
        "query_timeout": getattr(args, "agentic_query_timeout", 600),  # per-turn cap; bounds hung generations
        "max_context_len": getattr(args, "rollout_max_context_len", 131072),  # served window; per-turn gen cap
        "max_format_errors": getattr(args, "agentic_max_format_errors", 30),  # consecutive before episode dies
        "action_format": getattr(args, "agentic_action_format", None),  # "tool_call"|"bash_block"; None=per-model
        "diag_dump": getattr(args, "agentic_diag_dump", False),  # dump per-episode patch+grade detail (diagnostic)
        "ramp_window": getattr(args, "agentic_ramp_window", 0.0),  # cold-start stagger (s); spread the herd
    }
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    session_id = sample.session_id or str(uuid.uuid4())  # pin an episode's turns to one worker (prefix cache)

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        _episode_executor(args),
        _run_episode,
        task,
        state.tokenizer,
        state.sampling_params,
        router_url,
        limits,
        session_id,
    )
    # Hard wall-cap on the whole episode. mini-swe's wall_time_limit only fires between turns, so a single
    # slow/streaming generation on a congested engine can overshoot it by hours, poisoning the batch with
    # extreme staleness. Cap it here from the rollout side and recycle; the orphan thread unwinds on its own.
    # A legit episode runs to wall_limit + ONE final turn (bounded by query_timeout) + grade, so the cap must
    # cover all three or it preempts legit episodes mid-final-turn (false-negatives). +120 slack only.
    hard_cap = limits["episode_timeout"] + limits["query_timeout"] + limits["grade_timeout"] + 120
    try:
        reward, model, stats = await asyncio.wait_for(fut, timeout=hard_cap)
    except asyncio.TimeoutError:
        logger.warning(
            "episode exceeded hard cap %ds (instance=%s); recycling/dropping", hard_cap, task.get("instance_id")
        )
        return _recycle_or_drop(args, sample, state.tokenizer, task.get("problem_statement", ""), "HardCapTimeout")

    key = sample.label or task.get("instance_id") or ""

    if model.aborted:  # weight update aborted a turn mid-flight → recycle (async) / drop (sync)
        return _recycle_or_drop(
            args, sample, state.tokenizer, task.get("problem_statement", ""), "WeightUpdateAborted"
        )

    # No usable trajectory (boot failed / every turn rolled back): retry a few times for transient
    # failures, then ship a masked sample so the group leaves the buffer instead of recycling forever.
    if model.prompt_len is None or len(model.tokens) <= model.prompt_len:
        _boot_fails[key] = _boot_fails.get(key, 0) + 1
        if _boot_fails[key] <= getattr(args, "agentic_max_boot_retries", 3):
            return _recycle_or_drop(args, sample, state.tokenizer, task.get("problem_statement", ""), "BootRetry")
        logger.warning("instance %s unusable after %d boot failures; dropping", key, _boot_fails[key])
        return _ship_masked_sample(sample, state.tokenizer, task.get("problem_statement", ""), "ImageUnusable")

    _boot_fails.pop(key, None)
    sample.tokens = model.tokens
    sample.response_length = len(model.tokens) - model.prompt_len
    sample.loss_mask = model.loss_mask[model.prompt_len :]
    sample.rollout_log_probs = model.logprobs[model.prompt_len :]
    sample.weight_versions = [v for v in model.versions if v is not None]
    # Context-cap guard: penalize only trajectories nearing the 128k window (keyed on total_length),
    # and never zero a CORRECT episode — floor keeps any full solve above any partial. reward is the dense
    # baseline-relative signal; stats["solved"] is the strict 0/1 (set in _run_episode) for solved_frac.
    penalty = _overlong_penalty(stats["total_length"], args)
    shaped = reward + penalty
    if stats.get("solved", 0.0) >= 1.0:
        shaped = max(shaped, getattr(args, "agentic_overlong_correct_floor", 0.5))
    stats["overlong_penalty"] = round(shaped - reward, 4)
    sample.reward = shaped
    sample.status = Sample.Status.COMPLETED
    sample.prefix_cache_info.cached_tokens = model.cached_tokens
    sample.prefix_cache_info.total_prompt_tokens = model.input_tokens
    sample.metadata = {**sample.metadata, "agentic": stats}
    return sample
