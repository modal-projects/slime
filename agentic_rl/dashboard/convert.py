#!/usr/bin/env python3
"""Convert a slime rollout debug dump (``rollout_N.pt``) to dashboard JSON.

Slices each sample's decoded merged trajectory (a Qwen3 chat-template stream)
back into structured turns for the frontend, and strips the heavy
token/logprob arrays down to summary stats.

Usage:
    python3 convert.py <dump.pt> <out.json>             full view-model
    python3 convert.py --summary <dump.pt> <out.json>   per-dump stats only
"""

from __future__ import annotations

import json
import re
import sys

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# Qwen3.x token id for <|im_start|>. Lets us segment the response token ids into
# turns without a tokenizer (the dashboard image ships torch only), so each
# turn can carry its trained/masked token counts. A different served model
# family needs its own id here, just as parse_turns needs its string markers.
IM_START_ID = 248045

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_RETURNCODE_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>")
_OUTPUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>", re.DOTALL)


def _parse_tool_call(raw: str) -> dict:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return {"name": obj.get("name"), "arguments": obj.get("arguments")}
    except (json.JSONDecodeError, AttributeError):
        return {"raw": raw}


def _parse_tool_response(raw: str) -> dict:
    rc = _RETURNCODE_RE.search(raw)
    out = _OUTPUT_RE.search(raw)
    if rc or out:
        return {
            "returncode": int(rc.group(1)) if rc else None,
            "output": out.group(1) if out else "",
        }
    return {"raw": raw.strip()}


def _parse_turn_body(role: str, body: str) -> dict:
    turn: dict = {"role": role}
    rest = body

    thinks = [m.group(1).strip() for m in _THINK_RE.finditer(rest)]
    rest = _THINK_RE.sub("", rest)
    if thinks:
        turn["think"] = thinks

    if role == "assistant":
        calls = [_parse_tool_call(m.group(1)) for m in _TOOL_CALL_RE.finditer(rest)]
        rest = _TOOL_CALL_RE.sub("", rest)
        if calls:
            turn["tool_calls"] = calls
    else:
        resps = [_parse_tool_response(m.group(1)) for m in _TOOL_RESP_RE.finditer(rest)]
        rest = _TOOL_RESP_RE.sub("", rest)
        if resps:
            turn["tool_responses"] = resps

    text = rest.strip()
    if text:
        turn["text"] = text
    return turn


def _bash_command(turn: dict) -> str | None:
    """The bash command issued in a turn, if any (for pairing with its output)."""
    for call in turn.get("tool_calls") or []:
        args = call.get("arguments")
        cmd = args.get("command") if isinstance(args, dict) else None
        if call.get("name") == "bash" and isinstance(cmd, str):
            return cmd
    return None


def _pair_commands_with_responses(turns: list[dict]) -> None:
    """Stamp each tool response with the command that produced it (they arrive
    as separate turns), keeping the frontend's terminal rendering self-contained.
    """
    for prev, turn in zip(turns, turns[1:]):
        cmd = _bash_command(prev)
        if cmd is None:
            continue
        for resp in turn.get("tool_responses") or []:
            resp.setdefault("command", cmd)


def _per_turn_token_stats(resp_tokens: list, loss_mask: list) -> list | None:
    """Per-turn (total, trained) token counts: segment the response token ids at
    each IM_START and sum the loss mask within each segment. Aligns 1:1 by chunk
    index with ``response.split(IM_START)`` (head segment first), so parse_turns
    can attach each turn's training signal. None when the arrays don't line up
    (e.g. aborted samples with a 1-token response)."""
    if not resp_tokens or len(loss_mask) != len(resp_tokens):
        return None
    bounds = [i for i, t in enumerate(resp_tokens) if t == IM_START_ID]
    stats = []
    prev = 0
    for b in (*bounds, len(resp_tokens)):
        stats.append({"tok": b - prev, "trained": int(sum(loss_mask[prev:b]))})
        prev = b
    return stats


def parse_turns(response: str, token_stats: list | None = None) -> list[dict]:
    """Slice the merged chat-template stream back into role turns.

    The response starts mid-assistant-turn, so the chunk before the first
    ``<|im_start|>`` is assistant output. ``partition`` tolerates a truncated
    trajectory missing its trailing ``<|im_end|>``. When ``token_stats`` is
    given (from _per_turn_token_stats), each turn is stamped with its trained /
    total token counts — indexed by chunk position so dropped empty chunks stay
    aligned.
    """
    turns = []
    for i, chunk in enumerate(response.split(IM_START)):
        if i == 0:
            role = "assistant"
            body = chunk
        else:
            role, _, body = chunk.partition("\n")
            role = role.strip() or "unknown"
        body = body.partition(IM_END)[0]
        if not body.strip():
            continue
        turn = _parse_turn_body(role, body)
        if token_stats is not None and i < len(token_stats):
            turn["tok"] = token_stats[i]["tok"]
            turn["trained"] = token_stats[i]["trained"]
        turns.append(turn)
    _pair_commands_with_responses(turns)
    return turns


def _attach_truncated_tail(turns: list[dict], tail: dict) -> None:
    """Surface a rolled-back terminal generation (model.py drops it from the token
    stream when a turn has no tool call). In thinking mode the prompt opens
    ``<think>`` so the raw output begins *inside* the think block (no leading tag);
    split on a closing ``</think>`` if the model emitted one, else treat it all as
    unclosed reasoning. Attach to the trailing dangling-``<think>`` stub turn so the
    dashboard shows the real (often huge) reasoning instead of the 5-token stub."""
    text = tail.get("text") or ""
    if "</think>" in text:
        think, _, after = text.partition("</think>")
        post = _parse_turn_body("assistant", after)
    else:
        think, post = text, {}
    entry = {
        "tokens": tail.get("tokens"),
        "finish": tail.get("finish"),
        "closed_think": "</think>" in text,
        "think": think.strip() or None,
        "text": post.get("text"),
        "tool_calls": post.get("tool_calls"),
    }
    if turns and turns[-1].get("role") == "assistant" and not turns[-1].get("tool_calls"):
        turns[-1]["truncated_generation"] = entry
    else:
        turns.append({"role": "assistant", "truncated_generation": entry})


def parse_spans(trace) -> list[dict]:
    """Pair span_start/span_end trace events into a flat span list."""
    if not isinstance(trace, dict):
        return []
    starts: dict[str, dict] = {}
    spans = []
    for ev in trace.get("events") or []:
        sid = ev.get("span_id")
        if ev.get("type") == "span_start":
            starts[sid] = ev
        elif ev.get("type") == "span_end" and sid in starts:
            st = starts.pop(sid)
            spans.append(
                {
                    "name": st.get("name"),
                    "start_ts": st.get("ts"),
                    "end_ts": ev.get("ts"),
                    "duration_sec": (ev.get("ts") or 0) - (st.get("ts") or 0),
                    "parent_span_id": st.get("parent_span_id"),
                    "span_id": sid,
                    "attrs": st.get("attrs"),
                }
            )
    # Unclosed spans (run died mid-rollout) still matter for debugging.
    for st in starts.values():
        spans.append(
            {
                "name": st.get("name"),
                "start_ts": st.get("ts"),
                "end_ts": None,
                "duration_sec": None,
                "parent_span_id": st.get("parent_span_id"),
                "span_id": st.get("span_id"),
                "attrs": st.get("attrs"),
            }
        )
    spans.sort(key=lambda s: s["start_ts"] or 0)
    return spans


def _slim_steps(steps) -> list | None:
    """Reduce harbor_step_results to the bits the UI shows (name + reward)."""
    if not isinstance(steps, list):
        return None
    out = [
        {"name": st.get("name"), "reward": st.get("reward")}
        for st in steps
        if isinstance(st, dict)
    ]
    return out or None


def _slim_submissions(subs) -> list | None:
    """Trim the iterative-submission trace (frontier_cs) to the fields the UI shows."""
    if not isinstance(subs, list):
        return None
    out = [
        {
            "ordinal": s.get("ordinal"),
            "status": s.get("status"),
            "score": s.get("score"),
            "score_raw": s.get("score_raw"),
            "code_chars": s.get("code_chars"),
            "is_solved": s.get("is_solved"),
            "ts": s.get("ts_done") or s.get("ts_started"),
            "detail": ((s.get("detail") or "")[:200] or None),
        }
        for s in subs
        if isinstance(s, dict)
    ]
    return out or None


def _nonneg_seconds(value) -> float | None:
    try:
        sec = float(value)
    except (TypeError, ValueError):
        return None
    return sec if sec >= 0 else None


def _timing_phases(timing: dict, elapsed) -> list | None:
    """Display-ready wall-clock split without double-counting generation.

    ``timing["generate"]`` is the LLM-inference time and is a subset of the env's
    ``agent`` phase when that phase exists. Split agent into generation vs tool/agent
    overhead so the bar totals roughly match rollout elapsed time.
    """
    if not isinstance(timing, dict):
        return None

    phases: list[dict] = []

    def add(name: str, seconds, kind: str) -> None:
        sec = _nonneg_seconds(seconds)
        if sec is not None and sec > 0:
            phases.append({"name": name, "seconds": round(sec, 3), "kind": kind})

    add("boot", timing.get("work_boot"), "boot")
    add("prep", timing.get("prep"), "prep")

    gen_s = _nonneg_seconds(timing.get("generate"))
    agent_s = _nonneg_seconds(timing.get("agent"))
    if gen_s is not None:
        add("generate", gen_s, "generate")
        if agent_s is not None:
            add("agent/tools", max(0.0, agent_s - gen_s), "agent")
    else:
        add("agent", agent_s, "agent")

    add("diff", timing.get("diff"), "diff")
    add("verifier/RM", timing.get("verifier", timing.get("eval")), "verifier")

    known = sum(p["seconds"] for p in phases)
    elapsed_s = _nonneg_seconds(elapsed)
    if elapsed_s is not None and elapsed_s - known > 0.5:
        add("other/env", elapsed_s - known, "other")

    return phases or None


def sample_view(s: dict) -> dict:
    md = s.get("metadata") or {}
    # Env diagnostics (is_solved, timing, verifier, harbor_steps_*, ...) are nested
    # under metadata["agentic"] by generate.py (the metrics path reads them there), so
    # overlay them to the top level for the reads below. Genuine top-level keys
    # (instance_id, eval_dataset, budgets, abort_reason) don't collide with agentic's,
    # so the overlay is order-safe; absent on old/aborted dumps -> no-op.
    md = {**md, **(md.get("agentic") or {})}
    loss_mask = s.get("loss_mask") or []
    logprobs = s.get("rollout_log_probs") or []
    trained = int(sum(loss_mask))
    mean_logprob = None
    if trained and len(logprobs) == len(loss_mask):
        mean_logprob = sum(lp for lp, m in zip(logprobs, loss_mask) if m) / trained

    prompt = s.get("prompt")
    tokens = s.get("tokens") or []
    resp_tokens = tokens[-len(loss_mask):] if loss_mask else []
    token_stats = _per_turn_token_stats(resp_tokens, loss_mask)
    turns = parse_turns(s.get("response") or "", token_stats)
    tail = md.get("truncated_tail")
    if isinstance(tail, dict) and tail.get("text"):
        _attach_truncated_tail(turns, tail)
    timing = md.get("timing") or {}
    verifier = md.get("verifier") or {}
    budgets = md.get("budgets") or {}  # enforced budgets; fall back to task.toml values
    gen_s = timing.get("generate")
    elapsed = md.get("elapsed_sec")
    # Wall-clock not spent generating (in-sandbox tool exec + grading); divided
    # by turns it's a proxy for per-tool-call env latency.
    overhead_sec = (
        elapsed - gen_s if elapsed is not None and gen_s is not None else None
    )
    return {
        "index": s.get("index"),
        "group_index": s.get("group_index"),
        "rollout_id": s.get("rollout_id"),
        "session_id": s.get("session_id"),
        "status": s.get("status"),
        "reward": s.get("reward"),
        "eval_dataset": md.get("eval_dataset"),
        "instance_id": md.get("instance_id"),
        "repo": md.get("repo"),
        "is_solved": md.get("is_solved"),
        "applied_cleanly": md.get("applied_cleanly"),
        "abort_reason": md.get("abort_reason"),
        # Real terminal cause of a discarded (no-usable-turn) episode shipped via
        # _ship_null: ContextLengthExceeded / NoProgress / Aborted / ImageUnusable.
        # None for normal episodes. The rolled-back generation rides on truncated_tail.
        "exit_status": md.get("exit_status"),
        "finish_reason": md.get("finish_reason"),
        # In-sandbox agent process outcome: exit code (0 on a clean run) and, on a
        # nonzero exit, the tail of its stdout/stderr -- the "why" behind a
        # zero-turn adapter_session_empty (empty string on success / old dumps).
        "agent_exit_code": md.get("agent_exit_code"),
        "agent_tail": md.get("agent_tail"),
        "elapsed_sec": md.get("elapsed_sec"),
        "segment_idx": md.get("segment_idx"),
        "num_segments": md.get("num_segments"),
        "image": md.get("image"),
        "workdir": md.get("workdir"),
        "base_commit": md.get("base_commit"),
        "problem_statement": md.get("problem_statement")
        or (prompt if isinstance(prompt, str) else None),
        # Decoded first-turn model input; older dumps lack it -> None.
        "full_prompt": md.get("full_prompt"),
        "n_tokens": len(s.get("tokens") or []),
        "response_length": s.get("response_length"),
        "trained_tokens": trained,
        "mean_logprob": mean_logprob,
        "weight_versions": sorted(set(s.get("weight_versions") or [])),
        "prefix_cache": s.get("prefix_cache_info"),
        "n_turns": len(turns),
        # Rolled-back terminal generation summary (tokens/finish); the text rides on
        # the trailing turn's "truncated_generation". None when the run ended cleanly.
        "truncated_tail": {k: tail.get(k) for k in ("tokens", "finish")} if isinstance(tail, dict) else None,
        # --- timing / latency profile (md.timing; absent on old dumps) ---
        "gen_s": gen_s,
        "overhead_sec": overhead_sec,
        "recorded_turns": timing.get("n_turns"),
        "non_generation_time": s.get("non_generation_time"),
        "timing_phases": _timing_phases(timing, elapsed),
        # --- task provenance / sandbox resources ---
        "dockerfile": md.get("dockerfile"),
        "task_path": md.get("task_path"),
        "boot_timeout_sec": budgets.get("boot_sec"),
        "agent_timeout_sec": budgets.get("agent_sec", md.get("agent_timeout_sec")),
        "verifier_timeout_sec": budgets.get("eval_sec", verifier.get("timeout_sec")),
        "cpus": md.get("cpus"),
        "memory_mb": md.get("memory_mb"),
        "segment_kind": md.get("segment_kind"),
        # --- grading detail ---
        "harbor_steps_completed": md.get("harbor_steps_completed"),
        "harbor_steps_total": md.get("harbor_steps_total"),
        "harbor_step_results": _slim_steps(md.get("harbor_step_results")),
        # --- iterative submissions (frontier_cs); None for other task families ---
        "submissions": _slim_submissions(md.get("submissions")),
        "submission_summary": md.get("submission_summary"),
        # --- training bookkeeping ---
        "remove_sample": s.get("remove_sample"),
        "label": s.get("label"),
        "turns": turns,
        "spans": parse_spans(s.get("trace")),
    }


def _reward_number(reward) -> float | None:
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _bucket_stats(rows: list[dict]) -> dict:
    rewards = [r for s in rows if (r := _reward_number(s.get("reward"))) is not None]
    return {
        "n": len(rows),
        "solved": sum(1 for s in rows if s.get("solved")),
        "aborted": sum(1 for s in rows if s.get("status") == "aborted"),
        "truncated": sum(1 for s in rows if s.get("status") == "truncated"),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
    }


def summarize(samples: list[dict]) -> dict:
    """Per-dump stats for the run landing page (and instance history).

    Both this path and server.ts summarizeView emit this shape -- keep in sync.
    """
    summary = _bucket_stats(samples)
    datasets = sorted({s["dataset"] for s in samples if s.get("dataset")})
    if datasets:
        summary["datasets"] = {
            name: _bucket_stats([s for s in samples if s.get("dataset") == name]) for name in datasets
        }
    else:
        summary["datasets"] = None
    summary["instances"] = [
        {
            "id": s.get("instance"),
            "dataset": s.get("dataset"),
            "reward": _reward_number(s.get("reward")),
            "solved": bool(s.get("solved")),
            "status": s.get("status"),
        }
        for s in samples
        if s.get("instance")
    ]
    return summary


def _sample_brief(s: dict) -> dict:
    md = s.get("metadata") or {}
    md = {**md, **(md.get("agentic") or {})}  # is_solved is nested under "agentic" (see sample_view)
    return {
        "status": s.get("status"),
        "reward": s.get("reward"),
        "solved": md.get("is_solved"),
        "instance": md.get("instance_id"),
        "dataset": md.get("eval_dataset"),
    }


def _load_dump(dump_path: str) -> dict:
    import torch

    return torch.load(dump_path, map_location="cpu", weights_only=False)


def convert(dump_path: str, out_path: str) -> None:
    dump = _load_dump(dump_path)
    samples = dump.get("samples") or []
    view = {
        "rollout_id": dump.get("rollout_id"),
        "n_samples": len(samples),
        "samples": [sample_view(s) for s in samples],
    }
    with open(out_path, "w") as f:
        json.dump(view, f, default=str)


def convert_summary(dump_path: str, out_path: str) -> None:
    dump = _load_dump(dump_path)
    samples = dump.get("samples") or []
    summary = {"rollout_id": dump.get("rollout_id"), **summarize([_sample_brief(s) for s in samples])}
    with open(out_path, "w") as f:
        json.dump(summary, f, default=str)


if __name__ == "__main__":
    argv = sys.argv[1:]
    summary_mode = "--summary" in argv
    argv = [a for a in argv if a != "--summary"]
    if len(argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} [--summary] <dump.pt> <out.json>")
    (convert_summary if summary_mode else convert)(argv[0], argv[1])
