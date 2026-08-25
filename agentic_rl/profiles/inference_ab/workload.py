"""Simulated agentic-rollout workload for the inference A/B profile.

Runs *inside* the Modal container against a locally-booted SGLang engine and
mimics the Frontier-CS retro rollout's inference pattern as closely as the
production metrics describe it:

- real Frontier-CS prompts from ``/data/frontier_cs/train.jsonl`` (pre-rendered
  chat text, same as training with ``apply_chat_template=False``);
- groups of 8 siblings sharing one prompt (``n_samples_per_prompt=8``) launched
  together, so the radix cache sees the same sharing as production;
- multi-turn episodes: each turn POSTs ``/generate`` with the full ``input_ids``
  and ``return_logprob=True`` (identical request shape to
  ``agentic_rl.model._generate``), appends the generated ids plus a canned
  tool-observation block, sleeps a "sandbox exec" pause, and continues;
- per-sibling ``sampling_seed`` when the deterministic arm is active, matching
  ``sglang_rollout.GenerateState.group_sampling_seeds``.

A "step" is one batch of ``episodes`` episodes run under an in-flight cap of
``concurrency`` (production runs ~7 in-flight episodes per engine). Metrics are
computed with the same definitions as ``agentic_rl.metrics`` where they exist
(``decode_tok_per_s`` = episode output tokens / summed request latencies).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

# A plausible Frontier-CS tool observation (compile + run + partial score),
# rendered the way the qwen3 template renders a tool response inside a user
# turn. Content only matters insofar as EAGLE draft acceptance depends on
# text predictability; structure mirrors agentic_rl.prompts.OBSERVATION_TEMPLATE.
_OBSERVATION_TEXT = """
<tool_response>
<returncode>0</returncode>
<output>
g++ -O2 -std=c++17 -o /tmp/sol solution.cpp
compilation finished without warnings
running 12 sample cases...
case 01: OK (0.041s)
case 02: OK (0.038s)
case 03: WRONG ANSWER expected 194402 got 194371
case 04: OK (0.512s)
case 05: TIME LIMIT exceeded (2.001s)
case 06: OK (0.096s)
case 07: OK (0.104s)
case 08: WRONG ANSWER expected 7 got 6
case 09: OK (1.371s)
case 10: OK (0.088s)
case 11: OK (0.664s)
case 12: OK (0.312s)
partial score: 0.6667 (8/12)
</output>
</tool_response>
""".strip()


def _pctl(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return float(ys[min(len(ys) - 1, int(q * len(ys)))])


def _summary(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": _pctl(xs, 0.50),
        "p90": _pctl(xs, 0.90),
        "max": max(xs),
    }


class Workload:
    def __init__(
        self,
        *,
        base_url: str,
        tokenizer,
        prompts: list[str],
        deterministic: bool,
        steps: int = 5,
        episodes: int = 16,
        concurrency: int = 8,
        turns: int = 12,
        max_new: int = 4096,
        pause: float = 4.0,
        ctx_limit: int = 65536,
        seed: int = 20260802,
        siblings: int = 8,
    ):
        self.url = f"{base_url.rstrip('/')}/generate"
        self.tokenizer = tokenizer
        self.deterministic = deterministic
        self.steps = steps
        self.episodes = episodes
        self.concurrency = concurrency
        self.turns = turns
        self.max_new = max_new
        self.pause = pause
        self.ctx_limit = ctx_limit
        self.seed = seed
        self.siblings = siblings

        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is None or im_end < 0:
            raise ValueError("tokenizer has no <|im_end|> token")
        self.im_end = int(im_end)

        # Continuation delta appended after a finished assistant turn:
        # newline + rendered observation user turn + next generation prompt
        # (agentic_rl.model keeps the template's trailing "\n" separator).
        cont = f"\n<|im_start|>user\n{_OBSERVATION_TEXT}<|im_end|>\n<|im_start|>assistant\n"
        self.obs_ids = list(tokenizer.encode(cont, add_special_tokens=False))

        groups_needed = steps * max(1, episodes // siblings)
        if len(prompts) < groups_needed:
            raise ValueError(f"need {groups_needed} distinct prompts, got {len(prompts)}")
        self.prompt_ids = [list(tokenizer.encode(p, add_special_tokens=False)) for p in prompts[:groups_needed]]

    # -- one turn --------------------------------------------------------

    async def _turn(self, client, ids: list[int], sibling_idx: int) -> dict[str, Any]:
        remaining = self.ctx_limit - len(ids)
        sp: dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": min(self.max_new, remaining),
            "stop_token_ids": [self.im_end],
            "no_stop_trim": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        if self.deterministic:
            # sglang_rollout gives sibling i seed rollout_seed + i (same per group)
            sp["sampling_seed"] = self.seed + sibling_idx
        t0 = time.perf_counter()
        resp = await client.post(self.url, json={"input_ids": ids, "sampling_params": sp, "return_logprob": True})
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("meta_info") or {}
        lps = meta.get("output_token_logprobs") or []
        out_ids = [t[1] for t in lps]
        return {
            "latency": latency,
            "out_ids": out_ids,
            "prompt_tokens": meta.get("prompt_tokens", len(ids)),
            "completion_tokens": meta.get("completion_tokens", len(out_ids)),
            "cached_tokens": meta.get("cached_tokens", 0),
            "spec_verify_ct": meta.get("spec_verify_ct", 0),
            "spec_accept_token_num": meta.get("spec_accept_token_num", 0),
            "spec_draft_token_num": meta.get("spec_draft_token_num", 0),
            "finish": (meta.get("finish_reason") or {}).get("type", "?"),
        }

    # -- one episode ------------------------------------------------------

    async def _episode(self, client, sem, prompt_ids: list[int], ep_key: int, sibling_idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed * 1000003 + ep_key)
        ids = list(prompt_ids)
        turn_recs: list[dict[str, Any]] = []
        t_start = time.perf_counter()
        async with sem:
            for turn in range(self.turns):
                rec = await self._turn(client, ids, sibling_idx)
                out_ids = rec.pop("out_ids")
                rec["turn"] = turn
                turn_recs.append(rec)
                if not out_ids:
                    break
                ids += out_ids
                if out_ids[-1] != self.im_end:
                    ids.append(self.im_end)  # length-capped turn; close it like a rollback would
                if len(ids) + len(self.obs_ids) + 512 >= self.ctx_limit:
                    break
                ids += self.obs_ids
                # sandbox exec pause (agentic/exec_time: ~4s/call, jittered)
                await asyncio.sleep(self.pause * rng.uniform(0.5, 1.5))
        elapsed = time.perf_counter() - t_start
        gen_time = sum(r["latency"] for r in turn_recs)
        out_tok = sum(r["completion_tokens"] for r in turn_recs)
        return {
            "episode": ep_key,
            "turns": len(turn_recs),
            "elapsed": elapsed,
            "gen_time": gen_time,
            "output_tokens": out_tok,
            "final_ctx": len(ids),
            "decode_tok_per_s": out_tok / gen_time if gen_time > 0 else 0.0,
            "cached_tokens": sum(r["cached_tokens"] for r in turn_recs),
            "prompt_tokens": sum(r["prompt_tokens"] for r in turn_recs),
            "spec_verify_ct": sum(r["spec_verify_ct"] for r in turn_recs),
            "spec_accept_token_num": sum(r["spec_accept_token_num"] for r in turn_recs),
            "spec_draft_token_num": sum(r["spec_draft_token_num"] for r in turn_recs),
            "turn_latency": [round(r["latency"], 3) for r in turn_recs],
            "turn_tokens": [r["completion_tokens"] for r in turn_recs],
            "finish_mix": {f: sum(1 for r in turn_recs if r["finish"] == f) for f in {r["finish"] for r in turn_recs}},
        }

    # -- one step ---------------------------------------------------------

    async def _step(self, client, step: int) -> dict[str, Any]:
        sem = asyncio.Semaphore(self.concurrency)
        groups = max(1, self.episodes // self.siblings)
        tasks = []
        for g in range(groups):
            prompt = self.prompt_ids[step * groups + g]
            for s in range(self.siblings):
                ep_key = (step * groups + g) * self.siblings + s
                tasks.append(self._episode(client, sem, prompt, ep_key, s))
        t0 = time.perf_counter()
        episodes = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

        out_tok = sum(e["output_tokens"] for e in episodes)
        verify = sum(e["spec_verify_ct"] for e in episodes)
        draft = sum(e["spec_draft_token_num"] for e in episodes)
        accept = sum(e["spec_accept_token_num"] for e in episodes)
        prompt_tok = sum(e["prompt_tokens"] for e in episodes)
        cached_tok = sum(e["cached_tokens"] for e in episodes)
        return {
            "step": step,
            "wall_s": wall,
            "episodes": len(episodes),
            "turns": sum(e["turns"] for e in episodes),
            "output_tokens": out_tok,
            "engine_tok_per_s": out_tok / wall if wall > 0 else 0.0,
            "decode_tok_per_s": _summary([e["decode_tok_per_s"] for e in episodes]),
            "episode_elapsed": _summary([e["elapsed"] for e in episodes]),
            "spec_accept_length": out_tok / verify if verify else 0.0,
            "spec_accept_rate": accept / draft if draft else 0.0,
            "prefix_cache_hit_rate": cached_tok / prompt_tok if prompt_tok else 0.0,
            "episode_recs": episodes,
        }

    # -- entry ------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        import httpx

        step_recs = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=httpx.Limits(max_connections=256)) as client:
            for step in range(self.steps):
                rec = await self._step(client, step)
                step_recs.append(rec)
                brief = {k: v for k, v in rec.items() if k != "episode_recs"}
                print(f"[step {step}] {json.dumps(brief, default=float)}", flush=True)

        decode_all = [e["decode_tok_per_s"] for r in step_recs for e in r["episode_recs"]]
        walls = [r["wall_s"] for r in step_recs]
        return {
            "config": {
                "deterministic": self.deterministic,
                "steps": self.steps,
                "episodes_per_step": self.episodes,
                "concurrency": self.concurrency,
                "turns": self.turns,
                "max_new": self.max_new,
                "pause": self.pause,
                "ctx_limit": self.ctx_limit,
                "seed": self.seed,
                "siblings": self.siblings,
            },
            "summary": {
                "step_wall_s": _summary(walls),
                "decode_tok_per_s": _summary(decode_all),
                "engine_tok_per_s": _summary([r["engine_tok_per_s"] for r in step_recs]),
                "spec_accept_length": _summary([r["spec_accept_length"] for r in step_recs]),
                "spec_accept_rate": _summary([r["spec_accept_rate"] for r in step_recs]),
                "prefix_cache_hit_rate": _summary([r["prefix_cache_hit_rate"] for r in step_recs]),
                "episode_elapsed": _summary([e["elapsed"] for r in step_recs for e in r["episode_recs"]]),
                "total_output_tokens": sum(r["output_tokens"] for r in step_recs),
            },
            "steps_detail": step_recs,
        }
