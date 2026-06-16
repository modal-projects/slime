"""Qwen-family OpenAI adapter: render tool-call arguments as a dict.

slime's ``OpenAIAdapter`` stringifies ``function.arguments`` before
``apply_chat_template``, but the Qwen3-Coder family (Qwen3.6-35B-A3B) template
iterates ``arguments | items`` and needs a mapping -- a string raises on turn
2+, capping every episode at one turn. This adapter renders ``arguments`` as a
dict on the inbound path only (the outbound OpenAI response stays string-form).

It also splices each turn's raw ``output_ids`` into the next prompt rather than
re-rendering the parsed assistant message (see ``_build_prompt``): the
qwen3_coder parser strips trailing whitespace from tool-call arguments, so a
re-render is not token-identical to what the model generated, which makes
``merge_turns`` log "prefix drift" and mask whole turns out of training.

slime renders through free functions with no method seam, so we register our
own ``/v1/chat/completions`` handler. ``_run_turn`` / ``_handle_chat_completions``
below are faithful mirrors of slime's -- keep them in sync.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web

from slime.agent.adapters import openai as _slime_openai
from slime.agent.adapters.openai import OpenAIAdapter
from slime.agent.adapters.common import ADAPTER_KEY, BaseAdapter, TOKENIZER_KEY, render_token_ids


def _dictify_tool_arguments(messages: list[dict]) -> None:
    """In place: parse each tool call's JSON-string ``function.arguments`` into a
    dict so ``apply_chat_template`` can iterate it. Idempotent; non-JSON left as-is."""
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            fn = call.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                s = args.strip()
                if not s:
                    fn["arguments"] = {}
                    continue
                try:
                    fn["arguments"] = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    pass


def _template_ids(tok, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    """``apply_chat_template`` -> flat token-id list (tolerating the 1-element
    batch that some transformers versions return for ``tokenize=True``)."""
    enc = tok.apply_chat_template(
        messages, tools=None, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    ids = list(ids)
    if ids and isinstance(ids[0], list):  # transformers>=5 may return [[...ids...]]
        ids = ids[0]
    return ids


def _tool_continuation_ids(new_messages: list[dict], tok) -> list[int]:
    """Token delta to append after the previous turn's raw ``output_ids``: the
    tool-result/user message(s) mini-swe added this turn, plus the next
    generation prompt.

    The model's raw ``output_ids`` stop at ``<|im_end|>``; the chat template
    emits ``<|im_end|>\\n`` for a finished assistant turn, so the delta must
    restore that single inter-turn newline. We anchor on a sentinel user message
    and slice from its trailing newline onward.
    """
    continuation = _slime_openai._translate_chat_messages(
        [m for m in new_messages if isinstance(m, dict) and m.get("role") != "assistant"]
    )
    if not continuation:
        return []
    sentinel = [{"role": "user", "content": ""}]
    base = _template_ids(tok, sentinel, add_generation_prompt=False)
    full = _template_ids(tok, sentinel + continuation, add_generation_prompt=True)
    if len(base) < 1 or full[: len(base)] != base:
        return []  # unexpected render shape -> caller falls back to a full re-render
    return full[len(base) - 1 :]  # start at the sentinel's trailing "\n" (the inter-turn separator)


def _build_prompt(target, messages: list[dict], tools_schema: list[dict] | None, kind: str, tok) -> list[int]:
    """Build the next prompt.

    On the ``append`` path, splice the previous turn's **raw** ``output_ids``
    into the prompt instead of re-rendering the parsed assistant message. The
    qwen3_coder parser strips trailing whitespace from tool-call arguments, so a
    re-render is not token-identical to what the model generated; that mismatch
    makes ``merge_turns`` log "prefix drift" and mask whole turns out of training
    (and makes the rollout subtly off-policy). Splicing the raw tokens keeps the
    prompt == the training target by construction. The parsed message is still
    returned to mini-swe for tool execution -- only prompt reconstruction here
    changes. ``new``/``wipe`` (and the first turn) fall back to a full re-render.
    """
    new_messages = messages[target.seen_msgs :] if kind == "append" else []
    (_slime_openai._extend_chat_messages if kind == "append" else _slime_openai._replace_chat_messages)(
        target, messages, tools_schema
    )
    if kind == "append" and target.turns:
        continuation = _tool_continuation_ids(new_messages, tok)
        if continuation:
            last = target.turns[-1]
            return list(last.prompt_ids) + list(last.output_ids) + continuation
    _dictify_tool_arguments(target.chat_messages)
    return render_token_ids(target, tok)


async def _run_turn(
    request: web.Request, body: dict, messages: list[dict]
):
    """Mirror of ``openai._run_turn`` calling the dict-args ``_build_prompt``."""
    sid = _slime_openai._request_session_id(request, body)
    adapter = request.app[ADAPTER_KEY]
    if sid in adapter.closed:
        raise web.HTTPServiceUnavailable(text="session closed")
    app = request.app
    s = adapter.store.setdefault(sid, _slime_openai.Session())
    task = asyncio.current_task()
    adapter.inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:
            target = s.main
            tools_schema = _slime_openai._normalize_tools(body.get("tools"))
            kind = _slime_openai._select_kind(s, messages)
            prompt_ids = _build_prompt(target, messages, tools_schema, kind, app[TOKENIZER_KEY])
            turn = await _slime_openai._generate(prompt_ids, s, body, app, session_id=sid)
            parsed = _slime_openai._parse_turn(target, turn, app)
            target.turns.append(turn)
            return turn, parsed, len(prompt_ids), len(turn.output_ids)
    finally:
        adapter.inflight.get(sid, set()).discard(task)


async def _handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """Mirror of ``openai._handle_chat_completions`` via the dict-args ``_run_turn``."""
    body = await request.json()
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise web.HTTPBadRequest(text="messages must be a list")
    turn, parsed, in_tok, out_tok = await _run_turn(request, body, messages)
    if body.get("stream"):
        return await _slime_openai._stream_chat_completion(
            request, body, parsed, turn.finish_reason, in_tok, out_tok
        )
    return web.json_response(
        _slime_openai._chat_completion_response(body, parsed, turn.finish_reason, in_tok, out_tok)
    )


class QwenOpenAIAdapter(OpenAIAdapter):
    """``OpenAIAdapter`` rendering tool-call arguments as a dict (see module
    docstring); only the ``/v1/chat/completions`` handler differs."""

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        # Skip OpenAIAdapter.__init__: it binds slime's string-args handler and
        # aiohttp can't re-bind a route. Do BaseAdapter setup, then our routes.
        BaseAdapter.__init__(
            self,
            tokenizer=tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        self.app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        self.app.router.add_post("/v1/responses", _slime_openai._handle_responses)  # mini-swe unused
        self.app.router.add_get("/healthz", _slime_openai._ok)
        self.app.router.add_get("/v1/models", _slime_openai._ok)
