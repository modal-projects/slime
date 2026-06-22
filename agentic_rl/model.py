"""In-process mini-swe ``Model`` that records exact tokens — the model-API seam.

Token-in-token-out: each turn appends the new-context delta (loss_mask 0) and the
exact ids SGLang returned (loss_mask 1) plus per-token logprobs and the
``weight_version`` live at generation. The prompt is the one running token
sequence; the new-context delta is rendered from the *observation* messages only
(never re-rendering a prior assistant turn), so neither Qwen's ``<think>``
history-stripping nor the tool-call-arg whitespace rstrip can drift the prompt
away from the training target. Tool calls are parsed from the raw output with the
served model's SGLang parser (native tool-calls, qwen3_coder family).

A fresh ``DefaultAgent`` per episode leg (harbor multi-step) appears as a new
chain → a sibling training Sample. One instance per episode; read ``chains`` /
``aborted`` / stats after the run.
"""

from __future__ import annotations

import json
import secrets
import time
import urllib.request
from dataclasses import dataclass, field

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages

from slime.agent.parsing import parse_model_output

from .prompts import BASH_TOOL, FORMAT_ERROR_TEMPLATE, OBSERVATION_TEMPLATE

_RENDER_KEYS = ("role", "content", "tool_calls", "tool_call_id", "reasoning_content", "name")


@dataclass
class Chain:
    """One contiguous conversation = one training Sample.

    ``prompt_len`` is the masked-out prefix (first render); ``versions`` holds one
    weight_version per generated turn in this chain.
    """

    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)
    prompt_len: int = 0
    seen_msgs: int = 0
    msg_hashes: list[str] = field(default_factory=list)

    @property
    def has_response(self) -> bool:
        return len(self.tokens) > self.prompt_len


def _stable_hash(msg: dict) -> str:
    return json.dumps({k: msg.get(k) for k in _RENDER_KEYS}, sort_keys=True, ensure_ascii=False, default=str)


class RecordingModel:
    def __init__(
        self,
        tokenizer,
        sampling_params: dict,
        router_url: str,
        session_id: str,
        *,
        tool_parser: str | None,
        reasoning_parser: str | None,
        observation_template: str = OBSERVATION_TEMPLATE,
        format_error_template: str = FORMAT_ERROR_TEMPLATE,
        tools: list[dict] | None = None,
        max_context_len: int = 0,
        query_timeout: int = 600,
        max_empty_turns: int = 3,
    ):
        self.tokenizer = tokenizer
        self.sampling_params = dict(sampling_params)
        self.url = f"{router_url.rstrip('/')}/generate"
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.observation_template = observation_template
        self.format_error_template = format_error_template
        self.tools = tools if tools is not None else [BASH_TOOL]
        self.max_context_len = int(max_context_len or 0)
        self.query_timeout = query_timeout
        self.max_empty_turns = max_empty_turns
        # Consistent-hash routing → every turn hits the same worker (prefix cache).
        self.headers = {"Content-Type": "application/json", "X-SMG-Routing-Key": session_id}

        self.chains: list[Chain] = []
        self.aborted = False
        self.gen_time = 0.0
        self.cached_tokens = 0
        self.input_tokens = 0
        self.n_format_errors = 0
        self._empty_streak = 0

    @property
    def cur(self) -> Chain:
        return self.chains[-1]

    # mini-swe Model protocol -----------------------------------------------
    def query(self, messages: list[dict], **kwargs) -> dict:
        c = self.chains[-1] if self.chains else None
        hashes = [_stable_hash(m) for m in messages]
        is_append = c is not None and c.seen_msgs and len(hashes) >= c.seen_msgs and hashes[: c.seen_msgs] == c.msg_hashes
        if not is_append:
            c = Chain()
            self.chains.append(c)

        if c.seen_msgs == 0:
            self._extend(self._render_initial(messages), 0)
            c.prompt_len = len(c.tokens)
        else:
            self._extend(self._render_continuation(messages[c.seen_msgs :]), 0)
        c.seen_msgs = len(messages)
        c.msg_hashes = hashes

        out_ids, logps, finish, version = self._generate(c.tokens)
        self._append_generated(out_ids, logps, version)
        if finish == "abort":
            self.aborted = True
            raise LimitsExceeded(self._exit_message("Aborted", "generation aborted by weight update"))

        raw = self.tokenizer.decode(out_ids, skip_special_tokens=False) if out_ids else ""
        parsed = parse_model_output(
            raw,
            tools_schema=self.tools,
            tool_parser_name=self.tool_parser,
            reasoning_parser_name=self.reasoning_parser,
        )
        try:
            actions, tool_calls = self._actions_from_tool_uses(parsed.tool_uses)
        except FormatError:
            self._rollback_generated(len(out_ids))
            self.n_format_errors += 1
            self._empty_streak += 1
            if finish == "length" or self._empty_streak >= self.max_empty_turns:
                status = "ContextLengthExceeded" if finish == "length" else "NoProgress"
                raise LimitsExceeded(self._exit_message(status, f"no tool call (finish={finish})"))
            raise

        self._empty_streak = 0
        message: dict = {"role": "assistant", "content": parsed.text or None, "extra": {"actions": actions, "cost": 0.0}}
        if parsed.reasoning:
            message["reasoning_content"] = parsed.reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None):
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}

    # internals -------------------------------------------------------------
    def _actions_from_tool_uses(self, tool_uses: list[dict]) -> tuple[list[dict], list[dict]]:
        if not tool_uses:
            raise self._format_error("No tool calls found in the response. Every response MUST include a tool call.")
        actions, tool_calls = [], []
        for tu in tool_uses:
            name, args = tu.get("name"), tu.get("input") or {}
            err = ""
            if name != "bash":
                err += f"Unknown tool '{name}'."
            if not isinstance(args, dict) or "command" not in args:
                err += "Missing 'command' argument in bash tool call."
            if err:
                raise self._format_error(err.strip())
            call_id = f"call_{secrets.token_hex(12)}"
            actions.append({"command": args["command"], "tool_call_id": call_id})
            tool_calls.append(
                {"id": call_id, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": args["command"]})}}
            )
        return actions, tool_calls

    def _format_error(self, error: str) -> FormatError:
        content = Template(self.format_error_template, undefined=StrictUndefined).render(error=error, actions=[])
        return FormatError({"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}})

    @staticmethod
    def _exit_message(status: str, content: str) -> dict:
        return {"role": "exit", "content": content, "extra": {"exit_status": status, "submission": ""}}

    def _template_ids(self, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
        enc = self.tokenizer.apply_chat_template(
            messages, tools=self.tools, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
        ids = list(ids)
        if ids and isinstance(ids[0], list):  # transformers>=5 may return [[...]]
            ids = ids[0]
        return ids

    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in m.items() if k in _RENDER_KEYS and v is not None} for m in messages]

    def _render_initial(self, messages: list[dict]) -> list[int]:
        return self._template_ids(self._clean(messages), add_generation_prompt=True)

    def _render_continuation(self, new_messages: list[dict]) -> list[int]:
        """Token delta to append after the previous turn's raw output_ids: the
        observation message(s) added this turn plus the next generation prompt.

        Anchored on a sentinel user message so a prior assistant turn is never
        re-rendered. The model's raw output stops at ``<|im_end|>``; the template
        emits ``<|im_end|>\\n`` after a finished turn, so we keep the sentinel's
        trailing newline as the inter-turn separator.
        """
        obs = self._clean([m for m in new_messages if m.get("role") != "assistant"])
        if not obs:
            return []
        sentinel = [{"role": "user", "content": ""}]
        base = self._template_ids(sentinel, add_generation_prompt=False)
        full = self._template_ids(sentinel + obs, add_generation_prompt=True)
        if len(base) < 1 or full[: len(base)] != base:
            return self._template_ids(obs, add_generation_prompt=True)  # unexpected shape; best-effort
        return full[len(base) - 1 :]

    def _extend(self, ids: list[int], mask: int) -> None:
        c = self.cur
        c.tokens += ids
        c.loss_mask += [mask] * len(ids)
        c.logprobs += [0.0] * len(ids)

    def _append_generated(self, out_ids: list[int], logps: list[float], version: str | None) -> None:
        c = self.cur
        c.tokens += out_ids
        c.loss_mask += [1] * len(out_ids)
        c.logprobs += logps
        c.versions.append(version)

    def _rollback_generated(self, n: int) -> None:
        c = self.cur
        if n:
            del c.tokens[-n:], c.loss_mask[-n:], c.logprobs[-n:]
        c.versions.pop()

    def _generate(self, input_ids: list[int]) -> tuple[list[int], list[float], str, str | None]:
        sp = dict(self.sampling_params)
        if self.max_context_len > 0:
            remaining = self.max_context_len - len(input_ids)
            if remaining <= 0:
                return [], [], "length", None
            cap = int(sp.get("max_new_tokens") or remaining)
            sp["max_new_tokens"] = min(cap, remaining)

        payload = {"input_ids": input_ids, "sampling_params": sp, "return_logprob": True}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=self.headers)
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.query_timeout) as resp:
            data = json.loads(resp.read())
        self.gen_time += time.perf_counter() - t0

        meta = data.get("meta_info") or {}
        self.cached_tokens += meta.get("cached_tokens", 0)
        self.input_tokens += len(input_ids)
        lps = meta.get("output_token_logprobs") or []
        out_ids = [t[1] for t in lps]
        logps = [float(t[0]) for t in lps]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
        return out_ids, logps, finish, meta.get("weight_version")
