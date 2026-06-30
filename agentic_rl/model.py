"""mini-swe-agent Model that calls sglang directly and records exact tokens — the model-API
seam. Keeps ONE running token sequence: each turn appends the new context (loss_mask 0) and the
exact generated ids (loss_mask 1), so the trajectory is token-in-token-out faithful (one training
Sample, no re-tokenization). One instance per episode; read tokens/loss_mask/logprobs/versions after.
"""

import json
import time
import urllib.request

from minisweagent.exceptions import FormatError, InterruptAgentFlow
from minisweagent.models.utils.actions_text import parse_regex_actions

from .prompts import ACTION_REGEX, FORMAT_ERROR_TEMPLATE

# Per-turn generation budget against the served context window.
_CONTEXT_MARGIN = 256  # headroom for stop/special tokens
_MIN_GEN_TOKENS = 2048  # below this much room left, end the episode rather than spiral on truncated actions
_STOP_STRINGS = ["<|user|>", "<|observation|>"]  # role tokens that begin the next turn — never inside a valid action


class ContextExceeded(InterruptAgentFlow):
    """The growing trajectory left no room to generate — end the episode cleanly (like LimitsExceeded).
    With thinking kept in-context (clear_thinking=False) this is the natural end-state of a long episode
    once accumulated reasoning fills the window."""


class RecordingModel:
    config = None  # mini-swe Model protocol

    def __init__(
        self,
        tokenizer,
        sampling_params,
        router_url,
        observation_template,
        session_id,
        query_timeout=600,
        max_context_len=131072,
    ):
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.url = f"{router_url}/generate"
        self.query_timeout = query_timeout  # cap per-turn sglang call; bounds hung/queued generations
        self.max_context_len = max_context_len  # served window; per-turn gen is capped to the remainder
        self.observation_template = observation_template
        # consistent-hashing routing key → every turn hits the same worker (prefix cache across turns)
        self.headers = {"Content-Type": "application/json", "X-SMG-Routing-Key": session_id}
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.versions: list[str] = []
        self.prompt_len: int | None = None
        self.aborted = False
        self.gen_time = 0.0
        self.cached_tokens = 0  # radix-cache hits, summed over turns (prefix_cache_hit_rate)
        self.input_tokens = 0  # prompt tokens sent, summed over turns (hit-rate denominator)
        self.n_calls = 0  # total query() calls incl. format-error retries (vs len(versions) = productive turns)
        self.n_format_errors = 0
        self.n_length_truncations = 0  # turns the model overran the per-turn cap (finish_reason=length)
        self.reasoning_tokens = 0  # tokens spent inside <think>…</think>, summed over turns
        self._consumed = ""  # rendered text of the conversation already in `tokens`

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        # clear_thinking=False keeps each turn's reasoning in-context so the render stays a stable
        # prefix across turns — required by the Sample→Sample append below (GLM strips past <think>
        # by default, which would make the render non-monotonic and desync the recording).
        return self.tokenizer.apply_chat_template(
            clean, add_generation_prompt=add_generation_prompt, tokenize=False, clear_thinking=False
        )

    def _extend(self, text: str, mask: int) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.tokens += ids
        self.loss_mask += [mask] * len(ids)
        self.logprobs += [0.0] * len(ids)
        return len(ids)

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.n_calls += 1
        full = self._render(messages, add_generation_prompt=True)
        # Sample→Sample append: this turn's render MUST extend the prior verbatim (clear_thinking=False
        # keeps it prefix-stable). The slice is only correct under that invariant — guard it loudly.
        assert full.startswith(self._consumed), "chat-template render not prefix-stable — token recording would desync"
        n_ctx = self._extend(full[len(self._consumed) :], mask=0)  # new context since last turn, masked out
        if self.prompt_len is None:
            self.prompt_len = len(self.tokens)

        # Cap this turn's generation to the room left in the served window; if too little remains to
        # produce a useful turn, end the episode cleanly rather than spiral on truncated actions.
        remaining = self.max_context_len - len(self.tokens) - _CONTEXT_MARGIN
        if remaining < _MIN_GEN_TOKENS:
            raise ContextExceeded(
                {
                    "role": "exit",
                    "content": "ContextExceeded",
                    "extra": {"exit_status": "ContextExceeded", "submission": ""},
                }
            )
        sp = {
            **self.sampling_params,
            "stop": _STOP_STRINGS,
            "max_new_tokens": min(self.sampling_params.get("max_new_tokens", remaining), remaining),
        }

        n_in = len(self.tokens)
        payload = {"input_ids": self.tokens, "sampling_params": sp, "return_logprob": True}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=self.headers)
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.query_timeout) as resp:
            out = json.loads(resp.read())
        self.gen_time += time.perf_counter() - t0
        meta = out["meta_info"]
        self.cached_tokens += meta.get("cached_tokens", 0)
        self.input_tokens += n_in
        token_logprobs = meta.get("output_token_logprobs") or []
        text = out.get("text", "")
        finish = meta.get("finish_reason", {}).get("type", "stop")
        self.tokens += [t[1] for t in token_logprobs]  # exact generated ids (trained)
        self.loss_mask += [1] * len(token_logprobs)
        self.logprobs += [t[0] for t in token_logprobs]
        self.versions.append(meta.get("weight_version"))
        if finish == "abort":
            self.aborted = True
        if finish == "length":
            self.n_length_truncations += 1
        if "</think>" in text:  # reasoning the policy spent before its answer
            self.reasoning_tokens += len(self.tokenizer.encode(text.split("</think>")[0], add_special_tokens=False))

        # Parse the action from the post-reasoning segment only — a ```bash inside <think> is not an action.
        try:
            actions = parse_regex_actions(
                text.split("</think>")[-1],
                action_regex=ACTION_REGEX,
                format_error_template=FORMAT_ERROR_TEMPLATE,
                template_kwargs={"finish_reason": finish},
            )
        except FormatError:
            # mini-swe drops the malformed turn; drop BOTH this turn's generated ids AND the context
            # tokens appended above so `tokens` stays a faithful prefix of the next render (and
            # `_consumed`, left untouched, still matches `tokens`). The error feedback re-adds the context.
            drop = len(token_logprobs) + n_ctx
            if drop:
                del self.tokens[-drop:], self.loss_mask[-drop:], self.logprobs[-drop:]
            self.versions.pop()
            self.n_format_errors += 1
            raise
        self._consumed = self._render(messages + [{"role": "assistant", "content": text}], add_generation_prompt=False)
        return {"role": "assistant", "content": text, "extra": {"actions": actions, "cost": 0.0}}

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None):
        from minisweagent.models.utils.actions_text import format_observation_messages

        return format_observation_messages(
            outputs, observation_template=self.observation_template, template_vars=template_vars
        )

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}
