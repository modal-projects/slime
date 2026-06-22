"""mini-swe-agent Model that calls sglang directly and records exact tokens — the model-API
seam. Keeps ONE running token sequence: each turn appends the new context (loss_mask 0) and the
exact generated ids (loss_mask 1), so the trajectory is token-in-token-out faithful (one training
Sample, no re-tokenization). One instance per episode; read tokens/loss_mask/logprobs/versions after.
"""

import json
import time
import urllib.request

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_text import parse_regex_actions

from .prompts import ACTION_REGEX, FORMAT_ERROR_TEMPLATE


class RecordingModel:
    config = None  # mini-swe Model protocol

    def __init__(self, tokenizer, sampling_params, router_url, observation_template, session_id, query_timeout=600):
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.url = f"{router_url}/generate"
        self.query_timeout = query_timeout  # cap per-turn sglang call; bounds hung/queued generations
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
        self.n_format_errors = 0
        self._consumed = ""  # rendered text of the conversation already in `tokens`

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        return self.tokenizer.apply_chat_template(clean, add_generation_prompt=add_generation_prompt, tokenize=False)

    def _extend(self, text: str, mask: int) -> None:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.tokens += ids
        self.loss_mask += [mask] * len(ids)
        self.logprobs += [0.0] * len(ids)

    def query(self, messages: list[dict], **kwargs) -> dict:
        full = self._render(messages, add_generation_prompt=True)
        self._extend(full[len(self._consumed) :], mask=0)  # new context since last turn, masked out
        if self.prompt_len is None:
            self.prompt_len = len(self.tokens)

        n_in = len(self.tokens)
        payload = {"input_ids": self.tokens, "sampling_params": self.sampling_params, "return_logprob": True}
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
        self.tokens += [t[1] for t in token_logprobs]  # exact generated ids (trained)
        self.loss_mask += [1] * len(token_logprobs)
        self.logprobs += [t[0] for t in token_logprobs]
        self.versions.append(meta.get("weight_version"))
        if meta.get("finish_reason", {}).get("type") == "abort":
            self.aborted = True

        try:
            actions = parse_regex_actions(
                text,
                action_regex=ACTION_REGEX,
                format_error_template=FORMAT_ERROR_TEMPLATE,
                template_kwargs={"finish_reason": meta.get("finish_reason", {}).get("type", "stop")},
            )
        except FormatError:
            # mini-swe drops a malformed turn from the conversation, so drop its output tokens too.
            n = len(token_logprobs)
            if n:
                del self.tokens[-n:], self.loss_mask[-n:], self.logprobs[-n:]
            self.versions.pop()
            self.n_format_errors += 1
            self._consumed = self._render(messages, add_generation_prompt=False)
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
