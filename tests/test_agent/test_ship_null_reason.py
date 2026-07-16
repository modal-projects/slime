"""The dashboard's view of a no-usable-turn ("null") episode.

Pure stdlib target (``agentic_rl.dashboard.convert``) — no modal/slime/torch
imports — so this runs without a GPU/sandbox. It locks the contract between
``generate._ship_null`` (producer) and ``convert.sample_view`` (consumer): a
discarded episode must surface its *real* exit reason, and a rolled-back terminal
generation must render instead of showing a blank trajectory.
"""

from agentic_rl.dashboard import convert


def _null_sample(agentic: dict) -> dict:
    """A sample exactly as ``_ship_null`` writes it: empty response, fully masked,
    reward 0, with the given ``agentic`` metadata block."""
    return {
        "index": 0,
        "status": "completed",
        "reward": 0.0,
        "remove_sample": True,
        "tokens": list(range(513)),
        "response": "",
        "response_length": 1,
        "loss_mask": [0],
        "rollout_log_probs": [0.0],
        "weight_versions": [],
        "metadata": {"instance_id": "frontier-cs-algorithm-62", "agentic": agentic},
    }


def test_runaway_think_null_surfaces_reason_and_renders_generation():
    # _ship_null carries the model's exit_status plus the rolled-back generation.
    runaway = "<think>\n" + "reconsider the constraints. " * 500  # unclosed runaway think
    s = _null_sample(
        {
            "exit_status": "ContextLengthExceeded",
            "turns": 0,
            "truncated_tail": {"text": runaway, "tokens": 30000, "finish": "length"},
            "full_prompt": "<|im_start|>system\n...<|im_end|>",
        }
    )
    v = convert.sample_view(s)

    # The real reason is surfaced (was a blanket "ImageUnusable" before the fix).
    assert v["exit_status"] == "ContextLengthExceeded"
    assert v["truncated_tail"] == {"tokens": 30000, "finish": "length"}
    assert v["full_prompt"]

    # The rolled-back generation renders as a synthesized turn instead of nothing.
    assert v["n_turns"] == 1
    tg = v["turns"][0]["truncated_generation"]
    assert tg["finish"] == "length" and tg["closed_think"] is False
    assert tg["think"] and "reconsider the constraints" in tg["think"]


def test_boot_failure_null_keeps_image_unusable_and_no_turn():
    # No LLM call ever ran (empty chains): fallback reason, no generation to show.
    v = convert.sample_view(_null_sample({"exit_status": "ImageUnusable", "turns": 0}))
    assert v["exit_status"] == "ImageUnusable"
    assert v["n_turns"] == 0
    assert v["truncated_tail"] is None


def test_normal_episode_has_no_exit_status():
    # A completed multi-turn sample carries no exit_status (None, not surfaced as a reason).
    s = {
        "index": 1,
        "status": "completed",
        "reward": 1.0,
        "tokens": [1, 2, 3],
        "response": "<|im_start|>assistant\ndone<|im_end|>",
        "response_length": 3,
        "loss_mask": [1, 1, 1],
        "metadata": {"instance_id": "x", "agentic": {"turns": 1}},
    }
    assert convert.sample_view(s)["exit_status"] is None
