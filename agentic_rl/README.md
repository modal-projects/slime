# agentic_rl — fully-async SWE agent RL for slime

This package trains coding agents on the Python subset of SWE-Rebench V2 using
stock mini-swe-agent, Modal Sandboxes, and slime.

## Architecture

The agent loop runs in-process. mini-swe calls a pluggable model object, so
`RecordingModel` sends exact token IDs to SGLang and records the returned token
IDs, rollout log probabilities, and weight version. Only bash commands cross
into the task sandbox.

```text
fully_async_rollout
  └─ generate()                         one call per episode
       ├─ stock mini-swe DefaultAgent
       │    ├─ RecordingModel → SGLang  exact tokens and logprobs
       │    └─ Sandbox → Modal          bash only
       ├─ capture the source patch
       └─ grade in a fresh sandbox      held-out tests stay isolated
```

Generated assistant tokens have `loss_mask=1`; rendered prompts and tool
observations have `loss_mask=0`. A weight update aborts and recycles the whole
sample group, so partial old-policy trajectories never enter training.

## Files

| File | Role |
| --- | --- |
| `generate.py` | Runs one bounded episode and builds the token-faithful slime sample |
| `model.py` | Implements mini-swe's model protocol over SGLang |
| `sandbox.py` | Implements mini-swe's bash environment with Modal Sandbox |
| `grade.py` | Applies the model patch and held-out test patch in a fresh sandbox |
| `metrics.py` | Logs agent behavior, tail latency, and policy-version lag |
| `prompts.py` | Pins the agent prompt and action protocol |

## Slime integration

The accompanying slime changes are intentionally small:

- bound the fully-async generation lead with `--rollout-max-staleness`;
- apply dynamic sampling while draining the continuous rollout pool;
- reset every sibling to `PENDING` when an aborted group is requeued;
- exclude fully masked placeholders from GRPO reward normalization; and
- avoid loading a multimodal processor for text-only Qwen3.6 data.

`training_gym.patch` contains only those core diffs so a launcher can apply
them on top of its pinned slime image without replacing the full source tree.

Wire the package through slime's standard hooks:

```text
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
--custom-generate-function-path agentic_rl.generate.generate
--custom-rollout-log-function-path agentic_rl.metrics.log_rollout_data
```

Episode limits are supplied through `--custom-config-path`, including
`agentic_max_steps`, `agentic_episode_timeout`, `agentic_exec_timeout`,
`agentic_grade_timeout`, `agentic_query_timeout`, and
`agentic_max_boot_retries`.

The runtime image must include `modal` and `mini-swe-agent`. Dataset preparation
is launcher-owned: rows must contain `prompt`, `label`, and SWE-Rebench metadata
(`instance_id`, `image_name`, `repo`, `install_config`, `test_patch`,
`FAIL_TO_PASS`, and `PASS_TO_PASS`).
