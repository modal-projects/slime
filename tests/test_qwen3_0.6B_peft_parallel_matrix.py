import os

import slime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2


MATRIX_CASES = [
    {"name": "dp2_tp1_pp1", "tp": 1, "pp": 1, "ep": 1, "etp": 1},
    {"name": "dp1_tp2_pp1", "tp": 2, "pp": 1, "ep": 1, "etp": 1},
    {"name": "dp1_tp1_pp2", "tp": 1, "pp": 2, "ep": 1, "etp": 1},
]


def _selected_cases():
    raw = os.environ.get("SLIME_TEST_PEFT_MATRIX_CASES", "").strip()
    if not raw:
        return MATRIX_CASES

    wanted = {x.strip() for x in raw.split(",") if x.strip()}
    selected = [case for case in MATRIX_CASES if case["name"] in wanted]
    if not selected:
        raise ValueError(
            f"SLIME_TEST_PEFT_MATRIX_CASES={raw!r} did not match any cases. "
            f"Available: {[c['name'] for c in MATRIX_CASES]}"
        )
    return selected


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )


def _build_base_train_args():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 16 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    peft_args = (
        "--peft-type lora "
        "--lora-rank 16 "
        "--lora-alpha 16 "
        "--lora-dropout 0.0 "
        "--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 "
    )

    ci_args = (
        "--ci-test "
        "--ci-peft-exact "
        "--ci-disable-kl-checker "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-mem-fraction-static 0.70 "
        "--sglang-chunked-prefill-size 2048 "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--colocate "
        "--attention-backend flash "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--megatron-to-hf-mode bridge "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{peft_args} "
        f"{ci_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{U.get_default_wandb_args(__file__)} "
    )


def execute():
    base_train_args = _build_base_train_args()

    for case in _selected_cases():
        parallel_args = (
            f"--tensor-model-parallel-size {case['tp']} "
            f"--pipeline-model-parallel-size {case['pp']} "
            f"--expert-model-parallel-size {case['ep']} "
            f"--expert-tensor-parallel-size {case['etp']} "
        )
        if case["tp"] > 1:
            parallel_args += "--sequence-parallel "

        # Add a deterministic tag to simplify log grep in failures.
        case_tag = (
            "--wandb-run-name "
            f"peft-parallel-{case['name']}-{U.create_run_id()} "
            "--disable-wandb-random-suffix "
            if os.environ.get("WANDB_API_KEY")
            else ""
        )

        train_args = f"{base_train_args} {parallel_args} {case_tag}"

        print(f"[PEFT Matrix] Running case: {case['name']} ({case})")
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
