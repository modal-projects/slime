import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2

U.exec_command("mkdir -p /root/models /root/datasets")
U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
U.hf_download_dataset("zhuzilin/dapo-math-17k")
U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models")

ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}_torch_dist "
rollout_args = (
    "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
    "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
    "--rm-type deepscaler --num-rollout 1 --rollout-batch-size 4 --n-samples-per-prompt 4 "
    "--rollout-max-response-len 1024 --rollout-temperature 1.0 --global-batch-size 16 "
    "--use-dynamic-batch-size --max-tokens-per-gpu 4096 "
)
optimizer_args = "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
ppo_args = "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type k1 --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 4e-4 "
peft_args = "--peft-type lora --lora-rank 16 --lora-alpha 16 --lora-dropout 0.0 --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 "
ci_args = "--ci-test --ci-disable-kl-checker "
sglang_args = "--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.70 --sglang-chunked-prefill-size 2048 "
misc_args = (
    "--actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate "
    "--attention-backend flash --attention-dropout 0.0 --hidden-dropout 0.0 "
    "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
    "--megatron-to-hf-mode bridge --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
)
parallel_args = "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "

train_args = ckpt_args + rollout_args + optimizer_args + ppo_args + peft_args + ci_args + sglang_args + misc_args + parallel_args
print("[PEFT NO-CI-EXACT CONTROL] train_args:", train_args)
U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)
print("[PEFT NO-CI-EXACT CONTROL] completed")
