# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import re
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from slime.utils.misc import load_function


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True
            if bias:
                self.bias.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
        ) -> GPTModel:
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            return model

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        provider.finalize()

        # Apply PEFT transformation if enabled
        model_provider_func = provider.provide
        if getattr(args, "peft_type", "none") != "none":
            model_provider_func = wrap_model_provider_with_peft(model_provider_func, args)

        return model_provider_func

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        config: TransformerConfig = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        return model

    return model_provider


def wrap_model_provider_with_freeze(original_provider, args):
    def wrapped_provider(pre_process=True, post_process=True, vp_stage=None):
        sig = inspect.signature(original_provider)
        if "vp_stage" in sig.parameters:
            model = original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        else:
            model = original_provider(pre_process=pre_process, post_process=post_process)

        freeze_model_params(model, args)

        return model

    return wrapped_provider


def freeze_model_params(model: GPTModel, args: argparse.Namespace):
    if args.only_train_params_name_list:
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break

    if args.freeze_params_name_list:
        for name, param in model.named_parameters():
            for pattern in args.freeze_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break


def _freeze_non_lora_params(model, peft_type):
    """Freeze all parameters except LoRA adapter weights.

    ParallelLinearAdapter uses linear_in/linear_out for LoRA weights.
    LinearAdapter uses lora_a/lora_b for LoRA weights.

    We freeze everything except parameters whose names contain these LoRA-specific substrings.
    """
    # LoRA/DoRA adapter parameter name patterns
    lora_param_patterns = [
        ".linear_in.",  # ParallelLinearAdapter A matrix
        ".linear_out.",  # ParallelLinearAdapter B matrix
        ".lora_a.",  # LinearAdapter A matrix
        ".lora_b.",  # LinearAdapter B matrix
        ".weight_magnitude",  # DoRA per-row magnitude vector
    ]

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        is_lora_param = any(pattern in name for pattern in lora_param_patterns)
        if is_lora_param:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    print(f"PEFT freeze: {frozen_count} params frozen, {trainable_count} LoRA params trainable")


def wrap_model_provider_with_peft(original_provider, args):
    """Wrap model provider to apply PEFT (LoRA/DoRA) transformation.

    This wrapper:
    1. Creates the base model using the original provider
    2. Applies PEFT transformation recursively (injects LoRA/DoRA adapters)
    3. Freezes base model parameters (only adapters are trainable)

    Args:
        original_provider: The original model provider function
        args: Argument namespace with PEFT configuration

    Returns:
        Wrapped provider function that returns PEFT-transformed model
    """
    if args.peft_type == "none":
        return original_provider

    # Import PEFT classes from Megatron-Bridge
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.peft.dora import DoRA
    try:
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    except ImportError:
        # CanonicalLoRA may not be available in all versions
        CanonicalLoRA = None

    peft_classes = {
        "lora": LoRA,
        "dora": DoRA,
        "canonical_lora": CanonicalLoRA,
    }

    peft_cls = peft_classes.get(args.peft_type)
    if peft_cls is None:
        raise ValueError(f"Unknown or unavailable PEFT type: {args.peft_type}")

    # Create PEFT configuration
    peft_config = peft_cls(
        target_modules=args.lora_target_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    def wrapped_provider(pre_process=True, post_process=True, vp_stage=None):
        sig = inspect.signature(original_provider)
        if "vp_stage" in sig.parameters:
            model = original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        else:
            model = original_provider(pre_process=pre_process, post_process=post_process)

        # Apply PEFT transformation recursively using megatron-bridge's walk_utils
        # This properly traverses the model tree and applies transform to each module
        from megatron.bridge.peft.walk_utils import map as peft_map

        peft_map(model, peft_config.transform)

        # Count transformed modules for logging
        # Note: Megatron parallel models use ParallelLinearAdapter, not LinearAdapter
        adapter_count = 0
        try:
            from megatron.bridge.peft.lora import LinearAdapter, ParallelLinearAdapter
            adapter_count = sum(
                1 for _, m in model.named_modules()
                if isinstance(m, (LinearAdapter, ParallelLinearAdapter))
            )
            if adapter_count > 0:
                print(f"PEFT: Transformed {adapter_count} modules with LoRA adapters")
            else:
                print("PEFT Warning: No modules were transformed. Check target_modules config.")
        except ImportError:
            pass

        # Freeze base model parameters - only LoRA adapter weights should be trainable
        # ParallelLinearAdapter (unlike LinearAdapter) doesn't auto-freeze base weights
        if adapter_count > 0:
            _freeze_non_lora_params(model, args.peft_type)

        # Log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"PEFT ({args.peft_type}): {trainable_params:,} trainable params / {total_params:,} total params "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        return model

    return wrapped_provider
