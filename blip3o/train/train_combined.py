"""
Ablation training script: all three stages combined in one pass.

Trainable modules (union of Stages 1–3):
  Stage 1 modules  : Qwen LM layers, embed_tokens, lm_head,
                     diffusion_connector, sana.caption_projection
  Stage 2 modules  : sana.transformer_blocks, vae_connector
  Stage 3 modules  : sana_vae.encoder

Losses (all active simultaneously):
  CE loss   (Stage 1)   —  LLM token prediction
  Diff loss (Stage 2)   —  flow-matching on SANA
  VAE MSE   (Stage 3)   —  align degraded VAE latent toward clean latent
"""

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import deepspeed
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from tabulate import tabulate

from blip3o.data import make_supervised_data_module
from blip3o.model import blip3oQwenForCausalLMCombined
from blip3o.train.blip3o_trainer import blip3oTrainer
from blip3o.utils import rank0_print

try:
    from numpy._core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([LossScaler, _reconstruct, np.ndarray, np.dtype])
except ImportError:
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([LossScaler, _reconstruct, np.ndarray, np.dtype])

torch.multiprocessing.set_sharing_strategy("file_system")
local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    diffusion_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None)
    mm_tunable_parts: Optional[str] = field(default="mm_language_model")
    vae_connector: Optional[bool] = field(default=True)   # always True for combined
    version: Optional[str] = field(default="v0")
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_use_im_start_end: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)
    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)
    delay_load: Optional[bool] = field(default=True)
    num_image_tokens: Optional[int] = field(default=-1)
    image_token_format: str = field(default="<I{}>")
    num_scale_tokens: Optional[int] = field(default=3)
    scale_token_format: str = field(default="<S{}>")
    load_embeddings_from_vision: Optional[bool] = field(default=False)
    # loss weights — default 1.0 each
    ce_loss_weight: float = field(default=1.0)
    diff_loss_weight: float = field(default=1.0)
    vae_loss_weight: float = field(default=1.0)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    dataset_cls: str = field(default="blip3o")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=4096)
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2")
    dispatch_batches: Optional[bool] = field(default=None)
    split_batches: Optional[bool] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def get_model(model_args, training_args):
    customized_kwargs = {}
    overwrite_config = {}

    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = (
                cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            )
            overwrite_config["max_sequence_length"] = training_args.model_max_length

    if overwrite_config:
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    model = blip3oQwenForCausalLMCombined.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs,
    )
    return model


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model = get_model(model_args, training_args)
    model.config.use_cache = False

    # propagate loss weights into model config so forward() can read them
    model.config.ce_loss_weight   = model_args.ce_loss_weight
    model.config.diff_loss_weight = model_args.diff_loss_weight
    model.config.vae_loss_weight  = model_args.vae_loss_weight

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.diffusion_name_or_path = model_args.diffusion_name_or_path
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        rank0_print(f"[Combined ablation] mm_tunable_parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts

        # ------------------------------------------------------------------ #
        # Freeze everything first, then selectively unfreeze
        # ------------------------------------------------------------------ #
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        vision_tower.eval()

        tunable_parts = model_args.mm_tunable_parts.split(",")

        # Stage 1: Qwen LM (all non-vision-tower params)
        if "mm_language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name:
                    param.requires_grad_(True)

        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)

        if "mm_embedding" in tunable_parts:
            for name, param in model.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad_(True)

        # ------------------------------------------------------------------ #
        # Combined module unfreeze: apply after the tunable_parts logic so
        # the fine-grained rules below take precedence.
        # ------------------------------------------------------------------ #
        for name, param in model.named_parameters():
            # Freeze the whole SANA transformer first (will re-open below)
            if "sana" in name:
                param.requires_grad_(False)

            # Stage 1: caption projection inside SANA stays trainable
            if "caption" in name:
                param.requires_grad_(True)

            # Stage 2: SANA transformer blocks
            if "transformer_blocks" in name:
                param.requires_grad_(True)

            # Stage 2: vae_connector
            if "vae_connector" in name and model_args.vae_connector:
                param.requires_grad_(True)

            # Stage 3: VAE encoder (trainable; fixed_vae is separate & frozen)
            if "sana_vae.encoder" in name:
                param.requires_grad_(True)

        # ------------------------------------------------------------------ #
        # Log trainable parameters
        # ------------------------------------------------------------------ #
        total_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel()
            for p in model.parameters()
        )
        trainable_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel()
            for p in model.parameters() if p.requires_grad
        )
        rank0_print(f"Total parameters:     ~{total_params / 1e6:.2f} M")
        rank0_print(f"Trainable parameters: ~{trainable_params / 1e6:.2f} M")

        trainable_params_file = "trainable_parameters_combined.txt"
        with open(trainable_params_file, "w") as f:
            f.write("=== Combined Ablation — Trainable Parameters ===\n\n")
            total = 0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    cnt = p.numel()
                    f.write(f"Parameter: {name}\n  Shape: {list(p.shape)}\n  Count: {cnt:,}\n\n")
                    total += cnt
            f.write(f"=== Summary ===\n")
            f.write(f"Total trainable parameters: {total:,}\n")
            f.write(f"Total trainable parameters (M): {total / 1e6:.2f}M\n")
        rank0_print(f"Trainable parameter list saved to: {trainable_params_file}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = blip3oTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
