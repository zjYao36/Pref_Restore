import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import deepspeed
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from blip3o.data import make_supervised_data_module
from blip3o.model import blip3oQwenForCausalLMVAE
from blip3o.train.blip3o_trainer import blip3oTrainer
from blip3o.utils import rank0_print
from tabulate import tabulate
# 添加numpy相关的安全全局对象
import numpy as np
try:
    # 尝试使用新的API
    from numpy._core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([
        LossScaler,
        _reconstruct,
        np.ndarray,
        np.dtype,
    ])
except ImportError:
    # 如果新API不存在，回退到旧API
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([
        LossScaler,
        _reconstruct,
        np.ndarray,
        np.dtype,
    ])
torch.multiprocessing.set_sharing_strategy("file_system")
local_rank = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    diffusion_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from blip3oLlama, blip3oMixtral, blip3oMistral, Llama"})
    mm_tunable_parts: Optional[str] = field(default="sana")
    vae_connector: Optional[bool] = field(default=False)  # !! Step2 new
    version: Optional[str] = field(default="v0")
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
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

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in blip3o's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
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
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
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
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_model(model_args, training_args):
    customized_kwargs = {}
    overwrite_config = {}

    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path) # blip3oQwenConfig 被注册了
    '''——————————————————————————————————————默认不使用——————————————————————————————————————————'''
    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range
    '''
    1. 扩展模型的上下文窗口
    这是这段代码最主要的功能。许多预训练的语言模型（如 Llama, Mistral）都有一个固定的最大序列长度（例如 4096 个 token）。如果你想让模型处理更长的文本，就需要使用一些技巧来扩展它的上下文窗口。

    代码中的 rope_scaling 部分就是为了实现这个目的：

    RoPE (Rotary Position Embedding) 是一种先进的位置编码技术。
    RoPE Scaling 是一种微调技术，通过调整 RoPE 的计算方式，让模型能够理解比原始训练长度更长的序列。
    实际操作：当你在启动训练时传入 rope_scaling_factor（例如 4.0）和 rope_scaling_type（例如 "linear" 或 "dynamic")，这段代码会自动计算出新的 model_max_length（例如 4096 * 4.0 = 16384），并把这些配置更新到模型中。这样，你加载的模型就能在微调时处理 16384 长度的序列了。
    '''
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
    '''——————————————————————————————————————————————————————————————————————————————————————————'''
    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    model = blip3oQwenForCausalLMVAE.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs)
    return model


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    model = get_model(model_args, training_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"): # 走的这个逻辑
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right")
    if tokenizer.unk_token is not None: # 不走这个逻辑
        tokenizer.pad_token = tokenizer.unk_token # Qwen\Blip-3o: "pad_token": "<|endoftext|>", "unk_token": null
    
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp) # pretrain时Qwen在这步加载其他模块 ta-tok, sana, sana_vae, diffusion_connector

        vision_tower = model.get_vision_tower() # TA-Tok
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor # TA-Tok的image_processor SiglipImageProcessor()
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio # square
        model.config.diffusion_name_or_path = model_args.diffusion_name_or_path

        
        model.config.tokenizer_padding_side = tokenizer.padding_side # right
        model.config.tokenizer_model_max_length = tokenizer.model_max_length # .sh传入 2048

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end # True
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr # None
        training_args.use_im_start_end = model_args.mm_use_im_start_end # True

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer) # 给tokenizer新增 scale token, image token, start_end token

        ### Deciding train which part of the model 
        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}") # mm_language_model 训练除了vision tower的其他部分
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        # Set the entire model to not require gradients by default
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        vision_tower.eval()
        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name:
                    param.requires_grad_(True)
        if 'mm_embedding' in tunable_parts:
            for name, param in model.named_parameters():
                if "embed_tokens" in name or 'lm_head' in name:
                    param.requires_grad_(True)

        ## freeze sana except the caption projection
        for name, param in model.named_parameters(): # 第二阶段为了保证一致性，完全打开sana训练
            if "sana" in name:
                param.requires_grad_(False)

            if 'transformer_blocks' in name:  # unfreeze transformer blocks
                param.requires_grad_(True)

            if 'vae_connector' in name and model_args.vae_connector:  # unfreeze vae connector
                param.requires_grad_(True)

            if 'sana_vae.encoder' in name:  # unfreeze sana vae encoder
                param.requires_grad_(True)

        # for name, param in model.named_parameters():
        #     if "caption" in name:
        #         param.requires_grad_(True)   
                

        # 可训练参数汇总~1874.48 MB Qwen Embedding, Qwen layers, sana caption projection, diffusion_connector,
        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         rank0_print(f"Trainable parameter: {name}")

        # 创建文件来保存可训练参数信息
        trainable_params_file = "trainable_parameters.txt"
        with open(trainable_params_file, "w") as f:
            f.write("=== Trainable Parameters ===\n\n")
            total_params = 0
            
            for name, p in model.named_parameters():
                if p.requires_grad:
                    param_count = p.numel()
                    param_info = f"Parameter: {name}\n  Shape: {list(p.shape)}\n  Count: {param_count:,}\n\n"
                    f.write(param_info)
                    total_params += param_count
            
            f.write(f"=== Summary ===\n")
            f.write(f"Total trainable parameters: {total_params:,}\n")
            f.write(f"Total trainable parameters (M): {total_params/1e6:.2f}M\n")
        rank0_print(f"Trainable parameters saved to: {trainable_params_file}")
        rank0_print(f"Total trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)")


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
