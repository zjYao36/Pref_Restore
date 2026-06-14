#!/bin/bash
# Ablation study: all three stages trained jointly in a single run.
# Losses   : CrossEntropy (Stage-1) + Diffusion (Stage-2) + VAE MSE (Stage-3)
# Modules  : Qwen LM + diffusion_connector + sana.caption_projection   (Stage-1)
#          + sana.transformer_blocks + vae_connector                    (Stage-2)
#          + sana_vae.encoder                                           (Stage-3)
#
# PRETRAINED_MODEL = same checkpoint as Stage-1's starting point, so that
# both the staged pipeline and this combined run start from an identical base.
# New modules (vae_connector, sana_vae.encoder) are randomly initialized,
# exactly as they are when Stage-2 loads a Stage-1 checkpoint.
#
# resume: just keep LOCAL_DIR unchanged and re-run.

export WANDB_API_KEY='your_wandb_api_key'
export WANDB_PROJECT=blip3o_next
export HF_HOME=/your/hf/home/

VISION_MODEL=/data/phd/hf_models/Unified-Models/TA-Tok/ta_tok.pth
# Same starting point as Stage-1 — ensures a fair comparison against the staged pipeline
PRETRAINED_MODEL=/data/phd/hf_models/Unified-Models/BLIP3o/BLIP3o-NEXT-SFT-3B
DIFFUSION=/data/phd/hf_models/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
DATA_PATH=/data/phd/yaozhengjian/zjYao_Datasets/Pref-Restore/FFHQ/zjyao_data_txt/train_data_all_caption.txt

EPOCH=200
LR=2e-5
RUN_NAME="Face-Restoration_FFHQ_Combined_Ablation_AllStages"

echo "PRETRAINED_MODEL: ${PRETRAINED_MODEL}"
echo "DIFFUSION:        ${DIFFUSION}"
echo "RUN_NAME:         ${RUN_NAME}"

LOCAL_DIR="/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Models/Rebuttal/${RUN_NAME}"
mkdir -p ${LOCAL_DIR}

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29508 \
    blip3o/train/train_combined.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --diffusion_name_or_path ${DIFFUSION} \
    --version "qwen_1_5" \
    --vae_connector True \
    --dataset_cls 'restore' \
    --data_path ${DATA_PATH} \
    --dispatch_batches False \
    --vision_tower ${VISION_MODEL} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ${LOCAL_DIR} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 16000 \
    --save_total_limit 10 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True \
    --ce_loss_weight 1.0 \
    --diff_loss_weight 1.0 \
    --vae_loss_weight 1.0
