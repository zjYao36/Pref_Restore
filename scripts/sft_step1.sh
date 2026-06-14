# SFT Step 1 — start from the BLIP3o-NEXT-SFT-3B backbone.
# For resume training, just point LOCAL_DIR to the previous output_dir.
#
# Required edits before running:
#   - WANDB_API_KEY:    your W&B key (or remove --report_to wandb)
#   - HF_HOME:          your local HuggingFace cache dir
#   - VISION_MODEL:     local path to ta_tok.pth (from the BLIP-3o-NEXT release)
#   - DIFFUSION:        local path to a SANA 1.5 1024px diffusers folder
#   - DATA_PATH:        plain-text manifest; each line is a directory containing
#                       parquet/tar shards with columns {image, txt}
#   - LOCAL_DIR:        where to write checkpoints
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_PROJECT=prefRestore
export HF_HOME=/path/to/your/hf_cache

VISION_MODEL=/path/to/TA-Tok/ta_tok.pth
PRETRAINED_MODEL=BLIP3o/BLIP3o-NEXT-SFT-3B   # HF repo id; transformers will auto-download
DIFFUSION=/path/to/SANA1.5_1.6B_1024px_diffusers
DATA_PATH=/path/to/your/train_data.txt

EPOCH=200
LR=1e-4
RUN_NAME="Face-Restoration_FFHQ_Step1"


echo "PRETRAINED_MODEL: ${PRETRAINED_MODEL}"
echo "DIFFUSION: ${DIFFUSION}"
echo "RUN_NAME: ${RUN_NAME}"
LOCAL_DIR="./checkpoints/${RUN_NAME}"
mkdir -p ${LOCAL_DIR}


# Note: --torch_compile True triggers a one-off compilation that takes ~5-10 min
#       on first launch; set to False if you want a faster startup for debugging.
torchrun --nproc_per_node=8  \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29503 \
    blip3o/train/train_step1.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --diffusion_name_or_path  ${DIFFUSION} \
    --version "qwen_1_5" \
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
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 10000 \
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
    --dataloader_drop_last True
