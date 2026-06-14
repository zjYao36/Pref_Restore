# conda  activate  /data/phd/yaozhengjian/zjYao_Envs/blip3o-next
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_PROJECT=blip3o_next
export HF_HOME=/your/hf/home/

VISION_MODEL=/data/phd/hf_models/Unified-Models/TA-Tok/ta_tok.pth
AR_BACKBONE=/data/phd/hf_models/Qwen3-0.6B
DIFFUSION=/data/phd/hf_models/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
DATA_PATH=/data/phd/data/unified_model/T2I_sft/BLIP3o-60k
EPOCH=100
LR=5e-5
RUN_NAME="pretrain_data_60k"

echo "AR_BACKBONE: ${AR_BACKBONE}"
echo "DIFFUSION: ${DIFFUSION}"
echo "RUN_NAME: ${RUN_NAME}"
LOCAL_DIR="/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/${RUN_NAME}"
mkdir -p ${LOCAL_DIR}


# srun torchrun --nproc_per_node=8  --nnodes=$SLURM_NNODES \
#     --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29501 blip3o/train/train.py \
torchrun --nproc_per_node=8  \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 \
    blip3o/train/train.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path ${AR_BACKBONE} \
    --diffusion_name_or_path  ${DIFFUSION} \
    --version "qwen_1_5" \
    --dataset_cls 'mix' \
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
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True 
