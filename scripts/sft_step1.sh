# conda  activate  /data/phd/yaozhengjian/zjYao_Envs/blip3o-next
# resume的话直接用之前的的目录 {LOCAL_DIR} 就可以
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_PROJECT=blip3o_next
export HF_HOME=/your/hf/home/

VISION_MODEL=/data/phd/hf_models/Unified-Models/TA-Tok/ta_tok.pth
PRETRAINED_MODEL=/data/phd/hf_models/Unified-Models/BLIP3o/BLIP3o-NEXT-SFT-3B
DIFFUSION=/data/phd/hf_models/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
DATA_PATH=/data/phd/yaozhengjian/zjYao_Datasets/Pref-Restore/FFHQ/zjyao_data_txt/train_data_all_caption.txt   # 10 个epoch是 16760 iters


EPOCH=200
LR=1e-4 
RUN_NAME="Face-Restoration_FFHQ_Step1_ablation_training-curve_V6-LR1-4bsz16"


echo "PRETRAINED_MODEL: ${PRETRAINED_MODEL}"
echo "DIFFUSION: ${DIFFUSION}"
echo "RUN_NAME: ${RUN_NAME}"
LOCAL_DIR="/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Models/Rebuttal/${RUN_NAME}"
mkdir -p ${LOCAL_DIR}


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
    --dataloader_drop_last True 
