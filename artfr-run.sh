#!/usr/bin/env bash
# ============================================================================
# Pref-Restore — end-to-end command reference
# ----------------------------------------------------------------------------
# This file lists the exact commands for the full two-stage pipeline:
#   Stage A (SFT)  : hierarchical SFT of the BLIP-3o backbone   -> env `art-fr`
#   Stage B (RL)   : preference optimization with DiffusionNFT  -> env `DiffusionNFT`
#   Inference      : base (SFT) model  &  RL-finetuned LoRA model
#
# NOTE: paths of the form /data/phd/yaozhengjian/... are the author's local
#       paths. Replace them with your own before running (see README.md).
# ============================================================================
set -e

REPO=/data/phd/yaozhengjian/Code/RL/ART-FRv2          # <- edit me
CONDA=/data/phd/yaozhengjian/miniconda3/bin/activate  # <- edit me
ENV_SFT=art-fr                                        # <- your art-fr env
ENV_RL=DiffusionNFT                                   # <- your DiffusionNFT env


# ----------------------------------------------------------------------------
# Stage A — Supervised fine-tuning (SFT) of the backbone           [env: art-fr]
#   step1: SFT from the BLIP3o-NEXT-SFT-3B backbone
#   step2: VAE encoder + diffusion head
#   (toggle caption / reconstruction options in blip3o/data/dataset.py)
# ----------------------------------------------------------------------------
cd "$REPO"
source "$CONDA"; conda activate "$ENV_SFT"
bash scripts/sft_step1.sh
bash scripts/sft_step2.sh


# ----------------------------------------------------------------------------
# Stage B — Preference RL with DiffusionNFT                  [env: DiffusionNFT]
#   main pipeline       : train_nft_prefRestore.py  + config/pref_restore.py
#   GT-aware variant    : train_nft_prefRestore_gt.py via scripts/run_gt.sh
# ----------------------------------------------------------------------------
cd "$REPO/DiffusionNFT"
source "$CONDA"; conda activate "$ENV_RL"
export WANDB_PROJECT=DiffusionNFT_PrefRestore
# main (multi-reward) preference optimization
torchrun --nproc_per_node=8 --master_port=11234 \
    scripts/train_nft_prefRestore.py \
    --config config/pref_restore.py:pref_restore_multi_reward
# GT-aware reward variant (LMD + ArcFace), see scripts/run_gt.sh for all configs
# bash scripts/run_gt.sh pref_restore_gt_reward


# ----------------------------------------------------------------------------
# Inference — base (SFT) model                                     [env: art-fr]
# ----------------------------------------------------------------------------
cd "$REPO"
source "$CONDA"; conda activate "$ENV_SFT"
CUDA_VISIBLE_DEVICES=0 \
python inference_batch_noPrompt_fixLQ_vae.py \
    --model_path /path/to/SFT_checkpoint \
    --json_path  /path/to/Eval/captions_lq.json \
    --output_dir /path/to/results/base


# ----------------------------------------------------------------------------
# Inference — RL-finetuned LoRA model                        [env: DiffusionNFT]
# ----------------------------------------------------------------------------
cd "$REPO"
source "$CONDA"; conda activate "$ENV_RL"
CUDA_VISIBLE_DEVICES=0 \
python inference_batch_noPrompt_fixLQ_vae_lora.py \
    --model_path /path/to/SFT_checkpoint \
    --json_path  /path/to/Eval/captions_lq.json \
    --output_dir /path/to/results/rl \
    --lora_path  /path/to/DiffusionNFT/logs/.../checkpoints/checkpoint-XXX \
    --use_lora
