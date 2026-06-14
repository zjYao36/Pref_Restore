#!/bin/bash
# GT-aware face restoration RL training with LMD + ArcFace rewards
# Usage:
#   bash scripts/run_gt.sh <config_name>
#
# Available configs:
#   pref_restore_gt_reward        - 5 rewards (PickScore+HPSv2+CLIPScore+LMD+ArcFace), restore_face dataset
#   pref_restore_gt_reward_ffhq   - 5 rewards, FFHQ dataset
#   pref_restore_gt_only          - GT-only (LMD+ArcFace), restore_face dataset
#   pref_restore_gt_only_ffhq     - GT-only (LMD+ArcFace), FFHQ dataset

export WANDB_PROJECT="flow-grpo-gt"

CONFIG_NAME=${1:-pref_restore_gt_reward}

torchrun \
    --nproc_per_node=8 \
    --master_port=12345 \
    scripts/train_nft_prefRestore_gt.py \
    --config config/pref_restore_gt.py:${CONFIG_NAME}
