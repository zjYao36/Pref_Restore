cd /data/phd/yaozhengjian/Code/RL/ART-FRv2/DiffusionNFT
source /data/phd/yaozhengjian/miniconda3/bin/activate
conda activate /data/phd/yaozhengjian/zjYao_Envs/DiffusionNFT


export WANDB_ENTITY=your_wandb_entity
export WANDB_API_KEY='246286f3e4e4f0f6075dc23780b95a3c8fb523c7'
export WANDB_PROJECT=prefRestore
torchrun --nproc_per_node=8 --master_port=11234 scripts/train_nft_prefRestore.py --config config/pref_restore_gt.py:pref_restore_gt_reward
