cd /data/phd/yaozhengjian/Code/RL/ART-FRv2/DiffusionNFT
source /data/phd/yaozhengjian/miniconda3/bin/activate
conda activate /data/phd/yaozhengjian/zjYao_Envs/DiffusionNFT


export WANDB_ENTITY=your_wandb_entity
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_PROJECT=DiffusionNFT_PrefRestore
torchrun --nproc_per_node=8 --master_port=11234 scripts/train_nft_prefRestore.py --config config/pref_restore.py:pref_restore_multi_reward


torchrun --nproc_per_node=8 --master_port=11234 scripts/train_nft_prefRestore.py --config config/pref_restore.py:pref_restore_multi_reward_ffhq



export WANDB_PROJECT=DiffusionNFT_PrefRestore_debug
CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=8 --master_port=11234 scripts/train_nft_prefRestore.py --config config/pref_restore.py:pref_restore_multi_reward

