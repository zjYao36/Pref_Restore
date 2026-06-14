import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _get_config(base_model="prefRestore", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    config = base.get_config()
    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")

    config.pretrained.model = "/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Models/Face-Restoration_FFHQ_VAE_Step3_scaling+Text+Recon-V2_3/checkpoint-128000"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 30
    config.sample.guidance_scale = 2.0
    config.resolution = 512
    config.train.beta = 0.0001
    config.sample.noise_level = 0.7
    bsz = 9

    config.sample.num_image_per_prompt = 24
    num_groups = 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    config.sample.test_batch_size = bsz
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


# ============================================================
# GT-aware reward configs (LMD + ArcFace)
# ============================================================

def pref_restore_gt_reward():
    """All 6 rewards: PickScore + HPSv2 + CLIPScore + LMD + ArcFace + lpips (restore_face dataset)"""
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
        "lmd": 0.5,
        "arcface": 0.5,
        "lpips": 0.5, 
    }
    config = _get_config(
        base_model="prefRestore",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="restore_face_codeformer",
        reward_fn=reward_fn,
        name="codeformer_6_reward_0p5_scaling",
    )
    config.run_name = "prefRestore_codeformer_6_reward_0p5_scaling"
    config.sample.num_steps = 30
    config.beta = 0.1
    config.train.lora_path = "/data/phd/yaozhengjian/Code/RL/ART-FRv2/DiffusionNFT/logs/nft/prefRestore/gt_reward_6指标-CodeformerData/checkpoints/checkpoint-180/lora"
    return config

# ============================================================
# Original configs (kept for compatibility)
# ============================================================

def pref_restore_multi_reward():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="prefRestore",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="restore_face",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.run_name = f"prefRestore_multi-reward_dosampleFalse"
    config.sample.num_steps = 30
    config.beta = 0.1
    return config