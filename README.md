# Bridging Information Asymmetry: A Hierarchical Framework for Blind Face Restoration with Reduced Uncertainty

[![arXiv](https://img.shields.io/badge/arXiv-2601.19506-b31b1b.svg)](https://arxiv.org/abs/2601.19506)
[![TPAMI](https://img.shields.io/badge/IEEE-TPAMI%202026-004c97.svg)](https://www.computer.org/csdl/journal/tp)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> Official code release for our **IEEE TPAMI 2026** paper.
> 📄 **Paper:** [Bridging Information Asymmetry: A Hierarchical Framework for Blind Face Restoration with Reduced Uncertainty](https://arxiv.org/abs/2601.19506) (arXiv:2601.19506)

This repository contains the official **training and inference** code for **Pref-Restore**, a hierarchical framework for blind face restoration that bridges the information asymmetry between a degraded input and its high-quality target via reinforcement-learning–based preference optimization.

---

## 📖 Overview

Blind face restoration aims to reconstruct a detailed, high-quality face from a severely degraded input. The fundamental difficulty is **information asymmetry**: the sparse low-quality (LQ) input carries far less information than the dense high-quality (HQ) target, turning restoration into an ill-posed **one-to-many** problem that yields uncertainty and artifacts.

**Pref-Restore** is a hierarchical framework that *integrates discrete semantic logic with continuous texture generation*, and attacks the asymmetry from two complementary directions:

1. **Augmenting input density.** We employ an **auto-regressive integrator** to reformulate textual instructions into **dense latent queries**, injecting high-level semantics that compensate for the missing information in the LQ input.
2. **Pruning the output distribution.** We pioneer the integration of **on-policy reinforcement learning directly into the diffusion restoration loop**, aligning the model with perceptual preferences and **significantly reducing solution entropy** toward a deterministic, faithful reconstruction.

The result achieves **state-of-the-art performance across both synthetic and real-world benchmarks**.


---

## 📑 Table of Contents

- [Installation (environment setup)](#-installation-environment-setup)
- [Required external assets (weights & data)](#-required-external-assets-weights--data)
- [Training](#-training)
- [Inference](#-inference)
- [License](#-license)
- [Citation](#-citation)

---


## 🔧 Installation (environment setup)

This codebase requires **two separate Python environments**, because the SFT and RL stages depend on incompatible versions of `torch` / `accelerate` / `deepspeed`.

| Env | Used by | Python | Torch | Key packages |
|---|---|---|---|---|
| **`art-fr`** | SFT training (`blip3o/`) + base inference | 3.11 | 2.4 + cu124 | `accelerate==0.28.0`, `deepspeed==0.14.4`, `transformers==4.51.3`, `diffusers==0.34.0` |
| **`DiffusionNFT`** | preference-RL training (`DiffusionNFT/`) + RL / LoRA inference | 3.10 | 2.6 + cu126 | `accelerate==1.4.0`, `deepspeed==0.16.4`, `transformers==4.40.0`, `diffusers==0.33.1`, `flash-attn==2.7.4.post1`, `peft==0.10.0` |

> **Which env do I need?** Look at the top of `artfr-run.sh` — every command block is preceded by the right `conda activate` line.

### Environment 1 — `art-fr` (SFT + base inference)

```bash
conda create -n art-fr python=3.11 -y
conda activate art-fr

# PyTorch 2.4 + CUDA 12.4 (match your driver)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Project deps
pip install -r requirements.txt

# Install the BLIP-3o-NEXT package (this repo) and our modified BasicSR
pip install -e .
pip install -e BasicSR
```

### Environment 2 — `DiffusionNFT` (preference-RL training + RL / LoRA inference)

```bash
conda create -n DiffusionNFT python=3.10 -y
conda activate DiffusionNFT

# PyTorch 2.6 + CUDA 12.6 (match your driver)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install DiffusionNFT (pulls in flash-attn, deepspeed, peft, etc.)
pip install -e DiffusionNFT

# Also install this repo so the inference scripts can `import blip3o`
pip install -e .
```

---

## 📦 Required external assets (weights & data)

Everything below is **gitignored** and must be downloaded locally. Grab only what your target step needs:

| To run… | You need |
|---|---|
| **Inference** | a trained checkpoint (your Stage-A SFT / Stage-B RL output) + your own LQ images |
| **SFT training (Stage A)** | ① Backbone components + ③ Datasets |
| **Preference-RL training (Stage B)** | your SFT checkpoint + ② Reward models + ③ Datasets |

### ① Backbone components — needed to assemble the model for SFT (Stage A)

The model is built from **three separately-downloaded pieces**. The BLIP-3o-NEXT backbone is **not** self-contained — its code loads the TA-Tok tokenizer and the SANA decoder from external paths (the SFT scripts wire all three):

| Component | Download from | Wired via (in `scripts/sft_step*.sh`) | Role |
|---|---|---|---|
| **BLIP3o-NEXT-SFT-3B** (multimodal LLM backbone) | [HF: BLIP3o/BLIP3o-NEXT-SFT-3B](https://huggingface.co/BLIP3o/BLIP3o-NEXT-SFT-3B) ([code](https://github.com/JiuhaiChen/BLIP3o)) | `--model_name_or_path` ( `PRETRAINED_MODEL=` ) | auto-regressive backbone |
| `ta_tok.pth` (TA-Tok image tokenizer) | the TA-Tok / BLIP-3o-NEXT release | `--vision_tower` ( `VISION_MODEL=` ) — must be passed externally | image tokenizer |
| SANA 1.5 diffusion decoder (a diffusers folder) | [Efficient-Large-Model / SANA1.5](https://huggingface.co/Efficient-Large-Model) | `--diffusion_name_or_path` ( `DIFFUSION=` ) | diffusion decoder |

> **You only need these three to train Stage A from scratch.** For **inference and RL (Stage B)** you pass your **trained Stage-A SFT checkpoint** as `--model_path` / `model_name_or_path` — it loads the fully-assembled model, so you do not re-supply the raw backbone, TA-Tok, or SANA files.

### ② Reward models — only for RL training (Stage B) → place under `DiffusionNFT/reward_ckpts/`

The reward code expects these **exact paths** (see `DiffusionNFT/reward_ckpts/README.md`):

| File / folder | Download from | Exact local path |
|---|---|---|
| LAION CLIP-ViT-H/14 | [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | `DiffusionNFT/reward_ckpts/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/` |
| PickScore v1 | [yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1) | `DiffusionNFT/reward_ckpts/yuvalkirstain/PickScore_v1/` |
| OpenAI CLIP-ViT-L/14 | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | `DiffusionNFT/reward_ckpts/openai/clip-vit-large-patch14/` |
| HPS v2.1 | [tgxs002/HPSv2](https://github.com/tgxs002/HPSv2) | `DiffusionNFT/reward_ckpts/HPS_v2.1_compressed.pt` |

> The default config (`pref_restore_multi_reward`) uses the four models above. The **GT-aware variant only** (`bash scripts/run_gt.sh`) additionally needs landmark (LMD) + ArcFace identity models — see `DiffusionNFT/config/pref_restore_gt.py`. Skip those if you use the default.

### ③ Datasets

| Data | Download from | Used by |
|---|---|---|
| FFHQ-256 / FFHQ-512 | FFHQ | SFT + RL — HQ targets |
| CelebA-HQ | CelebA-HQ | SFT + RL — train / val |
| Real-world FR test sets (LFW / WebPhoto / WIDER / CelebChild) **or your own photos** | standard blind-FR benchmarks | inference inputs |

> You only need **high-quality (HQ) face images + one caption per image** to train. The degraded **low-quality (LQ) inputs are synthesized on the fly** during training (blur · down-sampling · noise · JPEG), so you do **not** pre-build LQ/HQ pairs.

#### How to organize the SFT training data

The SFT scripts take `--data_path` = a **plain-text manifest** (`train_data*.txt`). **Each line is a directory path**; every such directory is scanned recursively for `.parquet` (or `.tar` / WebDataset) shards:

```text
# train_data.txt  — one dataset directory per line
/your/data/FFHQ/parquet
/your/data/CelebA-HQ/parquet
```

Each shard must provide two columns:

| Column | Content |
|---|---|
| `image` | the HQ face image (decoded by 🤗 `datasets` as a PIL image) |
| `txt`   | a caption describing the image (a `text` column is auto-renamed to `txt`; leave empty for caption-free data) |

At training time each HQ `image` is degraded on the fly and the model learns **LQ → HQ**; a fraction of samples keep the original image as a pure reconstruction task. The caption is woven into the instruction ~90% of the time. See `blip3o/data/dataset.py` (`LazySupervisedRestoreDataset`) for the exact logic and the degradation parameters.

#### Inference input format

Inference takes `--json_path` = a JSON **list of objects**, one per LQ image:

```json
[
  {"image": "/path/to/lq_face_001.png", "caption": "a photo of a young woman, smiling"},
  {"image": "/path/to/lq_face_002.png", "caption": ""}
]
```

`image` is the LQ input path; `caption` is optional (use `""` if you have none). To synthesize LQ test images from HQ photos, use `process_image_degradation.py`. Restored images are written to `--output_dir`.

---

## 🏋️ Training

The full pipeline is **two stages**. See `artfr-run.sh` for the exact command sequence.

### Stage A — Hierarchical SFT of the backbone  `[env: art-fr]`

Two steps (toggle caption / reconstruction options in `blip3o/data/dataset.py`):

```bash
conda activate art-fr
bash scripts/sft_step1.sh        # step 1: SFT from the BLIP3o-NEXT-SFT-3B backbone
bash scripts/sft_step2.sh        # step 2: VAE encoder + diffusion head
```

| Step | Script | Trainer | Starts from |
|---|---|---|---|
| 1 | `scripts/sft_step1.sh` | `blip3o/train/train_step1.py` | `BLIP3o-NEXT-SFT-3B` (backbone + TA-Tok + SANA) |
| 2 | `scripts/sft_step2.sh` | `blip3o/train/train_step2.py` | the step-1 checkpoint |

DeepSpeed configs are under `scripts/zero1.json` / `scripts/zero2.json`.

### Stage B — Preference RL with DiffusionNFT  `[env: DiffusionNFT]`

```bash
conda activate DiffusionNFT
cd DiffusionNFT
export WANDB_PROJECT=DiffusionNFT_PrefRestore

# Main pipeline: multi-reward preference optimization
torchrun --nproc_per_node=8 --master_port=11234 \
    scripts/train_nft_prefRestore.py \
    --config config/pref_restore.py:pref_restore_multi_reward

# GT-aware variant (PickScore + HPSv2 + CLIPScore + LMD + ArcFace)
bash scripts/run_gt.sh pref_restore_gt_reward
```

| Script | Config | Description |
|---|---|---|
| `scripts/train_nft_prefRestore.py` | `config/pref_restore.py` | main multi-reward preference RL (paper method) |
| `scripts/train_nft_prefRestore_gt.py` | `config/pref_restore_gt.py` | GT-aware variant with landmark + ArcFace rewards |

RL checkpoints (including LoRA adapters) are written to `DiffusionNFT/logs/` (gitignored).

---

## 🖼 Inference

Two entry points, matching the two stages. Both read a JSON list of LQ images and write restored images to `--output_dir`.

### Base (SFT) model  `[env: art-fr]`

```bash
python inference_batch_noPrompt_fixLQ_vae.py \
    --model_path /path/to/SFT_checkpoint \
    --json_path  /path/to/captions_lq.json \
    --output_dir /path/to/results/base
```

### RL-finetuned (LoRA) model  `[env: DiffusionNFT]`

```bash
python inference_batch_noPrompt_fixLQ_vae_lora.py \
    --model_path /path/to/SFT_checkpoint \
    --json_path  /path/to/captions_lq.json \
    --output_dir /path/to/results/rl \
    --lora_path  /path/to/DiffusionNFT/logs/.../checkpoints/checkpoint-XXX \
    --use_lora
```

| Argument | Meaning |
|---|---|
| `--model_path` | the SFT backbone checkpoint (Stage A output) |
| `--json_path` | JSON list of low-quality input images |
| `--output_dir` | where restored images are saved |
| `--lora_path` | RL LoRA adapter (LoRA script only) |
| `--use_lora` | enable the LoRA adapter (LoRA script only) |

---

## 📜 License

This repository is released under the **Apache License 2.0** (see `LICENSE`).

The inlined third-party code retains its original license:

- `BasicSR/` — Apache-2.0 (XPixelGroup)
- `DiffusionNFT/` — original LICENSE preserved at `DiffusionNFT/LICENSE`
- `blip3o/` — see the upstream BLIP-3o repository

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yao2026prefrestore,
  title   = {Bridging Information Asymmetry: A Hierarchical Framework for Deterministic Blind Face Restoration},
  author  = {Yao, Zhengjian and Hu, Jiakui and Li, Kaiwen and He, Hangzhou and
             Zhang, Xinliang and Zeng, Shuang and Zhu, Lei and Lu, Yanye},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2026}
}
```

Preprint: [arXiv:2601.19506](https://arxiv.org/abs/2601.19506)
