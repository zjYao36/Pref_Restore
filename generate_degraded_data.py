"""
Generate three levels of degraded images (mild, moderate, severe) from clean images.
Uses the same degradation pipeline as LazySupervisedRestoreDataset.
"""
import os
import sys
import glob
import numpy as np
from PIL import Image
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blip3o.data.image_degradation import degrade_image

# --- Paths ---
SRC_DIR = "/data/phd/yaozhengjian/zjYao_Datasets/Pref-Restore/FFHQ_val/CelebA_HQ/validation_image"
DST_ROOT = "/data/phd/yaozhengjian/zjYao_Datasets/Pref-Restore/FFHQ_val/CelebA_HQ/validation_image_rebuttal"

# --- Original param ranges ---
# blur_sigma: [1, 15]        range=14
# downsample_range: [1, 15]  range=14
# noise_range: [0, 20]       range=20
# jpeg_range: [30, 90]       range=60 (reversed: higher value = less degradation)

# --- Three-way split ---
DEGRADATION_LEVELS = {
    "mild": {
        'gt_size': 512,
        'in_size': 512,
        'use_motion_kernel': False,
        'blur_kernel_size': 41,
        'blur_sigma': [1, 5.67],
        'downsample_range': [1, 5.67],
        'noise_range': [0, 6.67],
        'jpeg_range': [70, 90],
    },
    "moderate": {
        'gt_size': 512,
        'in_size': 512,
        'use_motion_kernel': False,
        'blur_kernel_size': 41,
        'blur_sigma': [5.67, 10.33],
        'downsample_range': [5.67, 10.33],
        'noise_range': [6.67, 13.33],
        'jpeg_range': [50, 70],
    },
    "severe": {
        'gt_size': 512,
        'in_size': 512,
        'use_motion_kernel': False,
        'blur_kernel_size': 41,
        'blur_sigma': [10.33, 15],
        'downsample_range': [10.33, 15],
        'noise_range': [13.33, 20],
        'jpeg_range': [30, 50],
    },
}


def process_single_image(img_path, level_name, params, dst_dir):
    """Process a single image with given degradation params."""
    try:
        img = Image.open(img_path).convert("RGB")
        degraded = degrade_image(img, **params)
        out_path = os.path.join(dst_dir, os.path.basename(img_path))
        degraded.save(out_path)
        return True
    except Exception as e:
        print(f"[{level_name}] Error processing {img_path}: {e}")
        return False


def process_level(level_name, params, img_paths):
    """Process all images for one degradation level."""
    dst_dir = os.path.join(DST_ROOT, level_name)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"\n=== Processing level: {level_name} ===")
    print(f"  Output dir: {dst_dir}")
    print(f"  Params: {params}")
    print(f"  Total images: {len(img_paths)}")

    fn = partial(process_single_image, level_name=level_name, params=params, dst_dir=dst_dir)

    success = 0
    fail = 0
    with Pool(processes=16) as pool:
        results = list(tqdm(
            pool.imap(fn, img_paths),
            total=len(img_paths),
            desc=level_name,
        ))
    success = sum(results)
    fail = len(results) - success
    print(f"  Done: {success} success, {fail} failed")


def main():
    # Collect all source images
    img_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.*")))
    print(f"Found {len(img_paths)} source images in {SRC_DIR}")

    os.makedirs(DST_ROOT, exist_ok=True)

    for level_name, params in DEGRADATION_LEVELS.items():
        process_level(level_name, params, img_paths)

    print("\n=== All done! ===")


if __name__ == "__main__":
    main()
