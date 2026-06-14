import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

ARCFACE_MODULE_ROOT = "/data/phd/yaozhengjian/Code/RL/ART-FRv2/metrics/VQFR/metric_paper"
ARCFACE_WEIGHT_PATH = os.path.join(
    os.path.dirname(ARCFACE_MODULE_ROOT),
    "experiments/pretrained_models/metric_weights/resnet18_110.pth",
)


class ArcFaceScorer:
    """ArcFace identity similarity scorer using ResNet-18.

    Computes cosine similarity between face embeddings of predicted and GT images.
    Returns cosine similarity directly (higher = more similar = better).
    """

    def __init__(self, device="cuda"):
        if ARCFACE_MODULE_ROOT not in sys.path:
            sys.path.insert(0, ARCFACE_MODULE_ROOT)

        from arcface.models.resnet import resnet_face18

        self.device = device
        model = resnet_face18(use_se=False)

        # Checkpoint saved with DataParallel, strip "module." prefix
        state_dict = torch.load(ARCFACE_WEIGHT_PATH, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[len("module."):] if k.startswith("module.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

        self.model = model.to(device)
        self.model.eval()

    @staticmethod
    def _preprocess(images_tensor):
        """Convert [B, 3, H, W] RGB float [0,1] to [B, 1, 128, 128] grayscale [-1,1].

        Uses luminance formula: Y = 0.2989*R + 0.5870*G + 0.1140*B
        """
        gray = (
            0.2989 * images_tensor[:, 0:1, :, :]
            + 0.5870 * images_tensor[:, 1:2, :, :]
            + 0.1140 * images_tensor[:, 2:3, :, :]
        )
        gray = F.interpolate(gray, (128, 128), mode="bilinear", align_corners=False)
        gray = gray * 2.0 - 1.0  # [0,1] -> [-1,1]
        return gray

    @torch.no_grad()
    def __call__(self, pred_images, gt_images_tensor):
        """
        Args:
            pred_images: torch.Tensor [B, 3, H, W] float [0,1]
            gt_images_tensor: torch.Tensor [B, 3, H, W] float [0,1]
        Returns:
            list of float: cosine similarities (higher = better)
        """
        pred_input = self._preprocess(pred_images.float().to(self.device))
        gt_input = self._preprocess(gt_images_tensor.float().to(self.device))

        pred_emb = self.model(pred_input)   # [B, 512]
        gt_emb = self.model(gt_input)       # [B, 512]

        cos_sim = F.cosine_similarity(pred_emb, gt_emb, dim=1)
        return cos_sim.cpu().tolist()
