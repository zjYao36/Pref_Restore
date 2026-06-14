import sys
import os
import numpy as np
import torch
import cv2

VQFR_ROOT = "/data/phd/yaozhengjian/Code/RL/ART-FRv2/metrics/VQFR"
LMD_WEIGHT_PATH = os.path.join(
    VQFR_ROOT, "experiments/pretrained_models/metric_weights/alignment_WFLW_4HG.pth"
)


class LMDScorer:
    """Landmark Distance scorer using FAN (Face Alignment Network).

    Computes mean L2 distance between predicted and GT facial landmarks (98 points).
    Returns normalized score in (0, 1] via 1/(1 + distance/scale), higher = better.
    """

    def __init__(self, device="cuda"):
        if VQFR_ROOT not in sys.path:
            sys.path.insert(0, VQFR_ROOT)

        from vqfr.utils.registry import ARCH_REGISTRY
        import vqfr.archs.awing_arch  # noqa: F401 - triggers FAN registration

        self.device = device
        self.model = ARCH_REGISTRY.get("FAN")()
        state_dict = torch.load(LMD_WEIGHT_PATH, map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(device)

    @staticmethod
    def _tensor_to_bgr_uint8(tensor_img):
        """Convert [C, H, W] float [0,1] tensor to BGR uint8 numpy (H, W, 3)."""
        img_np = (tensor_img.cpu().float().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _landmark_distance(gt_landmark, pred_landmark):
        return np.sqrt(((gt_landmark - pred_landmark) ** 2).sum(1)).mean()

    @torch.no_grad()
    def __call__(self, pred_images, gt_images_bgr):
        """
        Args:
            pred_images: torch.Tensor [B, 3, H, W] float [0,1]
            gt_images_bgr: list of BGR uint8 numpy arrays (H, W, 3)
        Returns:
            list of float: normalized scores in (0, 1], higher = better
        """
        scores = []
        for i in range(len(pred_images)):
            try:
                pred_bgr = self._tensor_to_bgr_uint8(pred_images[i])
                gt_bgr = gt_images_bgr[i]

                pred_landmark = self.model.get_landmarks(pred_bgr, device=self.device)
                gt_landmark = self.model.get_landmarks(gt_bgr, device=self.device)

                distance = self._landmark_distance(gt_landmark, pred_landmark)
                # Normalize to (0, 1]: distance=0 → 1.0, distance=20 → 0.5, distance=80 → 0.2
                scores.append(1.0 / (1.0 + float(distance) / 20.0))
            except Exception:
                # Face detection failure: assign near-zero score
                scores.append(0.01)
        return scores
