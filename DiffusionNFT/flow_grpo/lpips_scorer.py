import torch
import lpips


class LPIPSScorer:
    """LPIPS (Learned Perceptual Image Patch Similarity) scorer.

    Computes perceptual distance between predicted and GT images using VGG features.
    Returns negative LPIPS so that higher = better (for RL reward).
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.model = lpips.LPIPS(net="vgg", spatial=False).eval().to(device)

    @torch.no_grad()
    def __call__(self, pred_images, gt_images_tensor):
        """
        Args:
            pred_images: torch.Tensor [B, 3, H, W] float [0,1]
            gt_images_tensor: torch.Tensor [B, 3, H, W] float [0,1]
        Returns:
            list of float: negative LPIPS distances (higher = better)
        """
        # lpips expects [-1, 1] range
        pred_input = pred_images.float().to(self.device) * 2.0 - 1.0
        gt_input = gt_images_tensor.float().to(self.device) * 2.0 - 1.0

        distances = self.model(pred_input, gt_input)  # [B, 1, 1, 1]
        distances = distances.squeeze().cpu().tolist()

        # Handle single-image batch (squeeze reduces to scalar)
        if not isinstance(distances, list):
            distances = [distances]

        # Return negative so higher = better
        return [-d for d in distances]
