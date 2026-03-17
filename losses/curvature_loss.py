import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvatureLoss(nn.Module):
    def __init__(self, scales=4, use_log=True, eps=1e-6, beta=0.01):
        """
        Computes curvature = sqrt(dxx^2 + dyy^2 + 2*dxy^2) for both prediction
        and pseudo-GT, then compares them with Smooth L1 loss.

        Args:
            scales (int): Number of scales (progressive downsampling).
            use_log (bool): If True, operates in log-space for scale invariance.
            eps (float): Clamping epsilon for numerical stability.
            beta (float): Smooth L1 transition point. Smaller = stays in L1 regime
                          for small curvature differences, preventing over-squashing.
        """
        super().__init__()
        self.scales = scales
        self.use_log = use_log
        self.eps = eps
        self.beta = beta

        # Second-order finite difference kernels (3x3)
        # d^2f/dx^2
        kernel_dxx = torch.tensor([[0, 0, 0],
                                   [1, -2, 1],
                                   [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # d^2f/dy^2
        kernel_dyy = torch.tensor([[0, 1, 0],
                                   [0, -2, 0],
                                   [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # d^2f/dxdy
        kernel_dxy = torch.tensor([[ 1, 0, -1],
                                   [ 0, 0,  0],
                                   [-1, 0,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) * 0.25

        self.register_buffer('kernel_dxx', kernel_dxx)
        self.register_buffer('kernel_dyy', kernel_dyy)
        self.register_buffer('kernel_dxy', kernel_dxy)

    def _second_order_derivatives(self, x):
        """Compute dxx, dyy, dxy using finite difference convolutions."""
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        dxx = F.conv2d(x_pad, self.kernel_dxx)
        dyy = F.conv2d(x_pad, self.kernel_dyy)
        dxy = F.conv2d(x_pad, self.kernel_dxy)
        return dxx, dyy, dxy

    def _curvature(self, x):
        """Curvature = sqrt(dxx^2 + dyy^2 + 2*dxy^2), with eps inside sqrt for stability."""
        dxx, dyy, dxy = self._second_order_derivatives(x)
        curv = torch.sqrt(dxx.square() + dyy.square() + 2.0 * dxy.square() + self.eps)
        return curv

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred (Tensor): Predicted depth [B, 1, H, W]
            target (Tensor): Pseudo-GT depth [B, 1, H, W]
            mask (Tensor, optional): Validity mask [B, 1, H, W] (1=valid, 0=invalid)
        """
        if self.use_log:
            pred = torch.log(pred.clamp(min=self.eps))
            target = torch.log(target.clamp(min=self.eps))

        total_loss = 0.0

        for scale in range(self.scales):
            curv_pred = self._curvature(pred)
            curv_target = self._curvature(target)

            loss_map = F.smooth_l1_loss(curv_pred, curv_target, reduction='none', beta=self.beta)

            if mask is not None:
                loss_map = loss_map * mask
                valid = mask.sum()
                scale_loss = loss_map.sum() / valid.clamp(min=1.0)
            else:
                scale_loss = loss_map.mean()

            total_loss += scale_loss

            if scale < self.scales - 1:
                pred = F.interpolate(pred, scale_factor=0.5, mode='bilinear', align_corners=False,antialias=True)
                target = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False,antialias=True)
                if mask is not None:
                    mask = F.interpolate(mask.float(), scale_factor=0.5, mode='nearest')

        return total_loss / self.scales
