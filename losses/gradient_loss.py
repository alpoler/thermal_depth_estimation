import torch
import torch.nn as nn
import torch.nn.functional as F

class StableGradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='mean', use_log=True):
        """
        Args:
            scales (int): Number of scales to apply the loss (downsampling).
            reduction (str): 'mean' or 'sum'.
            use_log (bool): If True, computes gradients in log-space (highly recommended for depth).
        """
        super().__init__()
        self.scales = scales
        self.reduction = reduction
        self.use_log = use_log

        # Sobel Kernel for X direction
        kernel_x = torch.tensor([[-1, 0, 1], 
                                 [-2, 0, 2], 
                                 [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Sobel Kernel for Y direction
        kernel_y = torch.tensor([[-1, -2, -1], 
                                 [ 0,  0,  0], 
                                 [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Register as buffers so they move to GPU automatically but aren't trainable
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def compute_gradient(self, img):
        """Computes gradients using Sobel filters."""
        # Pad to keep output size same as input
        img_pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
        
        grad_x = F.conv2d(img_pad, self.kernel_x, groups=1)
        grad_y = F.conv2d(img_pad, self.kernel_y, groups=1)
        
        return grad_x, grad_y

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred (Tensor): Predicted depth [B, 1, H, W]
            target (Tensor): Ground truth/Teacher depth [B, 1, H, W]
            mask (Tensor, optional): Validity mask [B, 1, H, W] (1=valid, 0=invalid)
        """
        total_loss = 0.0
        
        # 1. Transform to Log Space for Stability (Scale Invariance)
        if self.use_log:
            # Clamp to prevent log(negative) NaN when predictions drift negative
            pred = torch.log(pred.clamp(min=1e-6))
            target = torch.log(target.clamp(min=1e-6))

        for scale in range(self.scales):
            # 2. Compute Gradients
            pred_grad_x, pred_grad_y = self.compute_gradient(pred)
            target_grad_x, target_grad_y = self.compute_gradient(target)

            # 3. Calculate L1 Error between gradients
            diff_x = torch.abs(pred_grad_x - target_grad_x)
            diff_y = torch.abs(pred_grad_y - target_grad_y)
            
            loss_map = diff_x + diff_y

            # 4. Apply Masking (if provided)
            if mask is not None:
                loss_map = loss_map * mask
                valid_pixels = mask.sum()
                if valid_pixels > 0:
                    scale_loss = loss_map.sum() / valid_pixels
                else:
                    scale_loss = 0.0
            else:
                scale_loss = loss_map.mean() if self.reduction == 'mean' else loss_map.sum()

            total_loss += scale_loss

            # 5. Downsample for next scale (Multi-scale stability)
            if scale < self.scales - 1:
                pred = F.interpolate(pred, scale_factor=0.5, mode='bilinear', align_corners=False)
                target = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False)
                if mask is not None:
                    mask = F.interpolate(mask.float(), scale_factor=0.5, mode='nearest')

        # Normalize by number of scales so loss magnitude doesn't change with scales
        return total_loss / self.scales