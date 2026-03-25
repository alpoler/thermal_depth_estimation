import torch
import torch.nn as nn


class BetaLaplacianNLLLoss(nn.Module):
    def __init__(self, beta=0.5):
        """
        beta: Hyperparameter to control the gradient starvation.
              beta=0.0 is standard Laplacian NLL.
              beta=1.0 fully cancels the variance-based gradient scaling.
        """
        super().__init__()
        self.beta = beta

    def forward(self, pred_depth, log_scale, target_lidar, valid_mask):
        """
        pred_depth: (B, 1, H, W) - The predicted depth
        log_scale: (B, 1, H, W) - The predicted uncertainty (s = log(b))
        target_lidar: (B, 1, H, W) - The ground truth disparity
        valid_mask: (B, 1, H, W) - Boolean mask of valid points
        """
        error = torch.nn.functional.smooth_l1_loss(pred_depth, target_lidar, reduction='none', beta=1.0)

        # NLL: loss = exp(-s) * ρ(y - y_hat) + s
        nll_loss = torch.exp(-log_scale) * error + log_scale

        if self.beta > 0:
            scale_parameter_detached = torch.exp(log_scale).detach()
            beta_weight = scale_parameter_detached ** self.beta
            loss = nll_loss * beta_weight
        else:
            loss = nll_loss

        return loss[valid_mask].mean()
