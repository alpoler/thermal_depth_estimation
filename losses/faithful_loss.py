import torch
import torch.nn as nn


class FaithfulLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_depth, variance, target_lidar, valid_mask):
        """
        pred_depth: (B, 1, H, W) - The predicted depth
        variance: (B, 1, H, W) - The predicted variance (positive, after softplus)
        target_lidar: (B, 1, H, W) - The ground truth disparity
        valid_mask: (B, 1, H, W) - Boolean mask of valid points
        """
        pred_valid = pred_depth[valid_mask]
        target_valid = target_lidar[valid_mask]
        var_valid = variance[valid_mask]

        l1_loss = torch.nn.functional.l1_loss(pred_valid, target_valid, reduction='none')

        # NLL: loss = (1/var) * |y - y_hat| + log(var), detach pred to avoid gradient starvation
        var_loss = (torch.nn.functional.l1_loss(pred_valid.detach(), target_valid, reduction='none')) / var_valid + torch.log(var_valid)

        return l1_loss.mean(), var_loss.mean()
