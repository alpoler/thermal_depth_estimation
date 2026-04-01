import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward


class DTCWTSubbandLoss(nn.Module):
    """
    Dual-Tree Complex Wavelet Transform subband loss.
    Compares high-frequency subbands between pseudo depth and prediction
    at selected scales with exponential decay weighting (fine scales weighted more).
    """

    def __init__(self, J=3, alpha=2.0, levels=None):
        """
        Args:
            J: Number of decomposition scales.
            alpha: Decay factor for scale weighting. Weight at scale j = 1 / alpha^j.
                   j=0 is finest scale (weight=1), j=1 (weight=1/alpha), etc.
            levels: List of 1-indexed levels to compute loss on, e.g. [1,2,3] for all,
                    [2,3] to skip the finest scale. If None, uses all J levels.
        """
        super().__init__()
        self.J = J
        self.alpha = alpha
        if levels is not None:
            self.levels = [l - 1 for l in levels]  # convert to 0-indexed
        else:
            self.levels = list(range(J))
        self.dtcwt = DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b')

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted disparity
            target: (B, 1, H, W) pseudo disparity
        Returns:
            Scalar loss (weighted smooth L1 over selected scales and subbands)
        """
        with torch.cuda.amp.autocast(enabled=False):
            pred = pred.float()
            target = target.float()
            _, yh_pred = self.dtcwt(pred)
            _, yh_target = self.dtcwt(target)

        # yh[j] shape: (B, C, 6, H_j, W_j, 2)
        #   6 oriented subbands (generalize LH, HL, HH)
        #   last dim 2 = [real, imaginary]
        #   j=0 finest, j=J-1 coarsest

        loss = 0.0

        for j in self.levels:
            mag_pred = torch.sqrt(yh_pred[j][..., 0] ** 2 + yh_pred[j][..., 1] ** 2 + 1e-8)
            mag_target = torch.sqrt(yh_target[j][..., 0] ** 2 + yh_target[j][..., 1] ** 2 + 1e-8)
            weight = 1.0 / (self.alpha ** j)    
            loss = loss + weight * F.smooth_l1_loss(mag_pred, mag_target)

        return loss 
