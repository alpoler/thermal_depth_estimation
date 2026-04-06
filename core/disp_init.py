"""
Disparity initialization strategies.

Provides a common interface for converting classifier scores (B, D, H, W)
into an initial disparity map (B, 1, H, W).

Usage in model __init__:
    self.disp_init = build_disp_init(args)

Usage in model forward:
    classifier_scores = self.classifier(comb_volume).squeeze(1)  # (B, D, H, W)
    if init_disp is None:
        init_disp = self.disp_init(classifier_scores)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.submodule import disparity_regression


class SoftmaxDisparityInit(nn.Module):
    """Original softmax + weighted sum disparity initialization."""

    def __init__(self, max_disp: int):
        super().__init__()
        self.max_disp = max_disp

    def forward(self, scores: Tensor) -> Tensor:
        """
        Args:
            scores: (B, D, H, W) raw classifier logits
        Returns:
            init_disp: (B, 1, H, W)
        """
        prob = F.softmax(scores, dim=1)
        return disparity_regression(prob, self.max_disp)


class OTDisparityInit(nn.Module):
    """
    Optimal transport disparity initialization.

    Cost matrix layout: W x (D + W + 1)
      - W rows: left pixel positions (j = 0..W-1)
      - D + W columns: right pixel coordinates from -D to W-1
        Column c represents right pixel coordinate c - D
        For left pixel j at disparity d: column = j - d + D
      - +1 column: dustbin for occlusions

    Every left pixel has exactly D valid entries, eliminating
    left-edge artifacts from the old W x W formulation.
    """

    def __init__(self, max_disp: int, image_width: int, ot_iter: int = 10,
                 epsilon: float = 1.0):
        super().__init__()
        self.max_disp = max_disp
        self.ot_iter = ot_iter
        self.epsilon = epsilon

        W = image_width // 4
        self.W = W
        C = max_disp + W  # real columns (excluding dustbin)
        self.C = C

        # Build marginals
        log_row, log_col = self._build_marginals(W, max_disp, C)
        self.register_buffer('log_row_marginal', log_row)
        self.register_buffer('log_col_marginal', log_col)



    @staticmethod
    def _build_marginals(W: int, D: int, C: int):
        dustbin_mass = 0.05

        # dustbin_mass fraction: x/(W+x) = dustbin_mass => x = dustbin_mass*W/(1-dustbin_mass)
        row_real = torch.ones(W)
        row_dust = dustbin_mass * W / (1 - dustbin_mass)
        row_marginal = torch.cat([row_real, torch.tensor([row_dust])])
        row_marginal = row_marginal / row_marginal.sum()

        col_real = torch.ones(C)
        col_dust = dustbin_mass * C / (1 - dustbin_mass)
        col_marginal = torch.cat([col_real, torch.tensor([col_dust])])
        col_marginal = col_marginal / col_marginal.sum()

        return row_marginal.log().reshape(1, 1, -1), \
               col_marginal.log().reshape(1, 1, -1)

    @staticmethod
    def _logsumexp(x: Tensor, dim: int) -> Tensor:
        max_val = x.max(dim=dim, keepdim=True).values.clamp(min=-1e30)
        return max_val.squeeze(dim) + (x - max_val).exp().sum(dim=dim).log()

    def _sinkhorn(self, attn: Tensor, log_row: Tensor, log_col: Tensor) -> Tensor:
        """Log-domain Sinkhorn. attn: (B, H, W+1, C+1) with dustbins."""
        lse = self._logsumexp
        u = -lse(attn, dim=3)
        v = log_col - lse(attn + u.unsqueeze(3), dim=2)
        u = log_row - lse(attn + v.unsqueeze(2), dim=3)
        for _ in range(self.ot_iter - 1):
            v = log_col - lse(attn + u.unsqueeze(3), dim=2)
            u = log_row - lse(attn + v.unsqueeze(2), dim=3)
        v = log_col - lse(attn + u.unsqueeze(3), dim=2)
        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _scatter_to_cost_matrix(self, scores: Tensor) -> Tensor:
        """
        Disparity-indexed (B, D, H, W) -> cost matrix (B, H, W, C).
        Column c = j - d + D. Every row j has D valid entries.
        """
        B, D, H, W = scores.shape
        C = self.C
        corr = scores.new_full((B, H, W, C), -1e4)
        for d in range(D):
            j = torch.arange(W)
            col = j - d + D  # column for each left pixel at disparity d
            corr[:, :, j, col] = scores[:, d, :, :]
        return corr

    def forward(self, scores: Tensor) -> Tensor:
        """
        Args:
            scores: (B, D, H, W) raw classifier logits
        Returns:
            init_disp: (B, 1, H, W)
        """
        B, D, H, W = scores.shape
        device, dtype = scores.device, scores.dtype

        corr = self._scatter_to_cost_matrix(scores)  # (B, H, W, C)


        softmax_prob = F.softmax(scores, dim=1)
        softmax_disp = disparity_regression(softmax_prob, self.max_disp)

        init_disp = self._solve_ot(corr, self.epsilon, B, D, H, W, device, dtype)
        return init_disp

    def _solve_ot(self, corr: Tensor, epsilon: float,
                  B: int, D: int, H: int, W: int,
                  device: torch.device, dtype: torch.dtype) -> Tensor:
        """Run Sinkhorn OT on W x C cost matrix and extract disparity."""
        attn = corr / epsilon

        # Pad for dustbin: rows +1, cols +1
        attn_padded = F.pad(attn, (0, 1, 0, 1), value=0)

        log_T = self._sinkhorn(attn_padded, self.log_row_marginal, self.log_col_marginal)

        # Remove dustbin dims
        prob = log_T[:, :, :-1, :-1].exp()  # (B, H, W, C)

        # Right pixel coordinates: -D, -(D-1), ..., W-1
        right_coords = torch.arange(self.C, device=device, dtype=dtype) - D  # (C,)
        j_coords = torch.arange(W, device=device, dtype=dtype)               # (W,)

        # Disparity = j - right_coord
        disp_map = j_coords.reshape(1, 1, W, 1) - right_coords.reshape(1, 1, 1, self.C)

        # Weighted average disparity
        prob_sum = prob.sum(dim=3, keepdim=True).clamp(min=1e-8)
        init_disp = (prob * disp_map).sum(dim=3, keepdim=True) / prob_sum  # (B, H, W, 1)
        init_disp = init_disp.squeeze(3).unsqueeze(1)  # (B, 1, H, W)

        # Confidence and occlusion
        self.conf = prob.max(dim=3).values.unsqueeze(1)  # (B, 1, H, W)
        self.occ = prob_sum.squeeze(3).unsqueeze(1)      # (B, 1, H, W)

        return init_disp


def build_disp_init(args) -> nn.Module:
    """Factory: build disparity init module from config."""
    max_disp = args.max_disp // 4
    method = args.get('disp_init', 'softmax')

    if method == 'softmax':
        return SoftmaxDisparityInit(max_disp)
    elif method == 'ot':
        ot_iter = args.get('ot_iter', 10)
        epsilon = args.get('ot_epsilon', 1.0)
        image_width = args.image_width
        return OTDisparityInit(max_disp, image_width, ot_iter=ot_iter,
                               epsilon=epsilon)
    else:
        raise ValueError(f"Unknown disp_init method: {method}")
