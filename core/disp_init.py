"""Disparity initialization via softmax or unbalanced optimal transport.

Unbalanced Sinkhorn with Dykstra-like recentering (Séjourné et al. 2022).
Constants derived from tau:
    kappa = (1 - tau) / 2
    xi    = (1 - tau) / (1 + tau)
    rho   = eps * tau / (1 - tau)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def disparity_regression(prob: Tensor, max_disp: int) -> Tensor:
    disp = torch.arange(max_disp, device=prob.device, dtype=prob.dtype)
    return (prob * disp.reshape(1, -1, 1, 1)).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Softmax baseline
# ---------------------------------------------------------------------------

class SoftmaxDisparityInit(nn.Module):
    def __init__(self, max_disp: int):
        super().__init__()
        self.max_disp = max_disp

    def forward(self, scores: Tensor) -> Tensor:
        return disparity_regression(F.softmax(scores, dim=1), self.max_disp)


# ---------------------------------------------------------------------------
# Unbalanced OT solver
# ---------------------------------------------------------------------------

class UnbalancedOT(nn.Module):
    r"""Log-domain unbalanced Sinkhorn with Dykstra recentering.

    Each iteration does a damped (tau) Sinkhorn half-step then applies
    kappa / xi corrections that recenter the potentials.
    """

    def __init__(self, W: int, C: int, num_iter: int = 10,
                 tau: float = 0.95, epsilon: float = 1.0):
        super().__init__()
        self.num_iter = num_iter
        self.tau = tau
        self.eps = epsilon

        # Derived constants
        self.kappa = (1.0 - tau) / 2.0
        self.xi = (1.0 - tau) / (1.0 + tau)

        # Uniform log-marginals: each row gets 1/W, each col gets 1/C
        # Row marginals: 1/W each, col marginals: 1/(W+C) each
        self.register_buffer("log_a", torch.full((W,), -math.log(W)))
        self.register_buffer("log_b", torch.full((C,), -math.log(C)))

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    @property
    def rho(self) -> float:
        """KL penalty  rho = eps * tau / (1 - tau)."""
        return self.eps * self.tau / (1.0 - self.tau)

    def _softmax(self, C: Tensor, f: Tensor, g: Tensor,
                 axis: int) -> Tensor:
        """Stabilised log-sum-exp of the Gibbs kernel.

        S = (C - f[..., None] - g[..., None, :]) / eps
        return eps * logsumexp(-S, dim=axis)

        axis = -1  ->  reduce over columns  ->  shape like f  (B,H,W)
        axis = -2  ->  reduce over rows     ->  shape like g  (B,H,C)
        """
        S = (C - f.unsqueeze(-1) - g.unsqueeze(-2)) / self.eps
        return self.eps * torch.logsumexp(-S, dim=axis)

    def _lse_kernel(self, C: Tensor, f: Tensor, g: Tensor,
                    axis: int) -> Tensor:
        """LSE kernel application minus the current potential."""
        w_res = self._softmax(C, f, g, axis)
        w_sgn = f if axis == -1 else g
        return w_res - torch.where(torch.isfinite(w_sgn), w_sgn,
                                   torch.zeros_like(w_sgn))

    def _sinkhorn_update(self, C: Tensor, f: Tensor, g: Tensor,
                         axis: int, log_marginal: Tensor) -> Tensor:
        """One balanced Sinkhorn half-step (before tau damping)."""
        app_lse = self._lse_kernel(C, f, g, axis)
        return self.eps * log_marginal - torch.where(
            torch.isfinite(app_lse), app_lse, torch.zeros_like(app_lse))

    def _smin(self, h: Tensor, log_marginal: Tensor) -> Tensor:
        """Soft-min:  smin(h, c) = -rho * logsumexp(-h/rho + log c, dim=-1)."""
        rho = self.rho
        return -rho * torch.logsumexp(-h / rho + log_marginal, dim=-1)

    # ------------------------------------------------------------------
    # Single iteration (Sinkhorn + Dykstra recentering)
    # ------------------------------------------------------------------

    def _one_iter(self, C: Tensor, f: Tensor, g: Tensor):
        """One full iteration: update g then f, each with recentering.

        Shapes
        ------
        C : (B, H, W, C_cols)   cost matrix
        f : (B, H, W)           row potentials
        g : (B, H, C_cols)      column potentials
        """
        old_f = f

        # ---- g update (reduce over rows, axis=-2) --------------------
        new_g = self.tau * self._sinkhorn_update(
            C, old_f, g, axis=-2, log_marginal=self.log_b)

        # Dykstra recentering for g
        new_g = new_g - self.kappa * self._smin(old_f, self.log_a).unsqueeze(-1)
        new_g = new_g + self.xi * self._smin(new_g, self.log_b).unsqueeze(-1)

        # ---- f update (reduce over cols, axis=-1, using new_g) -------
        new_f = self.tau * self._sinkhorn_update(
            C, f, new_g, axis=-1, log_marginal=self.log_a)

        # Dykstra recentering for f
        new_f = new_f - self.kappa * self._smin(new_g, self.log_b).unsqueeze(-1)
        new_f = new_f + self.xi * self._smin(new_f, self.log_a).unsqueeze(-1)

        return new_f, new_g

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _init_potentials(self, C: Tensor):
        B, H, W, C_cols = C.shape
        f = -self.eps * torch.logsumexp(-C / self.eps + self.log_b, dim=-1)
        g = C.new_zeros(B, H, C_cols)
        return f, g

    def forward(self, C: Tensor) -> Tensor:
        """Run unbalanced Sinkhorn and return transport plan P.

        Args:
            C: (B, H, W, C_cols) cost matrix (lower = better).
        Returns:
            P: (B, H, W, C_cols) transport plan.
        """
        f, g = self._init_potentials(C)

        for _ in range(self.num_iter):
            f, g = self._one_iter(C, f, g)

        S = (C - f.unsqueeze(-1) - g.unsqueeze(-2)) / self.eps
        P = torch.exp(-S)
        return P


# ---------------------------------------------------------------------------
# OT disparity initialisation
# ---------------------------------------------------------------------------

class OTDisparityInit(nn.Module):
    """Unbalanced OT disparity initialization.

    Cost matrix  W x C  where  C = D + W :
        Row j  = left pixel (0..W-1).
        Col c  = right pixel coordinate  c - D.
        Valid column for (j, d):  c = j - d + D.
    """

    def __init__(self, max_disp: int, image_width: int,
                 ot_iter: int = 10, epsilon: float = 1.0,
                 tau: float = 0.95):
        super().__init__()
        self.max_disp = max_disp
        self.epsilon = epsilon

        W = image_width // 4
        self.W = W
        self.C = max_disp + W

        self.ot = UnbalancedOT(W, self.C, num_iter=ot_iter,
                               tau=tau, epsilon=epsilon)

        # Scatter indices: col[d][j] = j - d + max_disp
        j = torch.arange(W)
        col_indices = torch.stack([j - d + max_disp for d in range(max_disp)])
        self.register_buffer("_col_idx", col_indices)  # (D, W)

    # ----- score -> cost matrix ----------------------------------------

    def _scatter_to_cost(self, scores: Tensor) -> Tensor:
        """(B, D, H, W) similarity scores -> (B, H, W, C) cost = -scores."""
        B, D, H, W = scores.shape
        cost = scores.new_full((B, H, W, self.C), 1e4)
        j_idx = torch.arange(W, device=scores.device)
        for d in range(D):
            col = self._col_idx[d].to(scores.device)
            cost[:, :, j_idx, col] = -scores[:, d, :, :]
        return cost

    # ----- transport plan -> disparity ---------------------------------

    def _plan_to_disparity(self, P: Tensor, D: int) -> Tensor:
        device, dtype = P.device, P.dtype
        right = torch.arange(self.C, device=device, dtype=dtype) - D
        left = torch.arange(self.W, device=device, dtype=dtype)
        disp_map = left.view(1, 1, -1, 1) - right.view(1, 1, 1, -1)

        mass = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        disp = (P * disp_map).sum(dim=-1, keepdim=True) / mass

        disp = disp.squeeze(-1).unsqueeze(1)
        self.conf = P.max(dim=-1).values.unsqueeze(1)
        self.occ = mass.squeeze(-1).unsqueeze(1)
        return disp

    # ----- forward ----------------------------------------------------

    def forward(self, scores: Tensor) -> Tensor:
        B, D, H, W = scores.shape
        C = self._scatter_to_cost(scores)
        P = self.ot(C)
        return self._plan_to_disparity(P, D)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_disp_init(args) -> nn.Module:
    max_disp = args.max_disp // 4
    method = args.get("disp_init", "softmax") if hasattr(args, "get") else "softmax"

    if method == "softmax":
        return SoftmaxDisparityInit(max_disp)
    if method == "ot":
        return OTDisparityInit(
            max_disp=max_disp,
            image_width=args.image_width,
            ot_iter=getattr(args, "ot_iter", 20),
            epsilon=getattr(args, "ot_epsilon", 1.0),
            tau=getattr(args, "ot_tau", 0.95),
        )
    raise ValueError(f"Unknown disp_init method: {method}")
