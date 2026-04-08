"""Disparity initialization via softmax, unbalanced OT, or dustbin OT.

Unbalanced Sinkhorn with Dykstra-like recentering (Séjourné et al. 2022).
Constants derived from tau:
    kappa = (1 - tau) / 2
    xi    = (1 - tau) / (1 + tau)
    rho   = eps * tau / (1 - tau)

Dustbin OT: balanced Sinkhorn (tau=1, no recentering) with an extra
dustbin row/column that absorbs unmatched mass.
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
# Base OT solver
# ---------------------------------------------------------------------------

class BaseOT(nn.Module):
    """Base class for log-domain Sinkhorn solvers.

    Provides shared primitives and the forward loop.

    Subclasses must implement:
        _build_marginals(W, D, col_counts) -> (log_a, log_b)
        _one_iter(C, f, g) -> (f, g)
    """

    def __init__(self, W: int, C: int, num_iter: int = 10,
                 epsilon: float = 1.0, occ_frac: float = 0.0,
                 adaptive_eps: bool = False, eps_base: float = 0.05):
        super().__init__()
        self.num_iter = num_iter
        self.eps = epsilon
        self.adaptive_eps = adaptive_eps
        self.eps_base = eps_base
        self.occ_frac = occ_frac

        D = C - W + 1
        col_counts = self._compute_col_counts(W, D)
        log_a, log_b = self._build_marginals(W, D, col_counts, occ_frac)
        self.register_buffer("log_a", log_a)
        self.register_buffer("log_b", log_b)

    @staticmethod
    def _compute_col_counts(W: int, D: int) -> Tensor:
        """Count valid rows per column.

        Scatter uses c = j - d + (D-1), valid disparities d in [0, D-1].
        Column c maps to right-pixel coordinate c - (D-1).
        Valid rows for column c: max(0, c - (D-1)) <= j <= min(W-1, c).
        Last reachable column: (W-1) - 0 + (D-1) = W+D-2 = C-1.
        """
        C = D + W - 1
        col_counts = torch.zeros(C)
        for c in range(C):
            lo = max(0, c - (D - 1))
            hi = min(W - 1, c)
            col_counts[c] = max(0, hi - lo + 1)
        return col_counts.clamp(min=0.5)

    @staticmethod
    def _build_marginals(W: int, D: int, col_counts: Tensor, occ_frac: float):
        raise NotImplementedError

    def _one_iter(self, C: Tensor, f: Tensor, g: Tensor):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Common primitives
    # ------------------------------------------------------------------

    def _softmax(self, C: Tensor, f: Tensor, g: Tensor,
                 axis: int) -> Tensor:
        """Stabilised log-sum-exp of the Gibbs kernel.

        axis = -1  ->  reduce over columns  ->  shape like f
        axis = -2  ->  reduce over rows     ->  shape like g
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
        """One balanced Sinkhorn half-step."""
        app_lse = self._lse_kernel(C, f, g, axis)
        return self.eps * log_marginal - torch.where(
            torch.isfinite(app_lse), app_lse, torch.zeros_like(app_lse))

    def _init_potentials(self, C: Tensor):
        B, H, W, C_cols = C.shape
        f = -self.eps * torch.logsumexp(-C / self.eps + self.log_b, dim=-1)
        g = C.new_zeros(B, H, C_cols)
        return f, g

    def forward(self, C: Tensor) -> Tensor:
        """Run Sinkhorn and return transport plan P.

        Args:
            C: (B, H, rows, cols) cost matrix (lower = better).
        Returns:
            P: (B, H, rows, cols) transport plan.
        """
        if self.adaptive_eps:
            with torch.no_grad():
                scale = C[C < 1e3].std() + 1e-6
            self.eps = self.eps_base * scale

        f, g = self._init_potentials(C)

        for _ in range(self.num_iter):
            f, g = self._one_iter(C, f, g)

        S = (C - f.unsqueeze(-1) - g.unsqueeze(-2)) / self.eps
        P = torch.exp(-S)
        return P


# ---------------------------------------------------------------------------
# Unbalanced OT solver
# ---------------------------------------------------------------------------

class UnbalancedOT(BaseOT):
    r"""Log-domain unbalanced Sinkhorn with Dykstra recentering."""

    def __init__(self, W: int, C: int, num_iter: int = 10,
                 tau: float = 0.95, epsilon: float = 1.0,
                 adaptive_eps: bool = False, eps_base: float = 0.05):
        self.tau = tau
        self.kappa = (1.0 - tau) / 2.0
        self.xi = (1.0 - tau) / (1.0 + tau)
        super().__init__(W, C, num_iter=num_iter, epsilon=epsilon,
                         adaptive_eps=adaptive_eps, eps_base=eps_base)

    @staticmethod
    def _build_marginals(W: int, D: int, col_counts: Tensor, occ_frac: float):
        log_a = torch.full((W,), -math.log(W))
        col_marginal = col_counts / col_counts.sum()
        log_b = col_marginal.log()
        return log_a, log_b

    @property
    def rho(self) -> float:
        """KL penalty  rho = eps * tau / (1 - tau)."""
        return self.eps * self.tau / (1.0 - self.tau)

    def _smin(self, h: Tensor, log_marginal: Tensor) -> Tensor:
        rho = self.rho
        return -rho * torch.logsumexp(-h / rho + log_marginal, dim=-1)

    def _one_iter(self, C: Tensor, f: Tensor, g: Tensor):
        old_f = f

        # g update (reduce over rows) with tau damping + recentering
        new_g = self.tau * self._sinkhorn_update(
            C, old_f, g, axis=-2, log_marginal=self.log_b)
        new_g = new_g - self.kappa * self._smin(old_f, self.log_a).unsqueeze(-1)
        new_g = new_g + self.xi * self._smin(new_g, self.log_b).unsqueeze(-1)

        # f update (reduce over cols) with tau damping + recentering
        new_f = self.tau * self._sinkhorn_update(
            C, f, new_g, axis=-1, log_marginal=self.log_a)
        new_f = new_f - self.kappa * self._smin(new_g, self.log_b).unsqueeze(-1)
        new_f = new_f + self.xi * self._smin(new_f, self.log_a).unsqueeze(-1)

        return new_f, new_g


# ---------------------------------------------------------------------------
# Dustbin OT solver (balanced Sinkhorn, tau=1, no recentering)
# ---------------------------------------------------------------------------

class DustbinOT(BaseOT):
    """Balanced Sinkhorn (tau=1, no recentering) with dustbin column only.

    No dustbin row — out-of-frame occlusion is already modeled by the
    left-padding columns.  Only a single dustbin column is appended for
    within-frame occlusion.

    occ_frac controls the fixed fraction of row mass allocated to dustbin.
    """

    def __init__(self, W: int, C: int, num_iter: int = 10,
                 epsilon: float = 1.0, occ_frac: float = 0.1,
                 adaptive_eps: bool = False, eps_base: float = 0.05):
        super().__init__(W, C, num_iter=num_iter, epsilon=epsilon,
                         occ_frac=occ_frac, adaptive_eps=adaptive_eps,
                         eps_base=eps_base)

    @staticmethod
    def _build_marginals(W: int, D: int, col_counts: Tensor, occ_frac: float):
        C = D + W - 1
        # Row marginal: uniform over W (no dustbin row)
        log_a = torch.full((W,), -math.log(W))

        # Col marginal: col_counts-weighted real cols get (1 - occ_frac),
        # dustbin col gets occ_frac
        col_real = col_counts / col_counts.sum() * (1.0 - occ_frac)
        col_marginal = torch.cat([col_real, torch.tensor([occ_frac])])
        log_b = col_marginal.log()

        return log_a, log_b

    def _one_iter(self, C: Tensor, f: Tensor, g: Tensor):
        # Balanced: tau=1, no recentering
        new_g = self._sinkhorn_update(C, f, g, axis=-2, log_marginal=self.log_b)
        new_f = self._sinkhorn_update(C, f, new_g, axis=-1, log_marginal=self.log_a)
        return new_f, new_g


# ---------------------------------------------------------------------------
# Base OT disparity initialisation
# ---------------------------------------------------------------------------

class BaseOTDisparityInit(nn.Module):
    """Base class for OT-based disparity initialization.

    Cost matrix  W x C  where  C = D + W :
        Row j  = left pixel (0..W-1).
        Col c  = right pixel coordinate  c - (D-1).
        Valid column for (j, d):  c = j - d + (D-1).

    Subclasses must set self.ot in __init__ and may override
    _augment_cost and _extract_plan.
    """

    def __init__(self, max_disp: int, image_width: int):
        super().__init__()
        self.max_disp = max_disp

        W = image_width // 4
        self.W = W
        self.C = max_disp + W - 1

        # Scatter indices: col[d][j] = j - d + (max_disp - 1)
        j = torch.arange(W)
        col_indices = torch.stack([j - d + (max_disp - 1) for d in range(max_disp)])
        self.register_buffer("_col_idx", col_indices)  # (D, W)

    def _scatter_to_cost(self, scores: Tensor) -> Tensor:
        """(B, D, H, W) similarity scores -> (B, H, W, C) cost = -scores."""
        B, D, H, W = scores.shape
        cost = scores.new_full((B, H, W, self.C), 1e4)
        j_idx = torch.arange(W, device=scores.device)
        for d in range(D):
            col = self._col_idx[d].to(scores.device)
            cost[:, :, j_idx, col] = -scores[:, d, :, :]
        return cost

    def _augment_cost(self, cost: Tensor) -> Tensor:
        """Hook for subclasses to augment cost (e.g. add dustbin)."""
        return cost

    def _extract_plan(self, P: Tensor) -> Tensor:
        """Hook for subclasses to extract relevant part of P."""
        return P

    def _plan_to_disparity(self, P: Tensor, D: int) -> Tensor:
        P = self._extract_plan(P)
        device, dtype = P.device, P.dtype
        right = torch.arange(self.C, device=device, dtype=dtype) - (D - 1)
        left = torch.arange(self.W, device=device, dtype=dtype)
        disp_map = left.view(1, 1, -1, 1) - right.view(1, 1, 1, -1)

        mass = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        disp = (P * disp_map).sum(dim=-1, keepdim=True) / mass

        disp = disp.squeeze(-1).unsqueeze(1)
        return disp

    def forward(self, scores: Tensor) -> Tensor:
        B, D, H, W = scores.shape
        cost = self._scatter_to_cost(scores)
        cost = self._augment_cost(cost)
        P = self.ot(cost)
        return self._plan_to_disparity(P, D)


# ---------------------------------------------------------------------------
# Unbalanced OT disparity initialisation
# ---------------------------------------------------------------------------

class OTDisparityInit(BaseOTDisparityInit):
    """Unbalanced OT disparity initialization."""

    def __init__(self, max_disp: int, image_width: int,
                 ot_iter: int = 10, epsilon: float = 1.0,
                 tau: float = 0.95,
                 adaptive_eps: bool = False, eps_base: float = 0.05):
        super().__init__(max_disp, image_width)
        self.epsilon = epsilon
        self.ot = UnbalancedOT(self.W, self.C, num_iter=ot_iter,
                               tau=tau, epsilon=epsilon,
                               adaptive_eps=adaptive_eps, eps_base=eps_base)


# ---------------------------------------------------------------------------
# Dustbin OT disparity initialisation
# ---------------------------------------------------------------------------

class DustbinOTDisparityInit(BaseOTDisparityInit):
    """Balanced OT with dustbin column only (no dustbin row).

    Dustbin cost per pixel is derived from two cues:
      - energy (logsumexp of scores): overall match confidence
      - margin (top1 - top2 scores): match uniqueness
    High energy + high margin = confident unique match = high occ cost.
    Low energy or low margin = ambiguous/uncertain = low occ cost (easy to occlude).
    """

    def __init__(self, max_disp: int, image_width: int,
                 ot_iter: int = 10, epsilon: float = 1.0,
                 occ_frac: float = 0.1,
                 adaptive_eps: bool = False, eps_base: float = 0.05):
        super().__init__(max_disp, image_width)
        self.epsilon = epsilon
        self.ot = DustbinOT(self.W, self.C, num_iter=ot_iter,
                            epsilon=epsilon, occ_frac=occ_frac,
                            adaptive_eps=adaptive_eps, eps_base=eps_base)
        self.occ_head = nn.Linear(2, 1)

    def forward(self, scores: Tensor) -> Tensor:
        B, D, H, W = scores.shape

        energy = torch.logsumexp(scores, dim=1)            # (B, H, W)
        top2 = scores.topk(2, dim=1).values                # (B, 2, H, W)
        margin = top2[:, 0] - top2[:, 1]                   # (B, H, W)

        feats = torch.stack([energy, margin], dim=-1)      # (B, H, W, 2)
        dustbin_cost = self.occ_head(feats)               # (B, H, W, 1)

        cost = self._scatter_to_cost(scores)
        cost = torch.cat([cost, dustbin_cost], dim=-1)      # (B, H, W, C+1)
        P = self.ot(cost)
        return self._plan_to_disparity(P, D)

    def _extract_plan(self, P: Tensor) -> Tensor:
        """Strip dustbin column."""
        return P[:, :, :, :self.C]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_disp_init(args) -> nn.Module:
    max_disp = args.max_disp // 4
    method = args.get("disp_init", "softmax") if hasattr(args, "get") else "softmax"

    if method == "softmax":
        return SoftmaxDisparityInit(max_disp)
    adaptive_eps = getattr(args, "ot_adaptive_eps", False)
    eps_base = getattr(args, "ot_eps_base", 0.05)

    if method == "ot":
        return OTDisparityInit(
            max_disp=max_disp,
            image_width=args.image_width,
            ot_iter=getattr(args, "ot_iter", 20),
            epsilon=getattr(args, "ot_epsilon", 1.0),
            tau=getattr(args, "ot_tau", 0.95),
            adaptive_eps=adaptive_eps,
            eps_base=eps_base,
        )
    if method == "ot_dustbin":
        return DustbinOTDisparityInit(
            max_disp=max_disp,
            image_width=args.image_width,
            ot_iter=getattr(args, "ot_iter", 20),
            epsilon=getattr(args, "ot_epsilon", 1.0),
            occ_frac=getattr(args, "ot_occ_frac", 0.1),
            adaptive_eps=adaptive_eps,
            eps_base=eps_base,
        )
    raise ValueError(f"Unknown disp_init method: {method}")
