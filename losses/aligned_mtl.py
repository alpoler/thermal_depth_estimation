import torch
import torch.nn as nn


def _eigh_2x2(M: torch.Tensor):
    """Closed-form eigendecomposition of a 2x2 symmetric matrix.

    Returns (eigenvalues [2], eigenvectors [2,2]).
    Eigenvalue order: [λ_max, λ_min] to match eigenvector columns
    V[:,0] = [cos θ, sin θ] for λ_max, V[:,1] = [-sin θ, cos θ] for λ_min.
    """
    a, b, d = M[0, 0], M[0, 1], M[1, 1]

    trace = a + d
    diff = a - d
    disc = (diff * diff + 4.0 * b * b).sqrt()

    # V[:,0] = [c, s] is eigenvector of λ_max, V[:,1] = [-s, c] is eigenvector of λ_min
    # so eigenvalues must follow the same column order: [λ_max, λ_min]
    lam = torch.stack([0.5 * (trace + disc), 0.5 * (trace - disc)])

    theta = 0.5 * torch.atan2(2.0 * b, diff)
    c = theta.cos()
    s = theta.sin()
    V = torch.stack([torch.stack([c, -s]),
                     torch.stack([s, c])])

    return lam, V


class AlignedMTL(nn.Module):
    """Aligned-MTL gradient combiner for exactly 2 tasks (Senushkin et al., CVPR 2023).

    Registers task weights as a buffer so they travel with the model state_dict
    and respect device/dtype transfers automatically.

    Usage:
        mtl = AlignedMTL(weights=[0.9, 0.1])
        combined_grad = mtl(grads_depth, grads_reg)
    """

    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        if weights is None:
            weights = [0.5, 0.5]
        self.register_buffer('w', torch.tensor(weights))

    def forward(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """Combine two flat gradient vectors via Aligned-MTL.

        Args:
            g1: Flat gradient for task 1, shape (|θ|,).
            g2: Flat gradient for task 2, shape (|θ|,).

        Returns:
            Combined gradient, shape (|θ|,). Modifies g1 in-place.
        """
        # Gram matrix (2x2)
        M = torch.stack([torch.stack([torch.dot(g1, g1), torch.dot(g1, g2)]),
                         torch.stack([torch.dot(g1, g2), torch.dot(g2, g2)])])

        lam, V = _eigh_2x2(M)
        lam = lam.clamp(min=1e-12)

        # B = V diag(sqrt(λ_min / λ_i)) V^T,  alpha = B @ w
        # lam order is [λ_max, λ_min], so λ_min = lam[1]
        scale = (lam[1] / lam).sqrt()
        alpha = V @ (scale * (V.t() @ self.w))

        g1.mul_(alpha[0].item()).add_(g2, alpha=alpha[1].item())
        return g1
