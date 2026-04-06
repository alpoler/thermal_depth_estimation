"""
Test Sinkhorn OT against scipy's linear_sum_assignment (exact OT).

Problem:
  u = [1/4, 1/3, 5/12]          (source marginal, sums to 1)
  v = [5/24, 35/120, 1/3, 1/6]  (target marginal, sums to 1)
  C = [[3, 1, 7, 4],
       [2, 6, 5, 9],
       [8, 3, 3, 2]]

  min  sum_ij C_ij * T_ij
  s.t. T @ 1 = u,  T^T @ 1 = v,  T >= 0
"""

import torch
import numpy as np
from scipy.optimize import linprog


def sinkhorn(C: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor,
             eps: float = 0.01, n_iter: int = 200) -> torch.Tensor:
    """
    Log-domain Sinkhorn matching the style in disp_init.py.

    Args:
        C: (M, N) cost matrix
        log_mu: (M,) log of source marginal
        log_nu: (N,) log of target marginal
        eps: entropic regularisation
        n_iter: Sinkhorn iterations
    Returns:
        T: (M, N) transport plan
    """
    # Kernel in log-space: attn_ij = -C_ij / eps
    attn = -C / eps

    def logsumexp(x, dim):
        max_val = x.max(dim=dim, keepdim=True).values.clamp(min=-1e30)
        return max_val.squeeze(dim) + (x - max_val).exp().sum(dim=dim).log()

    # Sinkhorn iterations (same structure as _sinkhorn in disp_init.py)
    v = log_nu - logsumexp(attn, dim=0)
    u = log_mu - logsumexp(attn + v.unsqueeze(0), dim=1)
    for _ in range(n_iter - 1):
        v = log_nu - logsumexp(attn + u.unsqueeze(1), dim=0)
        u = log_mu - logsumexp(attn + v.unsqueeze(0), dim=1)

    T = (attn + u.unsqueeze(1) + v.unsqueeze(0)).exp()
    return T


def solve_ot_linprog(C_np, u_np, v_np):
    """Exact OT via linear programming (scipy)."""
    M, N = C_np.shape
    # Variables: T flattened as (M*N,)
    c = C_np.flatten()

    # Equality constraints: row sums = u, col sums = v
    A_eq = []
    b_eq = []
    # Row sum constraints
    for i in range(M):
        row = np.zeros(M * N)
        row[i * N:(i + 1) * N] = 1.0
        A_eq.append(row)
        b_eq.append(u_np[i])
    # Column sum constraints
    for j in range(N):
        col = np.zeros(M * N)
        col[j::N] = 1.0
        A_eq.append(col)
        b_eq.append(v_np[j])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    bounds = [(0, None)] * (M * N)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    assert result.success, f"linprog failed: {result.message}"
    T_exact = result.x.reshape(M, N)
    return T_exact, result.fun


def main():
    # Define problem
    u = torch.tensor([1/4, 1/3, 5/12], dtype=torch.float64)
    v = torch.tensor([5/24, 35/120, 1/3, 1/6], dtype=torch.float64)
    C = torch.tensor([
        [3, 1, 7, 4],
        [2, 6, 5, 9],
        [8, 3, 3, 2],
    ], dtype=torch.float64)

    print("Source marginal u:", u.numpy(), "  sum =", u.sum().item())
    print("Target marginal v:", v.numpy(), "  sum =", v.sum().item())
    print("Cost matrix C:")
    print(C.numpy())
    print()

    # --- Exact OT (scipy linprog) ---
    T_exact, cost_exact = solve_ot_linprog(C.numpy(), u.numpy(), v.numpy())
    print("=" * 50)
    print("EXACT OT (scipy linprog)")
    print("=" * 50)
    print("Transport plan T*:")
    print(np.array2string(T_exact, precision=6))
    print(f"Optimal cost: {cost_exact:.6f}")
    print(f"Row sums:  {T_exact.sum(axis=1)}  (should be {u.numpy()})")
    print(f"Col sums:  {T_exact.sum(axis=0)}  (should be {v.numpy()})")
    print()

    # --- Sinkhorn OT (our implementation) ---
    for eps in [1.0, 0.1, 0.01]:
        T_sink = sinkhorn(C, u.log(), v.log(), eps=eps, n_iter=300)
        cost_sink = (C * T_sink).sum().item()

        print("=" * 50)
        print(f"SINKHORN (eps={eps})")
        print("=" * 50)
        print("Transport plan T:")
        print(np.array2string(T_sink.numpy(), precision=6))
        print(f"Cost: {cost_sink:.6f}  (exact: {cost_exact:.6f},  gap: {abs(cost_sink - cost_exact):.6f})")
        print(f"Row sums:  {T_sink.sum(dim=1).numpy()}  (should be {u.numpy()})")
        print(f"Col sums:  {T_sink.sum(dim=0).numpy()}  (should be {v.numpy()})")
        print()


if __name__ == "__main__":
    main()
