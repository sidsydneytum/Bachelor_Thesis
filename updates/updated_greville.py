# updates/updated_greville.py
"""
UpdatedGrevilleUpdater: refined (Zhu-style) block Greville updates.

Key ideas (high level):
- When appending columns H_new to A (A ∈ R^{m×n}, H ∈ R^{m×p}):
    D := A^+ H            ∈ R^{n×p}
    C := H - A D          ∈ R^{m×p}   (block residual)

  Cases:
    1) Zero-residual (‖C‖ ≈ 0):
       Use dimension-aware formulas for B^T that avoid large solves:
         Option p×p (cheap if p ≤ m):
            B^T = (I_p + D^T D)^{-1} (D^T A^+)
         Option m×m (cheap if m < p):
            Let 𝔇̃ := D^T A^+  (shape p×m)
            B^T = 𝔇̃ (I_m + H 𝔇̃)^{-1}

    2) Full column rank residual (C has full column rank):
       Use Cholesky-based pseudoinverse for C^+:
         C^+ = (C^T C)^{-1} C^T
       Compute via Cholesky factorization (stable).

    3) Otherwise (mixed/ill-conditioned):
       Fall back to iterative Greville, column-by-column.

- Row updates are done by transposition symmetry:
    (A^+)^T = (A^T)^+  →  add rows to A  ==  add columns to A^T

This module focuses on the block update core (no ridge here). Keep `updates/direct.py`
as a correctness/stability baseline (ridge pinv), and `updates/greville.py` for
the classic iterative Greville reference.
"""

from __future__ import annotations
from typing import Tuple
import torch

# Fallback: iterative Greville for mixed cases
try:
    from .greville import GrevilleUpdater
except Exception:
    GrevilleUpdater = None


@torch.no_grad()
def _cholesky_pinv_ctc(C: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute C^+ = (C^T C)^{-1} C^T using Cholesky (with a small ridge eps).

    Args:
        C   : (m, p) residual block
        eps : tiny ridge to ensure SPD when C^T C is near-singular

    Returns:
        C_plus : (p, m)
    """
    # Solve (C^T C + eps I) X = C^T  →  X = (C^T C + eps I)^{-1} C^T
    Ct = C.transpose(0, 1).contiguous()                # (p, m)
    G = Ct @ C                                         # (p, p)
    p = G.shape[0]
    I = torch.eye(p, device=C.device, dtype=C.dtype)
    G = G + eps * I

    # Cholesky factorization: G = L L^T
    L = torch.linalg.cholesky(G)                       # (p, p), lower
    # Solve G X = C^T using the factors: first L y = C^T, then L^T X = y
    # torch.cholesky_solve expects upper=True by default in older APIs;
    # here we can do triangular solves explicitly for clarity:
    y = torch.linalg.solve_triangular(L, Ct, upper=False)
    X = torch.linalg.solve_triangular(L.transpose(0, 1), y, upper=True)
    return X                                           # (p, m)


@torch.no_grad()
def _zero_residual_Bt_dimaware(
    A_plus: torch.Tensor,  # (n, m)
    H: torch.Tensor,       # (m, p)
    D: torch.Tensor,       # (n, p)  = A_plus @ H
) -> torch.Tensor:
    """
    Zero-residual case: compute B^T with a dimension-aware choice.

    Two algebraically equivalent forms (choose by size):
      (1) p×p system (prefer if p <= m):
          B^T = (I_p + D^T D)^{-1} (D^T A^+)

      (2) m×m system (prefer if m < p):
          𝔇̃ := D^T A^+  (p×m)
          B^T = 𝔇̃ (I_m + H 𝔇̃)^{-1}

    Returns:
        B^T : (p, m)
    """
    m, p = H.shape[0], H.shape[1]
    n = A_plus.shape[0]
    device, dtype = H.device, H.dtype

    # 𝔇̃ = D^T A^+  (p×m)
    Dt = D.transpose(0, 1).contiguous()                # (p, n)
    Dtilde = Dt @ A_plus                               # (p, m)

    if p <= m:
        # Solve (I_p + D^T D) X = D^T A^+
        I_p = torch.eye(p, device=device, dtype=dtype)
        M = I_p + (Dt @ D)                             # (p, p)
        # X has shape (p, m) = B^T
        Bt = torch.linalg.solve(M, Dtilde)
        return Bt
    else:
        # Solve X = 𝔇̃ (I_m + H 𝔇̃)^{-1}
        I_m = torch.eye(m, device=device, dtype=dtype)
        M = I_m + (H @ Dtilde)                         # (m, m)
        Minv = torch.linalg.inv(M)                     # (m, m)  (small m preferred)
        Bt = Dtilde @ Minv                             # (p, m)
        return Bt


class UpdatedGrevilleUpdater:
    """
    Zhu-style refined block Greville updater.

    - add_columns: one-shot block update with case handling
        * zero residual (dimension-aware B^T)
        * full column rank residual (Cholesky-based C^+)
        * fallback to iterative Greville otherwise

    - add_rows: apply the same logic to A^T (column update),
                then transpose the pseudoinverse back.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps (float): numerical threshold (used both as residual norm gate
                         and for the Cholesky ridge).
        """
        self.eps = eps

    @torch.no_grad()
    def add_columns(
        self,
        A_old: torch.Tensor,       # (m, n)
        A_plus_old: torch.Tensor,  # (n, m)
        H_new: torch.Tensor,       # (m, p)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append p columns to A using the refined block Greville update.

        Returns:
            A_aug      : (m, n+p)
            A_plus_new : (n+p, m)
        """
        if H_new.numel() == 0:
            return A_old, A_plus_old

        m, n = A_old.shape
        p = H_new.shape[1]

        # D = A^+ H,  C = H - A D
        D = A_plus_old @ H_new                      # (n, p)
        C = H_new - (A_old @ D)                    # (m, p)

        # Check residual magnitude
        C_norm = torch.linalg.norm(C)
        zero_residual = (float(C_norm) <= self.eps * max(1, m, p))

        if zero_residual:
            # CASE 1: Zero residual → choose dimension-aware B^T
            Bt = _zero_residual_Bt_dimaware(A_plus_old, H_new, D)    # (p, m)
            # Assemble the updated pseudoinverse:
            #   [A|H]^+ = [ A^+ - D B^T
            #               B^T         ]
            top = A_plus_old - (D @ Bt)                              # (n, m)
            A_plus_new = torch.cat([top, Bt], dim=0)                # (n+p, m)
            A_aug = torch.cat([A_old, H_new], dim=1)                # (m, n+p)
            return A_aug, A_plus_new

        # Check full column rank for C (numerically)
        # We attempt Cholesky on C^T C; if it fails, we fall back.
        try:
            # CASE 2: Full column rank residual → C^+ via Cholesky
            C_plus = _cholesky_pinv_ctc(C, eps=self.eps)            # (p, m)
            Bt = C_plus                                             # (p, m)

            top = A_plus_old - (D @ Bt)                             # (n, m)
            A_plus_new = torch.cat([top, Bt], dim=0)               # (n+p, m)
            A_aug = torch.cat([A_old, H_new], dim=1)               # (m, n+p)
            return A_aug, A_plus_new

        except RuntimeError:
            # CASE 3: Mixed/ill-conditioned → fall back to iterative Greville
            if GrevilleUpdater is None:
                raise RuntimeError(
                    "Greville fallback is unavailable. Ensure updates/greville.py is present."
                )
            g = GrevilleUpdater(eps=self.eps)
            return g.add_columns(A_old, A_plus_old, H_new)

    @torch.no_grad()
    def add_rows(
        self,
        A_old: torch.Tensor,       # (m, n)
        A_plus_old: torch.Tensor,  # (n, m)
        A_x_new: torch.Tensor,     # (q, n)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append q rows to A using the block method on the transposed system.

        We use the identity:
            (A^+)^T = (A^T)^+

        Procedure:
            1) Work on M := A^T (n×m) with P := (A^+)^T (m×n)
            2) Add columns H_cols := (A_x_new)^T (n×q) using the same column routine
            3) Transpose results back:
                 A_aug      = (M_aug)^T     (m+q, n)
                 A_plus_new = (P_new)^T     (n, m+q)
        """
        if A_x_new.numel() == 0:
            return A_old, A_plus_old

        # Transposed system
        M_old = A_old.transpose(0, 1).contiguous()        # (n, m)
        P_old = A_plus_old.transpose(0, 1).contiguous()   # (m, n)
        H_cols = A_x_new.transpose(0, 1).contiguous()     # (n, q)

        # Reuse the column algorithm on the transposed problem.
        # To avoid infinite recursion, call the same logic inline:
        # D_t = P_old @ H_cols   (m×q),   C_t = H_cols - M_old D_t (n×q)
        D_t = P_old @ H_cols
        C_t = H_cols - (M_old @ D_t)

        n, m = M_old.shape
        q = H_cols.shape[1]
        C_norm = torch.linalg.norm(C_t)
        zero_residual = (float(C_norm) <= self.eps * max(1, n, q))

        if zero_residual:
            # zero-residual on transposed system:
            # Bt_t = dimension-aware result for the transposed shapes
            Bt_t = _zero_residual_Bt_dimaware(P_old, H_cols, D_t)    # (q, n)
            top_t = P_old - (D_t @ Bt_t)                             # (m, n)
            P_new = torch.cat([top_t, Bt_t], dim=0)                 # (m+q, n)
            M_aug = torch.cat([M_old, H_cols], dim=1)               # (n, m+q)

        else:
            # Try Cholesky on C_t^T C_t
            try:
                C_t_plus = _cholesky_pinv_ctc(C_t, eps=self.eps)     # (q, n)
                Bt_t = C_t_plus
                top_t = P_old - (D_t @ Bt_t)                         # (m, n)
                P_new = torch.cat([top_t, Bt_t], dim=0)             # (m+q, n)
                M_aug = torch.cat([M_old, H_cols], dim=1)           # (n, m+q)
            except RuntimeError:
                # Fallback to iterative Greville on the transposed system
                if GrevilleUpdater is None:
                    raise RuntimeError(
                        "Greville fallback is unavailable. Ensure updates/greville.py is present."
                    )
                g = GrevilleUpdater(eps=self.eps)
                M_aug, P_new = g.add_columns(M_old, P_old, H_cols)

        # Transpose back to original orientation
        A_aug = M_aug.transpose(0, 1).contiguous()        # (m+q, n)
        A_plus_new = P_new.transpose(0, 1).contiguous()   # (n, m+q)
        return A_aug, A_plus_new
