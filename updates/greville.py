# updates/greville.py
"""
GrevilleUpdater: classic iterative Greville updates for the Moore–Penrose pseudoinverse.

- add_columns(A_old, A_plus_old, H_new): append columns (e.g., new enhancement nodes)
  using the single-column Greville update repeatedly.

- add_rows(A_old, A_plus_old, A_x_new): append rows (e.g., new training samples)
  by applying Greville to the transposed system and transposing back:
      (A^+)^T = (A^T)^+

This module intentionally implements the *original* Greville formulas (no ridge).
Use updates/direct.py as a stability/correctness baseline (ridge pinv).
Use updated_greville.py for Zhu's refined, block, and Cholesky-based variants.
"""

from __future__ import annotations
from typing import Tuple
import torch


@torch.no_grad()
def _greville_add_single_column(
    A: torch.Tensor,        # (m, n)
    A_plus: torch.Tensor,   # (n, m)
    a: torch.Tensor,        # (m,) or (m,1)
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Greville's single-column update.

    Given:
        A      ∈ ℝ^{m×n}
        A_plus ∈ ℝ^{n×m}  (Moore–Penrose pseudoinverse of A)
        a      ∈ ℝ^{m}    (new column to append)

    Define:
        d = A_plus a        ∈ ℝ^{n×1}
        c = a - A d         ∈ ℝ^{m×1}

    If ||c||_2 > 0 (full residual):
        b^T = c^T / (c^T c)
    Else (zero residual):
        b^T = (1 + d^T d)^{-1} d^T A_plus

    Update:
        [A | a]^+ = [ A_plus - d b^T
                       b^T               ]
    """
    if a.dim() == 1:
        a = a.unsqueeze(1)  # (m,1)

    # Shapes: A: (m,n), A_plus: (n,m), a: (m,1)
    d = A_plus @ a          # (n,1)
    c = a - (A @ d)         # (m,1)

    c_norm2 = (c * c).sum()  # scalar
    if float(c_norm2) > eps:
        # Full residual case
        bT = c.transpose(0, 1) / c_norm2  # (1,m)
    else:
        # Zero residual case
        denom = 1.0 + (d.transpose(0, 1) @ d)  # (1,1)
        # Numerical guard
        if float(denom) <= eps:
            denom = denom + eps
        bT = (d.transpose(0, 1) @ A_plus) / denom  # (1,m)

    # Top block: A_plus - d b^T  (n,m)
    top = A_plus - (d @ bT)
    # Bottom block: b^T          (1,m)
    A_plus_new = torch.cat([top, bT], dim=0)  # (n+1, m)
    A_new = torch.cat([A, a], dim=1)          # (m, n+1)

    return A_new, A_plus_new


class GrevilleUpdater:
    """
    Original Greville (iterative) updater.

    Notes:
      - No ridge regularization here (pure Greville). If your matrices are ill-conditioned,
        consider using updates/direct.py as a baseline for comparison.
      - add_columns processes H_new column-by-column.
      - add_rows applies the same single-column update to the transposed system (A^T),
        then transposes the resulting pseudoinverse back.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps (float): Numerical threshold to decide the zero-residual branch.
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
        Append p new columns using Greville updates.

        Args:
            A_old      (m,n): current A
            A_plus_old (n,m): current A^+
            H_new      (m,p): columns to append

        Returns:
            A_aug      (m,n+p)
            A_plus_new (n+p,m)
        """
        if H_new.numel() == 0:
            return A_old, A_plus_old

        A = A_old
        A_plus = A_plus_old
        # Iterate through each new column
        for j in range(H_new.shape[1]):
            col = H_new[:, j]
            A, A_plus = _greville_add_single_column(A, A_plus, col, eps=self.eps)
        return A, A_plus

    @torch.no_grad()
    def add_rows(
        self,
        A_old: torch.Tensor,       # (m, n)
        A_plus_old: torch.Tensor,  # (n, m)
        A_x_new: torch.Tensor,     # (q, n)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append q new rows using Greville via transposition.

        Idea:
            (A^+)^T = (A^T)^+
        Adding rows to A is equivalent to adding columns to A^T.

        Steps:
            1) Let M = A^T  (n×m), P = (A_plus_old)^T (m×n)
            2) For each new row r (1×n), treat r^T as a new column (n×1) for M
            3) Apply single-column Greville update on (M, P)
            4) Transpose the new pseudoinverse back to get A_plus_new

        Returns:
            A_aug      (m+q, n)
            A_plus_new (n, m+q)
        """
        if A_x_new.numel() == 0:
            return A_old, A_plus_old

        # Transposed system
        M = A_old.transpose(0, 1).contiguous()       # (n, m)
        P = A_plus_old.transpose(0, 1).contiguous()  # (m, n) = (A^T)^+

        # Add each new row as a column in the transposed system
        for i in range(A_x_new.shape[0]):
            row = A_x_new[i, :]          # (n,)
            col = row.unsqueeze(1)       # (n,1)
            M, P = _greville_add_single_column(M, P, col, eps=self.eps)

        # Build augmented A (rows appended)
        A_aug = A_old.new_empty((A_old.shape[0] + A_x_new.shape[0], A_old.shape[1]))
        A_aug[: A_old.shape[0], :] = A_old
        A_aug[A_old.shape[0] :, :] = A_x_new

        # Transpose P back to get A_plus_new
        A_plus_new = P.transpose(0, 1).contiguous()  # (n, m+q)

        return A_aug, A_plus_new
