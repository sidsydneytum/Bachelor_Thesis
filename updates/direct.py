# updates/direct.py
import torch

def ridge_pinv(A: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """
    Compute the ridge-regularized pseudoinverse:
        A^+ = (A^T A + λ I)^(-1) A^T

    Args:
        A (torch.Tensor): Input matrix of shape (m, n).
        lam (float): Regularization parameter λ to improve stability.

    Returns:
        torch.Tensor: Pseudoinverse of A with shape (n, m).
    """
    AtA = A.T @ A                     # (n, n) matrix
    n = AtA.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(AtA + lam * I, A.T)  # solve linear system instead of inverting

class DirectUpdater:
    """
    Baseline updater for experiments.
    - Does NOT use incremental Greville or Zhu updates.
    - Each time new rows or columns are added, it recomputes the pseudoinverse from scratch.
    - Useful as a correctness baseline, but computationally expensive.
    """

    def __init__(self, lam: float = 1e-3):
        """
        Args:
            lam (float): Regularization parameter λ for ridge pseudoinverse.
        """
        self.lam = lam

    def add_columns(self, A_old: torch.Tensor, A_plus_old: torch.Tensor, H_new: torch.Tensor):
        """
        Add new columns to A (e.g., adding enhancement nodes).

        Args:
            A_old (torch.Tensor): Original matrix A with shape (m, n).
            A_plus_old (torch.Tensor): Previous pseudoinverse (not used here).
            H_new (torch.Tensor): New columns to append, shape (m, p).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - A_aug: Updated matrix [A_old | H_new] with shape (m, n+p).
                - A_plus_new: New pseudoinverse with shape (n+p, m).
        """
        A_aug = torch.cat([A_old, H_new], dim=1)         # concatenate columns
        A_plus_new = ridge_pinv(A_aug, self.lam)         # recompute pseudoinverse
        return A_aug, A_plus_new

    def add_rows(self, A_old: torch.Tensor, A_plus_old: torch.Tensor, A_x_new: torch.Tensor):
        """
        Add new rows to A (e.g., adding new training samples).

        Args:
            A_old (torch.Tensor): Original matrix A with shape (m, n).
            A_plus_old (torch.Tensor): Previous pseudoinverse (not used here).
            A_x_new (torch.Tensor): New rows to append, shape (q, n).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - A_aug: Updated matrix with shape (m+q, n).
                - A_plus_new: New pseudoinverse with shape (n, m+q).
        """
        A_aug = torch.cat([A_old, A_x_new], dim=0)       # concatenate rows
        A_plus_new = ridge_pinv(A_aug, self.lam)         # recompute pseudoinverse
        return A_aug, A_plus_new
