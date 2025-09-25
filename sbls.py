# sbls.py
# Minimal Spiking Broad Learning System (SBLS) with pluggable pseudoinverse updater.
# Works on CPU and Apple Silicon (MPS). Requires: torch, torchvision (for data), snntorch.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import snntorch.spikegen as spikegen


# ---------------------------
# Helpers
# ---------------------------

def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode integer labels to float tensor (N, C)."""
    return F.one_hot(y.to(torch.long), num_classes=num_classes).float()


def ridge_pinv(A: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """Ridge-regularized pseudoinverse  A^+ = (A^T A + λI)^{-1} A^T  (n, m)."""
    AtA = A.T @ A
    n = AtA.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(AtA + lam * I, A.T)


# ---------------------------
# Configuration
# ---------------------------

@dataclass
class SBLSConfig:
    in_dim: int = 28 * 28
    out_dim: int = 10
    s: int = 30                         # spike-train length (timesteps)
    n_feat: int = 100                   # number of feature nodes (Z size)
    mw: int = 20                        # number of enhancement windows
    mp: int = 100                       # enhancement nodes per window
    lif_beta: float = 0.9               # LIF leak
    lif_thresh: float = 1.0             # LIF threshold
    lam: float = 1e-3                   # ridge λ for A^+
    seed: int = 0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    output_is_one_hot: bool = True      # keep True for classification


# ---------------------------
# SBLS model
# ---------------------------

class SBLS(nn.Module):
    """
    Spiking Broad Learning System (SBLS)
    - Random feature layer (linear + bias) -> rate coding (binary spikes)
    - Enhancement layer: multiple windows, each is linear + bias + LIF → spikes
    - Aggregate spike trains over time to dense features
    - Concatenate [Z | H] = A and train linear readout W with ridge pseudoinverse
    - Optional incremental updates:
        * add_enhancement_windows(...)  → add columns to A via updater.add_columns
        * add_training_rows(...)        → add rows to A via updater.add_rows
    """

    def __init__(self, cfg: SBLSConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        # Random weights for feature layer (input → Z)
        w1_range = (1.0 / max(1, cfg.in_dim)) ** 0.5
        self.W1 = nn.Parameter(
            torch.empty(cfg.in_dim, cfg.n_feat, device=self.device, dtype=cfg.dtype).uniform_(-w1_range, w1_range),
            requires_grad=False,
        )
        self.B1 = nn.Parameter(
            torch.empty(cfg.n_feat, device=self.device, dtype=cfg.dtype).uniform_(-w1_range, w1_range),
            requires_grad=False,
        )

        # Enhancement windows (Z → MP) lists
        w2_range = (1.0 / max(1, cfg.n_feat)) ** 0.5
        self.W2_list = nn.ParameterList([
            nn.Parameter(
                torch.empty(cfg.n_feat, cfg.mp, device=self.device, dtype=cfg.dtype).uniform_(-w2_range, w2_range),
                requires_grad=False,
            ) for _ in range(cfg.mw)
        ])
        self.B2_list = nn.ParameterList([
            nn.Parameter(
                torch.empty(cfg.mp, device=self.device, dtype=cfg.dtype).uniform_(-w2_range, w2_range),
                requires_grad=False,
            ) for _ in range(cfg.mw)
        ])

        # Spiking components
        self.spikegen = spikegen.rate             # rate encoder
        self.lif = snn.Leaky(beta=cfg.lif_beta, init_hidden=True).to(self.device)

        # Readout and caches (filled by fit_with_A)
        self.W_out: Optional[torch.Tensor] = None     # (n_cols, out_dim)
        self.A_plus: Optional[torch.Tensor] = None    # (n_cols, N)  pseudoinverse of training A
        self.A_train: Optional[torch.Tensor] = None   # (N, n_cols)  cached training A
        self.Y_train: Optional[torch.Tensor] = None   # (N, out_dim) one-hot
        self.X_train: Optional[torch.Tensor] = None   # cached inputs for rebuilding A on updates

    # ---------------------------
    # Core builders
    # ---------------------------

    def _feature_spikes(self, X: torch.Tensor) -> torch.Tensor:
        """
        Produce feature-layer spike trains via rate coding.
        X: (N, in_dim) scaled to [0,1] (torch.float)
        Returns: Z_spk (s, N, n_feat) in {0,1}
        """
        # Random linear transform + bias, squash to [0,1] for spike probability
        Z_lin = X @ self.W1 + self.B1          # (N, n_feat)
        Z_prob = torch.sigmoid(Z_lin / 3.0)    # softer spread → stable coding
        Z_spk = self.spikegen(Z_prob, num_steps=self.cfg.s)  # (s, N, n_feat)
        return Z_spk

    def _lif_enhancement_spikes(self, Z_spk: torch.Tensor) -> torch.Tensor:
        """
        Run LIF per window over time.
        Z_spk: (s, N, n_feat) → returns H_spk: (s, N, mw*mp)
        """
        s, N, _ = Z_spk.shape
        H_blocks = []
        for W2, B2 in zip(self.W2_list, self.B2_list):
            # Pre-activation per timestep: (N, mp)
            H_post_list = []
            self.lif.reset_mem()
            for t in range(s):
                H_pre_t = Z_spk[t] @ W2 + B2
                H_post_list.append(self.lif(H_pre_t))
            H_post = torch.stack(H_post_list, dim=0)  # (s, N, mp)
            H_blocks.append(H_post)
        return torch.cat(H_blocks, dim=2)            # (s, N, mw*mp)

    @staticmethod
    def _aggregate_time(spk: torch.Tensor, dim_time: int = 0) -> torch.Tensor:
        """Aggregate spike trains to dense features via mean over time (rate)."""
        return spk.mean(dim=dim_time)

    def _build_A_internal(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Z, H, A for inputs X.
        Returns:
            Z_agg: (N, n_feat)
            H_agg: (N, mw*mp)
            A    : (N, n_feat + mw*mp)
        """
        Z_spk = self._feature_spikes(X)     # (s, N, n_feat)
        H_spk = self._lif_enhancement_spikes(Z_spk)  # (s, N, mw*mp)
        Z_agg = self._aggregate_time(Z_spk)          # (N, n_feat)
        H_agg = self._aggregate_time(H_spk)          # (N, mw*mp)
        A = torch.cat([Z_agg, H_agg], dim=1)         # (N, n_cols)
        return Z_agg, H_agg, A

    # ---------------------------
    # Public hooks used by the runner
    # ---------------------------

    @torch.no_grad()
    def build_A(self, X: torch.Tensor) -> torch.Tensor:
        """
        Entry point used by runners: returns A for inputs X.
        X is expected in [0,1] with shape (N, in_dim).
        """
        X = X.to(device=self.device, dtype=self.cfg.dtype).reshape(-1, self.cfg.in_dim)
        _, _, A = self._build_A_internal(X)
        return A

    @torch.no_grad()
    def fit_with_A(self, A: torch.Tensor, y: torch.Tensor, lam: float | None = None) -> None:
        """
        Train linear readout W_out from a precomputed A and labels y.
        Stores A_plus, W_out, and caches training A/Y for future incremental updates.
        """
        lam = self.cfg.lam if lam is None else lam
        Y = one_hot(y.to(self.device), self.cfg.out_dim) if self.cfg.output_is_one_hot else y.to(self.device)
        A = A.to(self.device)
        A_plus = ridge_pinv(A, lam)                     # (n_cols, N)
        W = A_plus @ Y                                  # (n_cols, out_dim)
        self.A_plus = A_plus
        self.W_out = W
        self.A_train = A
        self.Y_train = Y
        # NOTE: set X_train via fit(...) if you want to support column-add updates
        # that regenerate new enhancement windows from the *original* inputs.

    @torch.no_grad()
    def predict_from_A(self, A: torch.Tensor) -> torch.Tensor:
        """Predict integer labels from a precomputed A using current W_out."""
        if self.W_out is None:
            raise RuntimeError("Model not trained. Call fit_with_A(...) first.")
        logits = A.to(self.device) @ self.W_out
        return logits.argmax(dim=1)

    # ---------------------------
    # Convenience end-to-end fit/predict (when you have X directly)
    # ---------------------------

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, store_X_for_updates: bool = True, lam: float | None = None) -> None:
        """
        Convenience method: builds A from X and trains.
        Optionally caches X for future enhancement-window growth (A3).
        """
        X = X.to(device=self.device, dtype=self.cfg.dtype).reshape(-1, self.cfg.in_dim)
        self.X_train = X if store_X_for_updates else None
        _, _, A = self._build_A_internal(X)
        self.fit_with_A(A, y, lam=lam)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Convenience: builds A then predicts."""
        A = self.build_A(X)
        return self.predict_from_A(A)

    # ---------------------------
    # Incremental updates
    # ---------------------------

    @torch.no_grad()
    def add_enhancement_windows(
        self,
        num_new_windows: int,
        updater=None,
        lam: Optional[float] = None,
    ) -> None:
        """
        Add enhancement windows (columns in A) and update A^+ and W_out.
        If an 'updater' with .add_columns(...) is provided, it is used.
        Otherwise, we recompute A^+ from scratch as a safe baseline.

        Requires: a cached training set (X_train, A_train, Y_train) from fit/fit_with_A.
        """
        if num_new_windows <= 0:
            return
        if self.A_train is None or self.Y_train is None:
            raise RuntimeError("No cached training set. Train with fit/fit_with_A first.")
        if self.X_train is None:
            # If we don't have X, we cannot regenerate new enhancement features consistently.
            # Fall back to recomputing A with current weights only (no new windows).
            raise RuntimeError("X_train not cached. Fit with store_X_for_updates=True to add windows.")

        cfg = self.cfg
        # 1) Create new random windows
        w2_range = (1.0 / max(1, cfg.n_feat)) ** 0.5
        new_W2 = []
        new_B2 = []
        for _ in range(num_new_windows):
            W2 = torch.empty(cfg.n_feat, cfg.mp, device=self.device, dtype=cfg.dtype).uniform_(-w2_range, w2_range)
            B2 = torch.empty(cfg.mp, device=self.device, dtype=cfg.dtype).uniform_(-w2_range, w2_range)
            new_W2.append(W2)
            new_B2.append(B2)

        # 2) Build H_new (aggregated enhancement features for training X) for *only* the new windows
        Z_spk = self._feature_spikes(self.X_train)  # (s, N, n_feat)
        H_blocks = []
        for W2, B2 in zip(new_W2, new_B2):
            self.lif.reset_mem()
            H_post_list = []
            for t in range(cfg.s):
                H_pre_t = Z_spk[t] @ W2 + B2
                H_post_list.append(self.lif(H_pre_t))
            H_post = torch.stack(H_post_list, dim=0)  # (s, N, mp)
            H_blocks.append(H_post)
        H_new_spk = torch.cat(H_blocks, dim=2)        # (s, N, num_new_windows*mp)
        H_new = self._aggregate_time(H_new_spk)       # (N, p)
        p = H_new.shape[1]

        # 3) Update A_plus (columns)
        lam_val = cfg.lam if lam is None else lam
        if updater is not None and hasattr(updater, "add_columns") and self.A_plus is not None:
            A_aug, A_plus_new = updater.add_columns(self.A_train, self.A_plus, H_new)
        else:
            # safe baseline: rebuild A with the new windows, then recompute A^+
            # Expand parameter lists permanently
            for W2, B2 in zip(new_W2, new_B2):
                self.W2_list.append(nn.Parameter(W2, requires_grad=False))
                self.B2_list.append(nn.Parameter(B2, requires_grad=False))
            # Rebuild full A for training set
            _, _, A_aug = self._build_A_internal(self.X_train)
            A_plus_new = ridge_pinv(A_aug, lam_val)

        # 4) Update readout W_out
        self.A_train = A_aug
        self.A_plus = A_plus_new
        self.W_out = self.A_plus @ self.Y_train

        # 5) If updater succeeded, also persist the new window parameters
        if len(self.W2_list) * cfg.mp < (A_aug.shape[1] - cfg.n_feat):
            # We appended columns virtually; persist weights now
            for W2, B2 in zip(new_W2, new_B2):
                self.W2_list.append(nn.Parameter(W2, requires_grad=False))
                self.B2_list.append(nn.Parameter(B2, requires_grad=False))

    @torch.no_grad()
    def add_training_rows(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        updater=None,
        lam: Optional[float] = None,
    ) -> None:
        """
        Add new training samples (rows in A) and update A^+ and W_out.
        If an 'updater' with .add_rows(...) is provided, it is used.
        Otherwise we recompute A^+ on concatenated A as a safe baseline.
        """
        if self.A_train is None or self.Y_train is None or self.A_plus is None:
            raise RuntimeError("No cached training set. Train with fit/fit_with_A first.")

        X_new = X_new.to(device=self.device, dtype=self.cfg.dtype).reshape(-1, self.cfg.in_dim)
        _, _, A_x_new = self._build_A_internal(X_new)
        Y_new = one_hot(y_new.to(self.device), self.cfg.out_dim) if self.cfg.output_is_one_hot else y_new.to(self.device)

        lam_val = self.cfg.lam if lam is None else lam
        if updater is not None and hasattr(updater, "add_rows"):
            A_aug, A_plus_new = updater.add_rows(self.A_train, self.A_plus, A_x_new)
        else:
            # safe baseline: stack A and recompute
            A_aug = torch.cat([self.A_train, A_x_new], dim=0)
            A_plus_new = ridge_pinv(A_aug, lam_val)

        Y_all = torch.cat([self.Y_train, Y_new], dim=0)
        self.A_train = A_aug
        self.A_plus = A_plus_new
        self.Y_train = Y_all
        self.W_out = self.A_plus @ self.Y_train
