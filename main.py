import os
import random
import torch
import time
import psutil
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any, Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# -------------------- GPU configuration -------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = (DEVICE.type == "cuda")
print(f"Using device: {DEVICE}")

# ------------------- Config ------------------- #
SEED = 0
NPS = [50, 100]          # Number of features to select
MWS = [10, 20]           # Enhancement windows
UPDATERS = ["direct", "greville", "updated_greville"]  # Updater methods to compare
MP = 100                 # Enhancement nodes per window
NW = 1                   # Feature windows (as in Kellinger MNIST)
SPIKE_STEPS = 30         # Number of spike steps
RIDGE = 0.0              # Ridge regularization
MNIST_PATH = "./data/mnist"
SAVE_DIR = "figures/confusionmatrix"

#-------------------Utlities-------------------#
def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_onehot(y:np.ndarray, num_classes:int) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

##---------------Time & Memory-------------##
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0

def peak_mem_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)

##----------------sbls helpers-------------##

def aggregate_spikes(spk):
    """Sum over time then per-sample min-max normalize to [0,1]. spk: (N,D,steps) -> (N,D)"""
    s = spk.sum(axis=2)
    mn = s.min(axis=1, keepdims=True)
    mx = s.max(axis=1, keepdims=True)
    return ((s - mn) / (mx - mn + 1e-8)).astype(np.float32)

def lif_like(x: np.ndarray) -> np.ndarray:
    """Lightweight nonlinearity mapped to [0,1]."""
    z = np.tanh(x).astype(np.float32)
    mn = z.min(axis=1, keepdims=True)
    mx = z.max(axis=1, keepdims=True)
    return ((z - mn) / (mx - mn + 1e-8)).astype(np.float32)

def rate_code(arr, steps):
    # arr in [0,1], out (N,D,steps) with Bernoulli(arr)
    N, D = arr.shape
    probs = np.repeat(arr[:, :, None], steps, axis=2)
    return (np.random.rand(N, D, steps) < probs).astype(np.float32)

##----------------pinv_ridge-------------##

def pinv_ridge_np(A: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    AtA = A.T @ A
    G = AtA + lam * np.eye(AtA.shape[0], dtype=A.dtype)
    try:
        X = np.linalg.solve(G, A.T)
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(G) @ A.T
    return X  # (d x N)

##----------------plot confusion matrix-----------##

def plot_confusionmatrix(cm: np.ndarray, title: str, path: str):
    cm = cm.astype(np.float32)
    rs = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, rs, out=np.zeros_like(cm), where=rs!=0)

    plt.figure(figsize=(5,4), dpi=140)
    plt.imshow(cm, interpolation="nearest")  # no colorbar
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    ticks = range(cm.shape[0])
    plt.xticks(ticks, fontsize=6)
    plt.yticks(ticks, fontsize=6)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=("black" if val > 0.5 else "white"), fontsize=7)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

#-------------------Data Loader (MNIST)-------------------#
def load_mnist(flatten: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tf = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(MNIST_PATH, train=True, download=True, transform=tf)
    mnist_test  = datasets.MNIST(MNIST_PATH, train=False, download=True, transform=tf)

    X_train = mnist_train.data.numpy().astype(np.float32) / 255.0  # (60000, 28, 28)
    y_train = mnist_train.targets.numpy().astype(np.int64)
    X_test  = mnist_test.data.numpy().astype(np.float32) / 255.0   # (10000, 28, 28)
    y_test  = mnist_test.targets.numpy().astype(np.int64)

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)  # (N, 784)
        X_test  = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test



#-------------------Random Layers-------------------#
def init_feature_layer(in_dim, np_per_win, nw, rng):
    Ws, bs = [], []
    for _ in range(nw):
        Ws.append(rng.randn(in_dim, np_per_win).astype(np.float32) * 0.2)
        bs.append((rng.rand(np_per_win).astype(np.float32) - 0.5) * 0.1)
    return Ws, bs

def forward_feature_layer(X, Ws, bs, steps, rng):
    outs = []
    for W, b in zip(Ws, bs):
        lin = X @ W + b[None, :]
        # rescale to [0,1], spike, aggregate
        mn = lin.min(axis=1, keepdims=True); mx = lin.max(axis=1, keepdims=True)
        norm = (lin - mn) / (mx - mn + 1e-8)
        spk = rate_code(norm, steps)
        outs.append(aggregate_spikes(spk))
    return np.concatenate(outs, axis=1).astype(np.float32)

def init_enh_layer(feat_dim, mp_per_win, mw, rng):
    Ws, bs = [], []
    for _ in range(mw):
        Ws.append(rng.randn(feat_dim, mp_per_win).astype(np.float32) * 0.2)
        bs.append((rng.rand(mp_per_win).astype(np.float32) - 0.5) * 0.1)
    return Ws, bs

def forward_enh_layer(Z, Ws, bs, steps, rng):
    outs = []
    for W, b in zip(Ws, bs):
        lin = Z @ W + b[None, :]
        z = lif_like(lin)
        spk = rate_code(z, steps)
        outs.append(aggregate_spikes(spk))
    return np.concatenate(outs, axis=1).astype(np.float32)


# =================== GPU helpers (Torch) =================== #
def rate_code_torch(arr: torch.Tensor, steps: int) -> torch.Tensor:
    N, D = arr.shape
    probs = arr.unsqueeze(-1).expand(N, D, steps)
    return (torch.rand((N, D, steps), device=arr.device, dtype=arr.dtype) < probs).to(arr.dtype)

def aggregate_spikes_torch(spk: torch.Tensor) -> torch.Tensor:
    s = spk.sum(dim=2)
    mn = s.min(dim=1, keepdim=True).values
    mx = s.max(dim=1, keepdim=True).values
    return (s - mn) / (mx - mn + 1e-8)

def lif_like_torch(x: torch.Tensor) -> torch.Tensor:
    z = torch.tanh(x)
    mn = z.min(dim=1, keepdim=True).values
    mx = z.max(dim=1, keepdim=True).values
    return (z - mn) / (mx - mn + 1e-8)

def init_feature_layer_torch(in_dim, np_per_win, nw, rng_np, device):
    Ws, bs = [], []
    for _ in range(nw):
        W = torch.tensor(rng_np.randn(in_dim, np_per_win).astype(np.float32) * 0.2, device=device)
        b = torch.tensor((rng_np.rand(np_per_win).astype(np.float32) - 0.5) * 0.1, device=device)
        Ws.append(W); bs.append(b)
    return Ws, bs

def init_enh_layer_torch(feat_dim, mp_per_win, mw, rng_np, device):
    Ws, bs = [], []
    for _ in range(mw):
        W = torch.tensor(rng_np.randn(feat_dim, mp_per_win).astype(np.float32) * 0.2, device=device)
        b = torch.tensor((rng_np.rand(mp_per_win).astype(np.float32) - 0.5) * 0.1, device=device)
        Ws.append(W); bs.append(b)
    return Ws, bs

def forward_feature_layer_torch(X: torch.Tensor, Ws: list, bs: list, steps: int) -> torch.Tensor:
    outs = []
    for W, b in zip(Ws, bs):
        lin = X @ W + b.unsqueeze(0)
        mn = lin.min(dim=1, keepdim=True).values
        mx = lin.max(dim=1, keepdim=True).values
        norm = (lin - mn) / (mx - mn + 1e-8)
        spk = rate_code_torch(norm, steps)
        outs.append(aggregate_spikes_torch(spk))
    return torch.cat(outs, dim=1).to(torch.float32)

def forward_enh_layer_torch(Z: torch.Tensor, Ws: list, bs: list, steps: int) -> torch.Tensor:
    outs = []
    for W, b in zip(Ws, bs):
        lin = Z @ W + b.unsqueeze(0)
        z = lif_like_torch(lin)
        spk = rate_code_torch(z, steps)
        outs.append(aggregate_spikes_torch(spk))
    return torch.cat(outs, dim=1).to(torch.float32)

def pinv_ridge_torch(A: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    AtA = A.T @ A
    n = AtA.shape[0]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(AtA + lam * I, A.T)


#-------------------Updaters wrappers-------------------#
def extend_columns_with_backend(updater_name: str, Z: np.ndarray, Z_pinv: np.ndarray, H: np.ndarray, lam: float):
    """
    - 'greville' or 'updated_greville': try your updates modules and common function names.
    - 'direct': call your updates/direct.py DirectUpdater.add_columns on torch tensors.
    - fallback: compute pinv on full A=[Z|H] in numpy.
    """
    if updater_name == "direct":
        try:
            from updates.direct import DirectUpdater
            A_old = torch.from_numpy(Z)
            A_plus_old = torch.from_numpy(Z_pinv)  # not used by DirectUpdater
            H_new = torch.from_numpy(H)
            upd = DirectUpdater(lam=lam)
            A_aug, A_plus_new = upd.add_columns(A_old, A_plus_old, H_new)
            return A_aug.numpy(), A_plus_new.numpy()
        except Exception as e:
            # fallback to numpy
            A = np.concatenate([Z, H], axis=1)
            return A, pinv_ridge_np(A, lam)

    # greville / updated_greville path
    mod = None
    try:
        if updater_name == "greville":
            import updates.greville as mod
        elif updater_name == "updated_greville":
            import updates.updated_greville as mod
    except Exception:
        mod = None

    if mod is not None:
        for fname in ["extend_columns", "column_partition_pinv", "extend"]:
            fn = getattr(mod, fname, None)
            if callable(fn):
                try:
                    A_pinv = fn(Z_pinv, Z, H, ridge=lam)
                except TypeError:
                    A_pinv = fn(Z_pinv, Z, H)
                A = np.concatenate([Z, H], axis=1)
                return A, A_pinv

    # ultimate fallback
    A = np.concatenate([Z, H], axis=1)
    return A, pinv_ridge_np(A, lam)

# ---------------- train & eval (GPU-aware) ---------------- #
def train_and_eval(updater, npw, mw, seed=SEED):
    set_seed(seed)
    rng = np.random.RandomState(seed)

    # data
    X_train, y_train, X_test, y_test = load_mnist(flatten=True)
    Y_train = to_onehot(y_train, 10)

    if USE_GPU:
        # ---- GPU path ----
        Xtr_t = torch.from_numpy(X_train).to(DEVICE)
        Xte_t = torch.from_numpy(X_test).to(DEVICE)

        Ws_f_t, bs_f_t = init_feature_layer_torch(Xtr_t.shape[1], npw, NW, rng, DEVICE)
        Z_t = forward_feature_layer_torch(Xtr_t, Ws_f_t, bs_f_t, SPIKE_STEPS)
        Ws_h_t, bs_h_t = init_enh_layer_torch(Z_t.shape[1], MP, mw, rng, DEVICE)
        H_t = forward_enh_layer_torch(Z_t, Ws_h_t, bs_h_t, SPIKE_STEPS)

        mem_before = peak_mem_mb()
        with Timer() as t:
            if updater == "direct":
                # full GPU solve
                A_t = torch.cat([Z_t, H_t], dim=1)
                A_pinv_t = pinv_ridge_torch(A_t, lam=max(RIDGE, 1e-3))
                Ytr_t = torch.from_numpy(Y_train).to(DEVICE)
                W_out_t = A_pinv_t @ Ytr_t
            else:
                # greville / updated_greville on CPU (NumPy); forward done on GPU
                Z = Z_t.detach().cpu().numpy()
                H = H_t.detach().cpu().numpy()
                Z_pinv = pinv_ridge_np(Z, lam=max(RIDGE, 1e-3))
                A, A_pinv = extend_columns_with_backend(updater, Z, Z_pinv, H, lam=max(RIDGE, 1e-3))
                W_out = (A_pinv @ Y_train).astype(np.float32)
        train_time = t.elapsed
        mem_after = peak_mem_mb()
        peak_mem = max(mem_before, mem_after)

        # eval
        Zte_t = forward_feature_layer_torch(Xte_t, Ws_f_t, bs_f_t, SPIKE_STEPS)
        Hte_t = forward_enh_layer_torch(Zte_t, Ws_h_t, bs_h_t, SPIKE_STEPS)
        if updater == "direct":
            At_t = torch.cat([Zte_t, Hte_t], dim=1)
            logits_t = At_t @ W_out_t
            y_pred = logits_t.argmax(dim=1).detach().cpu().numpy()
        else:
            At = np.concatenate([Zte_t.detach().cpu().numpy(), Hte_t.detach().cpu().numpy()], axis=1).astype(np.float32)
            logits = At @ W_out
            y_pred = np.argmax(logits, axis=1)

        acc = float((y_pred == y_test).mean())
        cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
        fname = f"confmat_{updater}_np{npw}_mw{mw}.png"
        plot_confusionmatrix(cm, f"{updater}  np={npw}  mw={mw}", os.path.join(SAVE_DIR, fname))
        return acc, train_time, peak_mem

     # ---- CPU path (original) ----
    Ws_f, bs_f = init_feature_layer(in_dim=X_train.shape[1], np_per_win=npw, nw=NW, rng=rng)
    Z = forward_feature_layer(X_train, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)
    Ws_h, bs_h = init_enh_layer(feat_dim=Z.shape[1], mp_per_win=MP, mw=mw, rng=rng)
    H = forward_enh_layer(Z, Ws_h, bs_h, steps=SPIKE_STEPS, rng=rng)
    mem_before = peak_mem_mb()
    with Timer() as t:
        Z_pinv = pinv_ridge_np(Z, lam=max(RIDGE, 1e-3))
        A, A_pinv = extend_columns_with_backend(updater, Z, Z_pinv, H, lam=max(RIDGE, 1e-3))
        W_out = (A_pinv @ to_onehot(y_train, 10)).astype(np.float32)
    train_time = t.elapsed
    mem_after = peak_mem_mb()
    peak_mem = max(mem_before, mem_after)
    Zt = forward_feature_layer(X_test, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)
    Ht = forward_enh_layer(Zt, Ws_h, bs_h, steps=SPIKE_STEPS, rng=rng)
    At = np.concatenate([Zt, Ht], axis=1).astype(np.float32)
    logits = At @ W_out
    y_pred = np.argmax(logits, axis=1)
    acc = float((y_pred == y_test).mean())
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    fname = f"confmat_{updater}_np{npw}_mw{mw}.png"
    plot_confusionmatrix(cm, f"{updater}  np={npw}  mw={mw}", os.path.join(SAVE_DIR, fname))
    return acc, train_time, peak_mem


# # ========== Sanity checks 5 (to one training pass) =========== #
# def _assert_finite(name, arr):
#     if not np.isfinite(arr).all():
#         raise ValueError(f"{name} contains NaN/Inf. shape={getattr(arr, 'shape', None)}")

# def _quick_accuracy(logits: np.ndarray, y_true: np.ndarray) -> float:
#     y_pred = logits.argmax(axis=1)
#     return float((y_pred == y_true).mean())

# def sanity_one_training_pass(npw: int = 50, mw: int = 10, ridge: float = None, seed: int = 0):
#     """
#     Runs a single end-to-end SBLS training pass using pinv_ridge_np (no incremental updater).
#     """
#     if ridge is None:
#         ridge = max(RIDGE, 1e-3)  # small ridge for sanity to avoid singularities

#     ensure_dirs()
#     set_seed(seed)

#     # 1) Load data
#     X_tr, y_tr, X_te, y_te = load_mnist(flatten=True)
#     print(f"[loader] Train: {X_tr.shape}, {y_tr.shape} | Test: {X_te.shape}, {y_te.shape}")
#     assert X_tr.shape[1] == 28 * 28, "Flattened MNIST should be 784-D"
#     _assert_finite("X_tr", X_tr); _assert_finite("X_te", X_te)

#     # 2) Random generators
#     rng = np.random.RandomState(seed)

#     # 3) Feature layer → Z
#     Ws_f, bs_f = init_feature_layer(in_dim=X_tr.shape[1], np_per_win=npw, nw=NW, rng=rng)
#     with Timer() as t_feat:
#         Z = forward_feature_layer(X_tr, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)  # (N, npw*NW)
#     print(f"[feature] Z: {Z.shape}, time={t_feat.elapsed:.3f}s")
#     assert Z.shape == (X_tr.shape[0], npw * NW)
#     _assert_finite("Z", Z)

#     # 4) Enhancement layer → H
#     Ws_h, bs_h = init_enh_layer(feat_dim=Z.shape[1], mp_per_win=MP, mw=mw, rng=rng)
#     with Timer() as t_enh:
#         H = forward_enh_layer(Z, Ws_h, bs_h, steps=SPIKE_STEPS, rng=rng)         # (N, MP*mw)
#     print(f"[enh]     H: {H.shape}, time={t_enh.elapsed:.3f}s")
#     assert H.shape == (X_tr.shape[0], MP * mw)
#     _assert_finite("H", H)

#     # 5) Concatenate A = [Z | H]
#     A = np.concatenate([Z, H], axis=1).astype(np.float32)
#     print(f"[concat]  A: {A.shape}")
#     _assert_finite("A", A)

#     # 6) Solve W = A^+ Y  (training)
#     Y_tr = to_onehot(y_tr, 10)
#     mem0 = peak_mem_mb()
#     with Timer() as t_solve:
#         A_pinv = pinv_ridge_np(A, lam=ridge)             # (d, N)
#         W_out  = (A_pinv @ Y_tr).astype(np.float32)      # (d, 10)
#     mem1 = peak_mem_mb()
#     print(f"[solve]   A^+: {A_pinv.shape}, W: {W_out.shape}, time={t_solve.elapsed:.3f}s, mem≈{max(mem0, mem1):.2f} MB")
#     assert A_pinv.shape == (A.shape[1], A.shape[0])
#     assert W_out.shape  == (A.shape[1], 10)
#     _assert_finite("A_pinv", A_pinv); _assert_finite("W_out", W_out)

#     # 7) Quick eval on test set
#     Zt = forward_feature_layer(X_te, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)
#     Ht = forward_enh_layer(Zt, Ws_h, bs_h, steps=SPIKE_STEPS, rng=rng)
#     At = np.concatenate([Zt, Ht], axis=1).astype(np.float32)
#     logits = At @ W_out
#     acc = _quick_accuracy(logits, y_te)
#     print(f"[eval]    Test acc: {acc*100:.2f}%")

#     # 8) Save confusion matrix
#     cm = confusion_matrix(y_te, logits.argmax(axis=1), labels=list(range(10)))
#     cm_path = os.path.join(SAVE_DIR, f"confmat_sanity_onepass_np{npw}_mw{mw}.png")
#     plot_confusionmatrix(cm, f"Sanity one-pass np={npw} mw={mw}", cm_path)
#     print(f"[confmat] saved -> {cm_path}")

#     print("\nSanity (one training pass) OK ✅")
#     return dict(acc=acc, train_time=t_solve.elapsed, peak_mem=max(mem0, mem1), A_shape=A.shape)

# # Run when this file is executed directly
# if __name__ == "__main__":
#     _ = sanity_one_training_pass(npw=50, mw=10, seed=0)

#-------------------main-------------------#
# ===================== E1 ===================== #
def experiment_1():
    ensure_dirs()
    print("== E1: Capacity Sweep (np in {50,100}, mw in {10,20}) ==")
    results = []
    for updater in UPDATERS:
        print(f"\n-- Updater: {updater} --")
        for npw in NPS:
            for mw in MWS:
                acc, tsec, pmem = train_and_eval(updater, npw, mw, seed=SEED)
                print(f"np={npw:3d}  mw={mw:3d}  |  acc={acc*100:5.2f}%  time={tsec:6.2f}s  peak_mem≈{pmem:7.2f} MB")
                results.append(dict(updater=updater, np=npw, mw=mw, acc=acc, time=tsec, peak_mem_mb=pmem))
    with open("e1_minimal_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

# ===================== E2 (updated_greville only) ===================== #
def experiment_2(
    train_sizes=(5_000, 10_000, 30_000, 60_000),
    seeds=tuple(range(10)),      # 10 runs (iterations)
    npw=100, mw=20
):
    """
    E2: Influence of training set size using only updated_greville.
    Outputs:
      - Console table (k vs mean accuracy % over seeds)
      - CSV: results/e2_table_accuracy_updated.csv
      - Figure (dots+line): figures/e2_acc_vs_k_updated.png
    """
    ensure_dirs()
    print("\n== E2: Training Set Size (updated_greville) ==")
    X_tr_full, y_tr_full, X_te, y_te = load_mnist(flatten=True)

    # stratified subset for balance & reproducibility
    def _k_indices_stratified(y: np.ndarray, k: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        per_class = {c: np.where(y == c)[0] for c in range(10)}
        for c in per_class: rng.shuffle(per_class[c])
        base, rem = k // 10, k % 10
        counts = [base + (1 if i < rem else 0) for i in range(10)]
        idx = np.concatenate([per_class[c][:counts[c]] for c in range(10)], axis=0)
        rng.shuffle(idx)
        return idx

    rows = []
    for k in train_sizes:
        accs = []
        for seed in seeds:
            set_seed(seed)
            rng = np.random.RandomState(seed)
            idx = _k_indices_stratified(y_tr_full, k, seed)
            X_tr = X_tr_full[idx]; y_tr = y_tr_full[idx]
            Y_tr = to_onehot(y_tr, 10)

            if USE_GPU:
                Xtr_t = torch.from_numpy(X_tr).to(DEVICE)
                Xte_t = torch.from_numpy(X_te).to(DEVICE)

                Ws_f_t, bs_f_t = init_feature_layer_torch(Xtr_t.shape[1], npw, NW, rng, DEVICE)
                Z_t = forward_feature_layer_torch(Xtr_t, Ws_f_t, bs_f_t, SPIKE_STEPS)
                Ws_h_t, bs_h_t = init_enh_layer_torch(Z_t.shape[1], MP, mw, rng, DEVICE)
                H_t = forward_enh_layer_torch(Z_t, Ws_h_t, bs_h_t, SPIKE_STEPS)

                # updater on CPU (NumPy)
                Z = Z_t.detach().cpu().numpy()
                H = H_t.detach().cpu().numpy()
                lam = max(RIDGE, 1e-3)
                Z_pinv = pinv_ridge_np(Z, lam=lam)
                A, A_pinv = extend_columns_with_backend("updated_greville", Z, Z_pinv, H, lam=lam)
                W_out = (A_pinv @ Y_tr).astype(np.float32)

                # eval
                Zte_t = forward_feature_layer_torch(Xte_t, Ws_f_t, bs_f_t, SPIKE_STEPS)
                Hte_t = forward_enh_layer_torch(Zte_t, Ws_h_t, bs_h_t, SPIKE_STEPS)
                At = np.concatenate([Zte_t.detach().cpu().numpy(), Hte_t.detach().cpu().numpy()], axis=1).astype(np.float32)
                y_pred = np.argmax(At @ W_out, axis=1)

            else:
                # CPU path
                Ws_f, bs_f = init_feature_layer(in_dim=X_tr.shape[1], np_per_win=npw, nw=NW, rng=rng)
                Z = forward_feature_layer(X_tr, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)
                Ws_h, bs_h = init_enh_layer(feat_dim=Z.shape[1], mp_per_win=MP, mw=mw, rng=rng)
                H = forward_enh_layer(Z, Ws_h, bs_h, steps=SPIKE_STEPS, rng=rng)

                lam = max(RIDGE, 1e-3)
                Z_pinv = pinv_ridge_np(Z, lam=lam)
                A, A_pinv = extend_columns_with_backend("updated_greville", Z, Z_pinv, H, lam=lam)
                W_out = (A_pinv @ Y_tr).astype(np.float32)

                Zt = forward_feature_layer(X_te, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)
                Ht = forward_enh_layer(Zt, Ws_h, bs_h, steps=SPIKE_STEPS, rng=rng)
                At = np.concatenate([Zt, Ht], axis=1).astype(np.float32)
                y_pred = np.argmax(At @ W_out, axis=1)

            acc = float((y_pred == y_te).mean())
            accs.append(acc)
            rows.append(dict(k=k, seed=seed, acc=acc))

        print(f"k={k:6d} | acc={np.mean(accs)*100:5.2f}% ± {np.std(accs)*100:4.2f}")

    df = pd.DataFrame(rows)
    table = (
        df.groupby("k")["acc"]
          .mean()
          .mul(100)
          .round(2)
          .reset_index(name="accuracy_pct")
          .sort_values("k")
    )
    table.to_csv("results/e2_table_accuracy_updated.csv", index=False)
    print("Saved: results/e2_table_accuracy_updated.csv")

    x = table["k"].values
    y = table["accuracy_pct"].values
    plt.figure(figsize=(7,4), dpi=140)
    plt.plot(x, y, marker="o", linestyle="-")   # dots + line
    plt.xlabel("Training set size (k samples)")
    plt.ylabel("Accuracy (%)")
    plt.title("E2 (updated_greville): Accuracy vs Training Size")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("figures/e2_acc_vs_k_updated.png")
    plt.close()
    print("Saved: figures/e2_acc_vs_k_updated.png")

# ===================== E3 ===================== #
def experiment_3(
    seeds=tuple(range(10)),    # 10 runs
    npw=100,                   # fixed feature size
    mp=MP,                     # 100 per window
    base_mw=20,
    target_mw=30,
    schedules=dict(
        V1=[2,2,2,2,2],       # 20→22→24→26→28→30
        V2=[5,5],             # 20→25→30
        V3=[10],              # 20→30
    ),
    ridge=None,                # if None -> use max(RIDGE, 1e-3)
):
    """
    E3: Incremental column updates by adding enhancement windows.
    Compares 'direct', 'greville', 'updated_greville' across schedules.
    Outputs: rows CSV + per-schedule accuracy curves.
    """
    ensure_dirs()
    lam = max(RIDGE, 1e-3) if ridge is None else ridge

    X_tr_full, y_tr_full, X_te, y_te = load_mnist(flatten=True)
    Y_tr_full = to_onehot(y_tr_full, 10)

    def make_blocks(feat_dim, inc_list, rng):
        blocks = []
        for dmw in inc_list:
            if USE_GPU:
                # we only need weights (they'll be used for torch or numpy forward)
                Ws_blk, bs_blk = init_enh_layer(feat_dim=feat_dim, mp_per_win=mp, mw=dmw, rng=rng)
            else:
                Ws_blk, bs_blk = init_enh_layer(feat_dim=feat_dim, mp_per_win=mp, mw=dmw, rng=rng)
            blocks.append((Ws_blk, bs_blk))
        return blocks

    rows = []  # dict(method, schedule, total_mw, seed, acc, step_time)

    print("\n== E3: Incremental Enhancement Nodes (Column Updates) ==")
    for schedule_name, inc_list in schedules.items():
        print(f"\n-- Schedule {schedule_name}: base={base_mw}, increments={inc_list}, target={target_mw} --")

        for seed in seeds:
            set_seed(seed)
            rng = np.random.RandomState(seed)

            # Features (Z) once per seed
            if USE_GPU:
                Xtr_t = torch.from_numpy(X_tr_full).to(DEVICE)
                Xte_t = torch.from_numpy(X_te).to(DEVICE)
                Ws_f_t, bs_f_t = init_feature_layer_torch(Xtr_t.shape[1], npw, NW, rng, DEVICE)
                Z_t = forward_feature_layer_torch(Xtr_t, Ws_f_t, bs_f_t, SPIKE_STEPS)
                Z = Z_t.detach().cpu().numpy()
            else:
                Ws_f, bs_f = init_feature_layer(in_dim=X_tr_full.shape[1], np_per_win=npw, nw=NW, rng=rng)
                Z = forward_feature_layer(X_tr_full, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)

            # Base enhancement windows
            Ws_base, bs_base = init_enh_layer(feat_dim=Z.shape[1], mp_per_win=mp, mw=base_mw, rng=rng)

            # Pre-generate increment blocks
            blocks = make_blocks(feat_dim=Z.shape[1], inc_list=inc_list, rng=rng)

            for method in ("direct", "greville", "updated_greville"):
                Ws_cum = list(Ws_base)
                bs_cum = list(bs_base)
                Z_pinv = pinv_ridge_np(Z, lam=lam)  # constant per seed/method

                total_mw_running = base_mw
                for (Ws_blk, bs_blk) in blocks:
                    Ws_cum += Ws_blk
                    bs_cum += bs_blk
                    total_mw_running += len(Ws_blk)

                    with Timer() as t_step:
                        # Build cumulative H
                        if USE_GPU:
                            # forward H on GPU for speed, then back to numpy if needed
                            Z_t = torch.from_numpy(Z).to(DEVICE)
                            # re-create torch weights for H_cum from numpy weights:
                            # (fast path: compute H via numpy to keep it simple & consistent)
                            H_cum = forward_enh_layer(Z, Ws_cum, bs_cum, steps=SPIKE_STEPS, rng=rng)
                        else:
                            H_cum = forward_enh_layer(Z, Ws_cum, bs_cum, steps=SPIKE_STEPS, rng=rng)

                        # Update A^+ and solve
                        A, A_pinv = extend_columns_with_backend(method, Z, Z_pinv, H_cum, lam=lam)
                        W_out = (A_pinv @ Y_tr_full).astype(np.float32)
                    step_time = t_step.elapsed

                    # Evaluate on test
                    if USE_GPU:
                        # compute Zt/Ht with torch if we had torch feature weights
                        Zt_np = None
                        if 'Ws_f_t' in locals():
                            Zt_t = forward_feature_layer_torch(Xte_t, Ws_f_t, bs_f_t, SPIKE_STEPS)
                            # build torch enh weights from numpy Ws_cum/bs_cum? simplest: reuse numpy path for eval too
                            Zt_np = Zt_t.detach().cpu().numpy()
                            Ht = forward_enh_layer(Zt_np, Ws_cum, bs_cum, steps=SPIKE_STEPS, rng=rng)
                            At = np.concatenate([Zt_np, Ht], axis=1).astype(np.float32)
                        else:
                            # fallback to numpy path
                            if 'Ws_f' in locals():
                                Zt = forward_feature_layer(X_te, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)
                                Ht = forward_enh_layer(Zt, Ws_cum, bs_cum, steps=SPIKE_STEPS, rng=rng)
                                At = np.concatenate([Zt, Ht], axis=1).astype(np.float32)
                            else:
                                raise RuntimeError("Missing feature weights for eval.")
                    else:
                        Zt = forward_feature_layer(X_te, Ws_f, bs_f, steps=SPIKE_STEPS, rng=rng)
                        Ht = forward_enh_layer(Zt, Ws_cum, bs_cum, steps=SPIKE_STEPS, rng=rng)
                        At = np.concatenate([Zt, Ht], axis=1).astype(np.float32)

                    y_pred = np.argmax(At @ W_out, axis=1)
                    acc = float((y_pred == y_te).mean())

                    rows.append(dict(
                        schedule=schedule_name,
                        method=method,
                        seed=seed,
                        total_mw=total_mw_running,
                        acc=acc,
                        step_time=step_time
                    ))

        # summary at final capacity
        df_sched = pd.DataFrame([r for r in rows if r["schedule"] == schedule_name])
        df_final = df_sched[df_sched["total_mw"] == target_mw]
        print("  Final (mw=30) accuracy mean ± std:")
        for m in ("direct", "greville", "updated_greville"):
            sub = df_final[df_final["method"] == m]["acc"]
            if not sub.empty:
                print(f"    {m:16s} {sub.mean()*100:5.2f}% ± {sub.std()*100:4.2f}%")

    # Aggregate & Plot
    df = pd.DataFrame(rows)
    df["acc_pct"] = df["acc"] * 100
    df.sort_values(["schedule","method","total_mw","seed"]).to_csv("results/e3_rows.csv", index=False)
    print("Saved: results/e3_rows.csv")

    for schedule_name in schedules.keys():
        d = df[df["schedule"] == schedule_name]
        if d.empty: 
            continue
        plt.figure(figsize=(7,4), dpi=140)
        for m in ("direct","greville","updated_greville"):
            sub = (d[d["method"] == m]
                   .groupby("total_mw")["acc_pct"].mean()
                   .reset_index()
                   .sort_values("total_mw"))
            if sub.empty: 
                continue
            plt.plot(sub["total_mw"], sub["acc_pct"], marker="o", label=m)
        plt.xlabel("Total enhancement windows (mw)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"E3: Accuracy vs mw — {schedule_name}")
        plt.grid(alpha=0.2); plt.legend(); plt.tight_layout()
        plt.savefig(f"figures/e3_acc_vs_mw_{schedule_name}.png"); plt.close()
        print(f"Saved: figures/e3_acc_vs_mw_{schedule_name}.png")

    df_final = df[df["total_mw"] == target_mw]
    if not df_final.empty:
        print("\nE3 — Final step time (mean ± std) [per-step at final increment]")
        for schedule_name in schedules.keys():
            for m in ("direct","greville","updated_greville"):
                sub = df_final[(df_final["schedule"]==schedule_name) & (df_final["method"]==m)]["step_time"]
                if not sub.empty:
                    print(f"{schedule_name:>3s} | {m:16s} time={sub.mean():.2f}s ± {sub.std():.2f}s")

    print("\n[E3] Done.")

# ---------------- Runner ---------------- #
if __name__ == "__main__":
    # experiment_1()
    experiment_2()
    # experiment_3()
    # print("Ready. Uncomment an experiment() at the bottom to run.")