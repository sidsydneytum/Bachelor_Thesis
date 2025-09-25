from updates.updated_greville import UpdatedGrevilleUpdater
import torch

# Base matrix and its pseudoinverse
A = torch.randn(5, 3)
A_plus = torch.linalg.pinv(A)

up = UpdatedGrevilleUpdater(eps=1e-10)

# ---- Add columns (H_new: 5×2) ----
H_new = torch.randn(5, 2)
A_aug, A_plus_new = up.add_columns(A, A_plus, H_new)

# Compare to direct pinv
pinv_direct = torch.linalg.pinv(torch.cat([A, H_new], dim=1))
print("Fro diff cols:", torch.norm(A_plus_new - pinv_direct).item())

# ---- Add rows (A_x_new: 4×3) ----
A_x_new = torch.randn(4, 3)
A_aug_r, A_plus_r = up.add_rows(A, A_plus, A_x_new)

pinv_direct_r = torch.linalg.pinv(torch.cat([A, A_x_new], dim=0))
print("Fro diff rows:", torch.norm(A_plus_r - pinv_direct_r).item())
