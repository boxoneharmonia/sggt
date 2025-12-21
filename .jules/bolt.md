## 2024-05-23 - Shape Mismatch in PoseLoss
**Learning:** The original `PoseLoss.voting_loss` implementation seemingly adds `(H, W, 2)` grid directly to `(B, N, 2, H, W)` offset tensor. This dimension mismatch suggests either `grid_norm` is expected to be permuted or the code relies on specific shape coincidences (which it doesn't here).
**Action:** When optimizing, verify input shapes carefully. In `PoseLossOptimized`, I explicitly permuted `grid_norm` to `(2, H, W)` to ensure correct broadcasting.

## 2024-05-23 - Broadcasting vs Repeat Interleave
**Learning:** Replacing `repeat_interleave` with broadcasting in `voting_loss` reduced memory footprint by avoiding large intermediate tensors like `(B*N, HW, 3)` and provided a modest speedup (1.06x on CPU, likely more on GPU with larger batches).
**Action:** Always prefer broadcasting for element-wise operations or matrix multiplications where one operand is shared across a dimension.
