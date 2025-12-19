
import torch
import torch.nn.functional as F
from einops import rearrange
import time
import math
from pose.loss_optimized import voting_loss_optimized

# Mock GridCache for original function to avoid dependency issues or changes in global state
class GridCache:
    _cache = {}
    @staticmethod
    def get_mesh_grid(height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        key = ("mesh_01", height, width, device, dtype)
        if key not in GridCache._cache:
            y_range = torch.linspace(0, 1, height, device=device, dtype=dtype)
            x_range = torch.linspace(0, 1, width, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1)
            GridCache._cache[key] = grid
        return GridCache._cache[key]

def voting_loss_original(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=False):
    B, C, H_feat, W_feat = pvmap.shape
    num_corners = C // 2
    device = pvmap.device

    mask_expanded = F.max_pool2d(mask_gt, kernel_size=3, stride=1, padding=1)
    grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H_feat, W_feat, 2)

    # Fix broadcasting: (H, W, 2) -> (2, H, W) -> (1, 1, 2, H, W)
    # Note: We apply the fix here to make the "original" code run for comparison.
    # The actual original code in net.py was likely buggy or relying on different GridCache behavior.
    grid_norm_perm = grid_norm.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
    corners_2d_norm = grid_norm_perm + corners_offset # (B, N_corners, 2, H_feat, W_feat)

    if aggregate_first:
        return 0.0 # Not testing this branch
    else:
        voter_mask = (mask_expanded > 0.5).repeat_interleave(num_corners, dim=0)  # (B*N_corners, 1, H_feat, W_feat)
        voter_mask = voter_mask.view(B * num_corners, H_feat * W_feat)
        num_voters_per_image = voter_mask.sum(dim=-1).clamp(min=1.0) # (B*N_corners, )
        scale_tensor = scale_tensor.view(1, 1, 2, 1, 1)
        corners_2d_scaled = corners_2d_norm * scale_tensor
        corners_homo = torch.cat([corners_2d_scaled, torch.ones_like(corners_2d_scaled[:, :, :1])], dim=2) # (B, N, 3, H, W)
        corners_homo = rearrange(corners_homo, 'b n c h w -> (b n) (h w) c') # (B*N_corners, H_feat*W_feat, 3)
        K_inv = torch.linalg.inv(cam_K) # (B, 3, 3)
        K_inv_expanded = K_inv.repeat_interleave(num_corners, dim=0) # (B*N_corners, 3, 3)
        rays = corners_homo @ K_inv_expanded.transpose(-1, -2) # (B*N_corners, H_feat*W_feat, 3)
        rays_norm = F.normalize(rays, p=2, dim=-1)

        pts3d_gt_expanded = pts3d_gt.view(B * num_corners, 1, 3).expand(-1, H_feat * W_feat, -1)
        dot_prod = (pts3d_gt_expanded * rays_norm).sum(dim=-1, keepdim=True)
        proj_point = dot_prod * rays_norm
        dist_3d = torch.norm(pts3d_gt_expanded - proj_point, p=2, dim=-1) # (B*N_corners, H_feat*W_feat)
        loss_per_corner = (dist_3d * voter_mask).sum(dim=-1) / num_voters_per_image # (B*N_corners, )
        return loss_per_corner.mean()

def run_test():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")

    B = 4
    num_corners = 8
    H_feat = 64
    W_feat = 64
    C = num_corners * 2

    pvmap = torch.randn(B, C, H_feat, W_feat, device=device)
    mask_gt = (torch.randn(B, 1, H_feat, W_feat, device=device) > 0).float()

    pts3d_gt = torch.randn(B, num_corners, 3, device=device)
    cam_K = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    scale_tensor = torch.tensor([W_feat, H_feat], device=device).float()

    # Warmup
    for _ in range(10):
        _ = voting_loss_original(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
        _ = voting_loss_optimized(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)

    # Check correctness
    loss_orig = voting_loss_original(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    loss_opt = voting_loss_optimized(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)

    print(f"Original Loss: {loss_orig.item()}")
    print(f"Optimized Loss: {loss_opt.item()}")

    if not torch.isclose(loss_orig, loss_opt, atol=1e-5):
        print("Mismatch!")
        diff = (loss_orig - loss_opt).abs()
        print(f"Diff: {diff.item()}")
        print(f"Max Diff: {diff.max().item()}")
    else:
        print("Matches!")

    # Benchmark
    iters = 100
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = voting_loss_original(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    orig_time = (time.time() - start) / iters
    print(f"Original Time: {orig_time*1000:.3f} ms")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = voting_loss_optimized(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    opt_time = (time.time() - start) / iters
    print(f"Optimized Time: {opt_time*1000:.3f} ms")

    print(f"Speedup: {orig_time / opt_time:.2f}x")

if __name__ == "__main__":
    run_test()
