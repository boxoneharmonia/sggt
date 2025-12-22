
import torch
import torch.nn.functional as F
import time
import sys
import os

# Add parent directory to path to import pose
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose.grid_cache import GridCache

def original_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True):
    B, C, H_feat, W_feat = pvmap.shape
    num_corners = C // 2
    device = pvmap.device

    grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H_feat, W_feat, 2)
    grid_norm = grid_norm.permute(2, 0, 1)
    corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
    corners_2d_norm = grid_norm + corners_offset # (B, N_corners, 2, H_feat, W_feat)

    fx = cam_K[:, 0, 0].view(B, 1, 1)
    fy = cam_K[:, 1, 1].view(B, 1, 1)
    cx = cam_K[:, 0, 2].view(B, 1, 1)
    cy = cam_K[:, 1, 2].view(B, 1, 1)

    if aggregate_first:
        mask_for_agg = mask_gt.view(B, 1, 1, H_feat, W_feat)
        weighted_sum = (corners_2d_norm * mask_for_agg).sum(dim=(-1, -2))
        total_weight = mask_for_agg.sum(dim=(-1, -2)).clamp(min=1.0)
        corners_2d_agg_norm = weighted_sum / total_weight
        corners_2d_scaled = corners_2d_agg_norm * scale_tensor.view(1, 1, 2)
        u = corners_2d_scaled[..., 0]
        v = corners_2d_scaled[..., 1]
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        ray_z = torch.ones_like(ray_x)

        rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)
        rays_norm = F.normalize(rays, p=2, dim=-1)
        dot_prod = (pts3d_gt * rays_norm).sum(dim=-1, keepdim=True)
        proj_point = dot_prod * rays_norm
        dist_3d = torch.norm(pts3d_gt - proj_point, p=2, dim=-1)
        return dist_3d.mean()
    else:
        return torch.tensor(0.0)

def optimized_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True):
    B, C, H_feat, W_feat = pvmap.shape
    num_corners = C // 2
    device = pvmap.device

    grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H_feat, W_feat, 2)
    grid_norm = grid_norm.permute(2, 0, 1)
    corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
    corners_2d_norm = grid_norm + corners_offset # (B, N_corners, 2, H_feat, W_feat)

    fx = cam_K[:, 0, 0].view(B, 1, 1)
    fy = cam_K[:, 1, 1].view(B, 1, 1)
    cx = cam_K[:, 0, 2].view(B, 1, 1)
    cy = cam_K[:, 1, 2].view(B, 1, 1)

    if aggregate_first:
        mask_for_agg = mask_gt.view(B, 1, 1, H_feat, W_feat)
        weighted_sum = (corners_2d_norm * mask_for_agg).sum(dim=(-1, -2))
        total_weight = mask_for_agg.sum(dim=(-1, -2)).clamp(min=1.0)
        corners_2d_agg_norm = weighted_sum / total_weight
        corners_2d_scaled = corners_2d_agg_norm * scale_tensor.view(1, 1, 2)

        u = corners_2d_scaled[..., 0]
        v = corners_2d_scaled[..., 1]

        # Element-wise operations for rays
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy

        # Avoid stack and explicit normalize
        # norm_sq = x^2 + y^2 + 1
        norm_sq = ray_x.square() + ray_y.square() + 1.0
        inv_norm = torch.rsqrt(norm_sq) # 1 / sqrt(x^2 + y^2 + 1)

        # P dot D_norm = (Px * rx + Py * ry + Pz * 1) * inv_norm
        Px = pts3d_gt[..., 0]
        Py = pts3d_gt[..., 1]
        Pz = pts3d_gt[..., 2]

        dot_prod = (Px * ray_x + Py * ray_y + Pz) * inv_norm

        # ||P||^2
        P_sq = pts3d_gt.square().sum(dim=-1)

        # dist^2 = ||P||^2 - (P dot D)^2
        dist_sq = P_sq - dot_prod.square()

        # numerical stability
        dist_3d = dist_sq.clamp(min=1e-8).sqrt()

        return dist_3d.mean()
    else:
        return torch.tensor(0.0)

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on {device}")

    B = 32
    num_corners = 8
    C = num_corners * 2
    H_feat = 64
    W_feat = 64

    pvmap = torch.randn(B, C, H_feat, W_feat, device=device)
    mask_gt = torch.rand(B, 1, H_feat, W_feat, device=device)
    pts3d_gt = torch.randn(B, num_corners, 3, device=device)
    cam_K = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    cam_K[:, 0, 0] = 500
    cam_K[:, 1, 1] = 500
    cam_K[:, 0, 2] = 32
    cam_K[:, 1, 2] = 32
    scale_tensor = torch.tensor((W_feat*4-1., H_feat*4-1.), device=device).view(1, 1, 2)

    # Warmup
    for _ in range(100):
        original_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
        optimized_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)

    torch.cuda.synchronize() if device.type == 'cuda' else None

    start_time = time.time()
    for _ in range(1000):
        loss = original_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    orig_time = time.time() - start_time

    start_time = time.time()
    for _ in range(1000):
        loss_opt = optimized_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    opt_time = time.time() - start_time

    print(f"Original time: {orig_time:.4f}s")
    print(f"Optimized time: {opt_time:.4f}s")
    print(f"Speedup: {orig_time / opt_time:.2f}x")

    # Correctness check
    loss_orig = original_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    loss_opt = optimized_voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)

    print(f"Original Loss: {loss_orig.item()}")
    print(f"Optimized Loss: {loss_opt.item()}")
    print(f"Difference: {abs(loss_orig.item() - loss_opt.item())}")

    assert torch.allclose(loss_orig, loss_opt, atol=1e-5), "Losses do not match!"

if __name__ == "__main__":
    run_benchmark()
