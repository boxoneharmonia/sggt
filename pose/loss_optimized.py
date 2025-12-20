
import torch
import torch.nn.functional as F
from einops import rearrange
from .grid_cache import GridCache

def voting_loss_optimized(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=False):
    """
    Optimized implementation of voting_loss using broadcasting instead of repeat_interleave
    where possible, and fixing grid permutation issue.
    """
    B, C, H_feat, W_feat = pvmap.shape
    num_corners = C // 2
    device = pvmap.device

    # Pool the mask
    mask_expanded = F.max_pool2d(mask_gt, kernel_size=3, stride=1, padding=1)

    # Get grid and offsets
    # FIXED: permute grid to match (2, H, W) to align with (B, N, 2, H, W)
    grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device).permute(2, 0, 1) # (2, H, W)
    corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
    corners_2d_norm = grid_norm + corners_offset # (B, N_corners, 2, H_feat, W_feat)

    if aggregate_first:
        mask_for_agg = mask_expanded.view(B, 1, 1, H_feat, W_feat)
        weighted_sum = (corners_2d_norm * mask_for_agg).sum(dim=(-1, -2))
        total_weight = mask_for_agg.sum(dim=(-1, -2)).clamp(min=1.0)
        corners_2d_agg_norm = weighted_sum / total_weight
        corners_2d_scaled = corners_2d_agg_norm * scale_tensor.view(1, 1, 2)
        corners_homo = torch.cat([corners_2d_scaled, torch.ones_like(corners_2d_scaled[:, :, :1])], dim=-1)
        K_inv = torch.linalg.inv(cam_K)
        rays = corners_homo @ K_inv.transpose(-1, -2)
        rays_norm = F.normalize(rays, p=2, dim=-1)
        dot_prod = (pts3d_gt * rays_norm).sum(dim=-1, keepdim=True)
        proj_point = dot_prod * rays_norm
        dist_3d = torch.norm(pts3d_gt - proj_point, p=2, dim=-1)
        return dist_3d.mean()
    else:
        # Optimized Logic: Use broadcasting to avoid large memory copies from repeat_interleave

        # 1. Expand mask without repeat_interleave
        # mask_expanded: (B, 1, H, W)
        voter_mask = (mask_expanded > 0.5).expand(-1, num_corners, -1, -1)
        voter_mask_flat = voter_mask.reshape(B, num_corners, H_feat * W_feat)
        num_voters_per_image = voter_mask_flat.sum(dim=-1).clamp(min=1.0) # (B, N)

        # 2. Scale corners
        scale_tensor = scale_tensor.view(1, 1, 2, 1, 1)
        corners_2d_scaled = corners_2d_norm * scale_tensor

        # 3. Homogeneous coordinates: (B, N, 3, H, W)
        corners_homo = torch.cat([corners_2d_scaled, torch.ones_like(corners_2d_scaled[:, :, :1])], dim=2)

        # 4. Rearrange to (B, N, H*W, 3) for efficient batch matmul
        corners_homo_flat = corners_homo.permute(0, 1, 3, 4, 2).reshape(B, num_corners, H_feat * W_feat, 3)

        # 5. Inverse K
        K_inv = torch.linalg.inv(cam_K) # (B, 3, 3)
        # Broadcast multiply: (B, N, HW, 3) @ (B, 1, 3, 3)^T
        K_inv_bcast = K_inv.unsqueeze(1) # (B, 1, 3, 3)

        rays = corners_homo_flat @ K_inv_bcast.transpose(-1, -2) # (B, N, HW, 3)
        rays_norm = F.normalize(rays, p=2, dim=-1)

        # 6. Distance calculation
        # pts3d_gt: (B, N, 3) -> (B, N, 1, 3)
        pts3d_gt_expanded = pts3d_gt.unsqueeze(2)

        # dot_prod: (B, N, HW, 1)
        dot_prod = (pts3d_gt_expanded * rays_norm).sum(dim=-1, keepdim=True)
        proj_point = dot_prod * rays_norm

        # dist_3d: (B, N, HW)
        dist_3d = torch.norm(pts3d_gt_expanded - proj_point, p=2, dim=-1)

        loss_per_corner = (dist_3d * voter_mask_flat).sum(dim=-1) / num_voters_per_image # (B, N)
        return loss_per_corner.mean()
