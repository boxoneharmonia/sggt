import torch
import torch.nn.functional as F
from .grid_cache import GridCache

def voting_loss_optimized(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=False):
    """
    Optimized version of voting_loss that avoids repeat_interleave and heavy reshapes.
    """
    B, C, H_feat, W_feat = pvmap.shape
    num_corners = C // 2
    device = pvmap.device

    # Pooling mask
    mask_expanded = F.max_pool2d(mask_gt, kernel_size=3, stride=1, padding=1)

    # Grid generation
    grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H, W, 2)

    # Fix broadcasting: (H, W, 2) -> (1, 1, 2, H, W)
    # The original code expected broadcasting that likely failed or relied on implicit behavior.
    # We explicitly permute to match (B, N, 2, H, W)
    grid_norm_perm = grid_norm.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
    corners_2d_norm = grid_norm_perm + corners_offset

    if aggregate_first:
        # Keep original logic for aggregate_first=True, but fix grid_norm
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
        # Optimized Logic for aggregate_first=False

        # 1. Mask processing
        # mask_expanded: (B, 1, H, W)
        mask_flat = mask_expanded.flatten(2) # (B, 1, HW)
        voter_mask = (mask_flat > 0.5) # (B, 1, HW)
        # Sum over HW. num_voters: (B, 1)
        num_voters = voter_mask.sum(dim=-1).clamp(min=1.0)

        # 2. Corners scaling
        scale_tensor = scale_tensor.view(1, 1, 2, 1, 1)
        corners_2d_scaled = corners_2d_norm * scale_tensor # (B, N, 2, H, W)
        corners_homo = torch.cat([corners_2d_scaled, torch.ones_like(corners_2d_scaled[:, :, :1])], dim=2) # (B, N, 3, H, W)

        # Flatten HW: (B, N, 3, H, W) -> (B, N, 3, HW) -> transpose to (B, N, HW, 3)
        corners_homo_flat = corners_homo.flatten(3).transpose(2, 3) # (B, N, HW, 3)

        # 3. K_inv and rays
        K_inv = torch.linalg.inv(cam_K) # (B, 3, 3)
        # Broadcasting K_inv: (B, 1, 3, 3)
        # rays = corners @ K_inv.T

        rays = corners_homo_flat @ K_inv.unsqueeze(1).transpose(-1, -2) # (B, N, HW, 3)
        rays_norm = F.normalize(rays, p=2, dim=-1)

        # 4. 3D distance
        # pts3d_gt: (B, N, 3)
        pts3d_gt_expanded = pts3d_gt.unsqueeze(2) # (B, N, 1, 3)

        dot_prod = (pts3d_gt_expanded * rays_norm).sum(dim=-1, keepdim=True)
        proj_point = dot_prod * rays_norm
        dist_3d = torch.norm(pts3d_gt_expanded - proj_point, p=2, dim=-1) # (B, N, HW)

        # 5. Loss
        # voter_mask: (B, 1, HW) -> broadcasts to (B, N, HW)
        # num_voters: (B, 1) -> broadcasts to (B, N)
        loss_per_corner = (dist_3d * voter_mask).sum(dim=-1) / num_voters # (B, N)
        return loss_per_corner.mean()
