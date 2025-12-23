import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import PoseLoss, GridCache

class PoseLossOptimized(PoseLoss):
    """
    Optimized version of PoseLoss.
    Optimizes voting_loss by using matrix multiplication instead of broadcasting
    large tensors, significantly reducing memory usage and improving speed.
    """
    def voting_loss(self, pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True):
        B, C, H_feat, W_feat = pvmap.shape
        num_corners = C // 2
        device = pvmap.device

        fx = cam_K[:, 0, 0].view(B, 1, 1)
        fy = cam_K[:, 1, 1].view(B, 1, 1)
        cx = cam_K[:, 0, 2].view(B, 1, 1)
        cy = cam_K[:, 1, 2].view(B, 1, 1)

        grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H, W, 2)
        grid_norm = grid_norm.permute(2, 0, 1) # (2, H, W)

        # Original logic for aggregate_first=False (less common path)
        if not aggregate_first:
            corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
            corners_2d_norm = grid_norm + corners_offset

            voter_mask = (mask_gt > 0.5).reshape(B, 1, H_feat * W_feat)
            num_voters_per_image = voter_mask.sum(dim=-1).clamp(min=1.0) # (B, 1)
            scale_tensor = scale_tensor.view(1, 1, 2, 1, 1)
            corners_2d_scaled = corners_2d_norm * scale_tensor
            corners_2d_scaled = corners_2d_scaled.flatten(3)
            u = corners_2d_scaled[:, :, 0, :] # (B, N, HW)
            v = corners_2d_scaled[:, :, 1, :] # (B, N, HW)
            ray_x = (u - cx) / fx
            ray_y = (v - cy) / fy
            ray_z = torch.ones_like(ray_x)

            rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)
            rays_norm = F.normalize(rays, p=2, dim=-1)

            pts3d_gt_expanded = pts3d_gt.unsqueeze(2)
            dot_prod = (pts3d_gt_expanded * rays_norm).sum(dim=-1, keepdim=True) # (B, N, HW, 1)
            proj_point = dot_prod * rays_norm # (B, N, HW, 3)
            dist_3d = torch.norm(pts3d_gt_expanded - proj_point, p=2, dim=-1) # (B, N, HW)
            loss_per_corner = (dist_3d * voter_mask).sum(dim=-1) # (B, N)
            loss_per_corner = loss_per_corner / num_voters_per_image
            return loss_per_corner.mean()

        # Optimized Path for aggregate_first=True
        # Avoid explicit creation of corners_2d_norm = grid + offset (Shape: B, N, 2, H, W)
        # Instead, compute weighted sum of grid and offset separately using matrix multiplication.

        # 1. Grid Component
        # weighted_sum_grid = sum_{hw} (grid_{hw} * mask_{b,hw})
        # Matrix form: mask_flat @ grid_flat.T
        grid_flat = grid_norm.flatten(1) # (2, HW)
        mask_flat = mask_gt.view(B, -1) # (B, HW)
        term1 = mask_flat @ grid_flat.t() # (B, 2)

        # 2. Offset Component
        # weighted_sum_offset = sum_{hw} (offset_{b,c,hw} * mask_{b,1,hw})
        # Matrix form: bmm(pvmap_flat, mask_flat.T)
        pvmap_flat = pvmap.flatten(2) # (B, C, HW)
        mask_flat_t = mask_gt.flatten(2).transpose(1, 2) # (B, HW, 1)
        term2 = torch.bmm(pvmap_flat, mask_flat_t) # (B, C, 1)
        term2 = term2.view(B, num_corners, 2)

        # Combine
        weighted_sum = term1.unsqueeze(1) + term2 # (B, N, 2)

        # Total weight
        total_weight = mask_flat.sum(dim=1).clamp(min=1.0).view(B, 1, 1)

        corners_2d_agg_norm = weighted_sum / total_weight

        # Ray casting and Distance (Same as original, but on small tensors)
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
