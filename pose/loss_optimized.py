import torch
import torch.nn.functional as F
from .net import PoseLoss
from .grid_cache import GridCache

class PoseLossOptimized(PoseLoss):
    """
    Optimized version of PoseLoss that uses algebraic simplification for
    geometric voting loss calculation to reduce memory usage and increase speed.
    """
    def voting_loss(self, pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True):
        """
        Optimized voting loss calculation.

        Optimizations:
        1. Avoids explicit stacking of rays (BxNx3) to save memory.
        2. Avoids explicit normalization of rays.
        3. Computes dot product and distance using decomposed coordinates and
           algebraic identity: ||P - proj||^2 = ||P||^2 - (P . D)^2
           where D is the normalized ray direction.
        """
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

            # Optimization:
            # D = (rx, ry, 1) / sqrt(rx^2 + ry^2 + 1)
            # We want dist = ||P - (P.D)D||
            # dist^2 = ||P||^2 - (P.D)^2  (since ||D||=1)

            norm_sq = ray_x.square() + ray_y.square() + 1.0
            inv_norm = torch.rsqrt(norm_sq) # 1 / sqrt(x^2 + y^2 + 1)

            # P dot D = (Px * rx + Py * ry + Pz * 1) * inv_norm
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
            # Fallback to original implementation for aggregate_first=False
            # as it is not the main path and logic is different (per pixel loss)
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
