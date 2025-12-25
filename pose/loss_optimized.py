import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import PoseLoss
from .grid_cache import GridCache

class PoseLossOptimized(PoseLoss):
    """
    Optimized version of PoseLoss that uses batch matrix multiplication
    for the voting loss aggregation step, improving performance.
    """
    def voting_loss(self, pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True):
        """
        Calculates the voting loss with optimized aggregation using bmm.
        """
        if not aggregate_first:
            # The parent implementation expects aggregate_first argument
            return super().voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=False)

        B, C, H_feat, W_feat = pvmap.shape
        num_corners = C // 2
        device = pvmap.device

        # Precompute common shapes
        HW = H_feat * W_feat

        # 1. Prepare Mask
        # mask_gt: (B, H, W) -> (B, HW, 1) for broadcasting
        # We need to perform weighted sum over H,W dimensions.
        mask_flat = mask_gt.view(B, HW, 1)
        total_weight = mask_flat.sum(dim=1).clamp(min=1.0).view(B, 1, 1) # (B, 1, 1)

        # 2. Grid Term: sum(grid * mask)
        # grid: (H, W, 2) -> (HW, 2)
        grid = GridCache.get_mesh_grid(H_feat, W_feat, device=device)
        grid_flat = grid.reshape(HW, 2)

        # We want sum_{h,w} (grid_{h,w} * mask_{b,h,w})
        # (B, 1, HW) @ (HW, 2) -> (B, 1, 2)
        # Transpose mask to (B, 1, HW)
        grid_term = torch.matmul(mask_flat.transpose(1, 2), grid_flat) # (B, 1, 2)

        # 3. Offset Term: sum(offset * mask)
        # pvmap: (B, C, H, W) -> (B, C, HW)
        # We want sum_{h,w} (pvmap_{b,c,h,w} * mask_{b,h,w})
        # (B, C, HW) @ (B, HW, 1) -> (B, C, 1)
        offset_term = torch.bmm(pvmap.view(B, C, HW), mask_flat) # (B, C, 1)

        # Reshape to (B, N, 2)
        offset_term = offset_term.view(B, num_corners, 2)

        # 4. Combine and Normalize
        # grid_term (B, 1, 2) + offset_term (B, N, 2) -> (B, N, 2)
        weighted_sum = grid_term + offset_term
        corners_2d_agg_norm = weighted_sum / total_weight # (B, N, 2)

        # 5. Project and Calculate Distance (Same as original)
        corners_2d_scaled = corners_2d_agg_norm * scale_tensor.view(1, 1, 2)

        fx = cam_K[:, 0, 0].view(B, 1, 1)
        fy = cam_K[:, 1, 1].view(B, 1, 1)
        cx = cam_K[:, 0, 2].view(B, 1, 1)
        cy = cam_K[:, 1, 2].view(B, 1, 1)

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
