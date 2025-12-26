import torch
import torch.nn.functional as F
from .net import PoseLoss
from .grid_cache import GridCache

class PoseLossOptimized(PoseLoss):
    """
    Optimized version of PoseLoss that uses batched matrix multiplication
    for the voting loss calculation to improve memory usage and speed.
    """
    def voting_loss(self, pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True):
        """
        Optimized voting loss calculation using torch.bmm.
        """
        B, C, H_feat, W_feat = pvmap.shape
        num_corners = C // 2
        device = pvmap.device

        fx = cam_K[:, 0, 0].view(B, 1, 1)
        fy = cam_K[:, 1, 1].view(B, 1, 1)
        cx = cam_K[:, 0, 2].view(B, 1, 1)
        cy = cam_K[:, 1, 2].view(B, 1, 1)

        if aggregate_first:
            # Flatten spatial dimensions
            L = H_feat * W_feat

            # 1. Compute weighted sum of offsets (pvmap)
            # pvmap: (B, C, H, W) -> (B, C, L)
            pvmap_flat = pvmap.view(B, C, L)

            # mask: (B, 1, H, W) -> (B, 1, L)
            mask_flat = mask_gt.view(B, 1, L)

            # Weighted sum of pvmap: (B, C, L) @ (B, L, 1) -> (B, C, 1)
            # mask_flat is (B, 1, L), so we use transpose to get (B, L, 1)
            mask_flat_t = mask_flat.transpose(1, 2) # (B, L, 1)

            # (B, N*2, L) @ (B, L, 1) -> (B, N*2, 1)
            weighted_sum_offset = torch.bmm(pvmap_flat, mask_flat_t).view(B, num_corners, 2) # (B, N, 2)

            # 2. Compute weighted sum of grid
            grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H, W, 2)
            # grid_norm is (H, W, 2). Permute to (2, H, W). Flatten to (1, 2, L)
            grid_flat = grid_norm.permute(2, 0, 1).reshape(1, 2, L) # (1, 2, L)
            grid_flat = grid_flat.expand(B, 2, L) # (B, 2, L)

            # (B, 2, L) @ (B, L, 1) -> (B, 2, 1)
            weighted_sum_grid = torch.bmm(grid_flat, mask_flat_t).view(B, 1, 2) # (B, 1, 2)

            # 3. Combine
            # Broadcasting (B, 1, 2) to (B, N, 2)
            weighted_sum = weighted_sum_grid + weighted_sum_offset

            # 4. Total weight
            total_weight = mask_flat.sum(dim=-1).view(B, 1, 1).clamp(min=1.0) # (B, 1, 1)

            corners_2d_agg_norm = weighted_sum / total_weight # (B, N, 2)
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
            # Fallback to original implementation for aggregate_first=False
            # We need to reconstruct the original logic or call super().voting_loss
            # calling super().voting_loss is cleaner but might re-do some setup.
            # But since aggregate_first=False uses different logic, it's safer to just call super
            return super().voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=False)
