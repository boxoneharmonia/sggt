import torch
import torch.nn.functional as F
from pose.grid_cache import GridCache
from pose.net import PoseLoss

class PoseLossOptimized(PoseLoss):
    """
    Optimized version of PoseLoss that uses vectorized operations for voting_loss
    to improve performance and memory usage.
    """
    def voting_loss(self, pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=False):
        """
        Optimized voting loss calculation using broadcasting instead of repeat_interleave.
        """
        B, C, H_feat, W_feat = pvmap.shape
        num_corners = C // 2
        device = pvmap.device

        mask_expanded = F.max_pool2d(mask_gt, kernel_size=3, stride=1, padding=1)

        grid_norm = GridCache.get_mesh_grid(H_feat, W_feat, device=device) # (H_feat, W_feat, 2)

        # PERMUTE FIX: Ensure grid_norm is (2, H, W) to match (B, N, 2, H, W)
        if grid_norm.shape[-1] == 2:
             grid_norm = grid_norm.permute(2, 0, 1)

        corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
        corners_2d_norm = grid_norm + corners_offset # (B, N_corners, 2, H_feat, W_feat)

        if aggregate_first:
            return super().voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True)
        else:
            # Optimized path for aggregate_first=False

            # mask_expanded: (B, 1, H, W) -> (B, 1, HW)
            voter_mask = (mask_expanded > 0.5).reshape(B, 1, H_feat * W_feat)
            num_voters_per_image = voter_mask.sum(dim=-1).clamp(min=1.0) # (B, 1)

            scale_tensor = scale_tensor.view(1, 1, 2, 1, 1)
            corners_2d_scaled = corners_2d_norm * scale_tensor # (B, N, 2, H, W)

            # Flatten spatial dims: (B, N, 2, HW)
            corners_2d_scaled = corners_2d_scaled.flatten(3)

            # Create homogeneous coords: (B, N, 3, HW)
            corners_homo = torch.cat([corners_2d_scaled, torch.ones_like(corners_2d_scaled[:, :, :1, :])], dim=2)
            # Permute to (B, N, HW, 3) for matmul
            corners_homo = corners_homo.permute(0, 1, 3, 2)

            K_inv = torch.linalg.inv(cam_K) # (B, 3, 3)

            # Rays computation: (B, N, HW, 3) @ (B, 1, 3, 3).mT
            # We broadcast K_inv across N
            rays = corners_homo @ K_inv.transpose(-1, -2).unsqueeze(1)

            rays_norm = F.normalize(rays, p=2, dim=-1) # (B, N, HW, 3)

            # pts3d_gt: (B, N, 3) -> (B, N, 1, 3)
            pts3d_gt_expanded = pts3d_gt.unsqueeze(2)

            # Project points onto rays
            dot_prod = (pts3d_gt_expanded * rays_norm).sum(dim=-1, keepdim=True) # (B, N, HW, 1)
            proj_point = dot_prod * rays_norm # (B, N, HW, 3)

            dist_3d = torch.norm(pts3d_gt_expanded - proj_point, p=2, dim=-1) # (B, N, HW)

            # Apply mask and average
            # voter_mask broadcasts to (B, N, HW)
            loss_per_corner = (dist_3d * voter_mask).sum(dim=-1) # (B, N)

            # num_voters_per_image broadcasts to (B, N)
            loss_per_corner = loss_per_corner / num_voters_per_image

            return loss_per_corner.mean()
