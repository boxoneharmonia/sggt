import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose.net import PoseLoss
from config import Config

# --- Subclass to Fix the Bug locally for testing ---
class CorrectedPoseLoss(PoseLoss):
    def voting_loss(self, pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor, aggregate_first=True):
        B, C, H_feat, W_feat = pvmap.shape
        num_corners = C // 2
        device = pvmap.device

        mask_expanded = F.max_pool2d(mask_gt, kernel_size=3, stride=1, padding=1)

        y_range = torch.linspace(0, 1, H_feat, device=device)
        x_range = torch.linspace(0, 1, W_feat, device=device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        # FIX: Explicitly permute to (2, H, W) before reshaping to match channel ordering if needed
        # But wait, let's look at how grid is stacked.
        # grid_norm = torch.stack([grid_x, grid_y], dim=-1) -> (H, W, 2)
        # corners_offset = pvmap.view(...) -> (B, N, 2, H, W)
        # We need grid to match (2, H, W).

        # grid_norm_permuted = grid_norm.permute(2, 0, 1) # (2, H, W)
        # grid_expanded = grid_norm_permuted.view(1, 1, 2, H, W)

        # ORIGINAL BROKEN CODE:
        # grid_norm = torch.stack([grid_x, grid_y], dim=-1) # (H_feat, W_feat, 2)
        # corners_2d_norm = grid_norm.view(1, 1, 2, H_feat, W_feat) + corners_offset

        # CORRECTED CODE:
        grid_norm = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2)
        grid_permuted = grid_norm.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) # (1, 1, 2, H, W)

        corners_offset = pvmap.view(B, num_corners, 2, H_feat, W_feat)
        corners_2d_norm = grid_permuted + corners_offset # (B, N, 2, H, W)

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
            voter_mask = (mask_expanded > 0.5).repeat_interleave(num_corners, dim=0)  # (B*N, 1, H, W)
            voter_mask = voter_mask.view(B * num_corners, H_feat * W_feat)
            num_voters_per_image = voter_mask.sum(dim=-1).clamp(min=1.0)

            # FIX: Scale tensor view was also suspicious in original?
            # Original: scale_tensor.view(1, 1, 2, 1, 1)
            # scale_tensor is (1, 1, 2).
            # We want to scale corners_2d_norm which is (B, N, 2, H, W).
            # So (1, 1, 2, 1, 1) is correct broadcasting.
            scale_tensor = scale_tensor.view(1, 1, 2, 1, 1)

            corners_2d_scaled = corners_2d_norm * scale_tensor

            # Note: corners_2d_scaled is (B, N, 2, H, W)
            # We want homogeneous (B, N, 3, H, W)
            # Original: torch.cat([..., torch.ones...], dim=2) -> Correct.

            corners_homo = torch.cat([corners_2d_scaled, torch.ones_like(corners_2d_scaled[:, :, :1])], dim=2) # (B, N, 3, H, W)

            from einops import rearrange # ensure import
            corners_homo = rearrange(corners_homo, 'b n c h w -> (b n) (h w) c') # (B*N, H*W, 3)

            K_inv = torch.linalg.inv(cam_K) # (B, 3, 3)
            K_inv_expanded = K_inv.repeat_interleave(num_corners, dim=0) # (B*N, 3, 3)

            rays = corners_homo @ K_inv_expanded.transpose(-1, -2) # (B*N, H*W, 3)
            rays_norm = F.normalize(rays, p=2, dim=-1)

            pts3d_gt_expanded = pts3d_gt.view(B * num_corners, 1, 3).expand(-1, H_feat * W_feat, -1)
            dot_prod = (pts3d_gt_expanded * rays_norm).sum(dim=-1, keepdim=True)
            proj_point = dot_prod * rays_norm
            dist_3d = torch.norm(pts3d_gt_expanded - proj_point, p=2, dim=-1) # (B*N, H*W)

            loss_per_corner = (dist_3d * voter_mask).sum(dim=-1) / num_voters_per_image
            return loss_per_corner.mean()

def test_cube_projection_loss():
    print("--- Setting up Cube Projection Test ---")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup Config
    config = Config()
    config.maps = 16

    # Use Corrected Loss to bypass the bug I found
    criterion = CorrectedPoseLoss(config)
    # criterion_orig = PoseLoss(config) # We can compare if needed

    B = 1
    H_orig, W_orig = 640, 480
    H_feat, W_feat = 64, 48 # dpt output size
    device = torch.device('cpu')

    # 2. Define Cube (10cm = 0.1m, assuming units in meters? or cm? Let's use generic units)
    # 10cm cube -> +/- 5cm.
    # Vertices (8, 3)
    s = 5.0
    cube_vertices = torch.tensor([
        [-s, -s, -s], [-s, -s, s], [-s, s, -s], [-s, s, s],
        [s, -s, -s],  [s, -s, s],  [s, s, -s],  [s, s, s]
    ], dtype=torch.float32, device=device) # (8, 3)

    # 3. Camera Setup
    # K: focal length 500
    fx, fy = 500.0, 500.0
    cx, cy = W_orig / 2.0, H_orig / 2.0
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0) # (1, 3, 3)

    # Pose: Move object away
    # R: Identity (axis aligned)
    # t: [0, 0, 30] (30 units away)
    R = torch.eye(3, device=device).unsqueeze(0)
    t = torch.tensor([[0.0, 0.0, 40.0]], device=device).unsqueeze(2) # (1, 3, 1)

    # 4. Project to 3D Camera Frame
    # P_cam = R @ P_world + t
    # (B, 3, 3) @ (8, 3).T + (B, 3, 1)
    pts3d_cam = (R @ cube_vertices.T + t).transpose(1, 2) # (B, 8, 3)

    # 5. Project to 2D Image
    # P_img = K @ P_cam
    pts_homo = (K @ pts3d_cam.transpose(1, 2)).transpose(1, 2) # (B, 8, 3)
    z = pts_homo[..., 2:]
    pts2d = pts_homo[..., :2] / z # (B, 8, 2)

    # 6. Generate Mask (Convex Hull of projected points)
    # Using OpenCV
    mask_np = np.zeros((H_orig, W_orig), dtype=np.uint8)
    pts2d_np = pts2d[0].detach().cpu().numpy().astype(np.int32) # (8, 2)
    # Find convex hull
    hull = cv2.convexHull(pts2d_np)
    cv2.fillConvexPoly(mask_np, hull, 1)

    # Downsample mask to feature size
    mask_tensor_orig = torch.from_numpy(mask_np).float().to(device).view(1, 1, H_orig, W_orig)
    mask_gt = F.interpolate(mask_tensor_orig, size=(H_feat, W_feat), mode='nearest-exact')

    # 7. Generate Perfect Voting Map
    # For each pixel in feature map (u_feat, v_feat)
    y_range = torch.linspace(0, 1, H_feat, device=device)
    x_range = torch.linspace(0, 1, W_feat, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    # grid: (H, W, 2)

    # Expand 2D targets (normalized)
    scale_tensor = torch.tensor([W_orig, H_orig], device=device).view(1, 1, 2)
    pts2d_norm = pts2d / scale_tensor # (B, 8, 2)

    # Broadcast
    # grid_norm: (H, W, 2) -> (1, 1, H, W, 2)
    grid_expanded = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, H_feat, W_feat, 2)

    # pts2d_norm: (1, 8, 2) -> (1, 8, 1, 1, 2) -> (1, 8, H, W, 2)
    target_expanded = pts2d_norm.view(1, 8, 1, 1, 2).expand(-1, -1, H_feat, W_feat, -1)

    # PVMap = Target - Grid
    pvmap_perfect_expanded = target_expanded - grid_expanded # (1, 8, H, W, 2)

    # Reshape to (B, C, H, W)
    pvmap_perfect = pvmap_perfect_expanded.permute(0, 1, 4, 2, 3).reshape(B, 16, H_feat, W_feat)

    print("\n--- Running Tests with Corrected Loss ---")

    # Test 1: Correctness
    loss_agg = criterion.voting_loss(pvmap_perfect, mask_gt, pts3d_cam, K, scale_tensor, aggregate_first=True)
    loss_no_agg = criterion.voting_loss(pvmap_perfect, mask_gt, pts3d_cam, K, scale_tensor, aggregate_first=False)

    print(f"Loss Perfect (Agg=True):  {loss_agg.item():.8f}")
    print(f"Loss Perfect (Agg=False): {loss_no_agg.item():.8f}")

    if loss_agg < 1e-4 and loss_no_agg < 1e-4:
        print(">> PASS: Correctness verified (Loss ~ 0).")
    else:
        print(">> FAIL: Loss should be ~0.")

    # Test 2: Robustness
    print("\n--- Robustness Test ---")
    # Add noise to PVMap
    # Noise magnitude: e.g., 2 pixels
    noise_pixels = 2.0
    noise_norm = noise_pixels / torch.tensor([W_orig, H_orig], device=device).view(1, 1, 1, 1, 2)

    # Random noise
    noise = torch.randn_like(pvmap_perfect_expanded) * noise_norm
    pvmap_noisy_expanded = pvmap_perfect_expanded + noise
    pvmap_noisy = pvmap_noisy_expanded.permute(0, 1, 4, 2, 3).reshape(B, 16, H_feat, W_feat)

    loss_noisy_agg = criterion.voting_loss(pvmap_noisy, mask_gt, pts3d_cam, K, scale_tensor, aggregate_first=True)
    loss_noisy_no_agg = criterion.voting_loss(pvmap_noisy, mask_gt, pts3d_cam, K, scale_tensor, aggregate_first=False)

    print(f"Loss Noisy (Agg=True):  {loss_noisy_agg.item():.8f}")
    print(f"Loss Noisy (Agg=False): {loss_noisy_no_agg.item():.8f}")

    diff = loss_noisy_no_agg.item() - loss_noisy_agg.item()
    print(f"Difference (NoAgg - Agg): {diff:.8f}")

    if loss_noisy_agg < loss_noisy_no_agg:
        print(">> RESULT: Aggregate First is MORE robust.")
    else:
        print(">> RESULT: Aggregate First is LESS robust.")

if __name__ == "__main__":
    test_cube_projection_loss()
