import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add repo root to path to allow imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose.grid_cache import GridCache

def verify_geometry():
    print("=== Starting Geometry Verification ===\n")

    # 1. Setup Mock Data
    torch.manual_seed(42)
    B = 1 # Batch size
    H, W = 480, 640 # Image dimensions
    device = torch.device('cpu')

    # Mock Intrinsics (K)
    fx = 500.0
    fy = 500.0
    cx = W / 2.0
    cy = H / 2.0
    cam_K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=torch.float32)

    # Mock Extrinsics (R, t) -> World to Camera
    R_cam = torch.eye(3, dtype=torch.float32)
    t_cam = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float32)

    # Mock 3D Point in World Frame
    pts3d_world = torch.tensor([[0.5, 0.5, 2.0],
                                [-0.5, -0.3, 3.0]], dtype=torch.float32)

    print(f"Intrinsics (K):\n{cam_K}")
    print(f"World 3D Points:\n{pts3d_world}\n")

    # 2. Simulate Dataset Projection (3D -> 2D)
    # Removing epsilon for pure geometric verification
    pts_cam = pts3d_world @ R_cam.t() + t_cam.view(1, 3)
    pts_proj = pts_cam @ cam_K.t()
    pts2d_gt = pts_proj[:, :2] / pts_proj[:, 2:3] # No epsilon

    print(f"Camera Frame 3D Points:\n{pts_cam}")
    print(f"Projected 2D Points (Pixels):\n{pts2d_gt}\n")

    # 3. Simulate Loss Function Logic (2D -> Ray -> Distance)
    pts3d_gt_for_loss = pts_cam
    u = pts2d_gt[:, 0]
    v = pts2d_gt[:, 1]

    print("--- Verifying Ray Reconstruction Math (Assuming correct 2D inputs) ---")
    fx_l = cam_K[0, 0]
    fy_l = cam_K[1, 1]
    cx_l = cam_K[0, 2]
    cy_l = cam_K[1, 2]

    ray_x = (u - cx_l) / fx_l
    ray_y = (v - cy_l) / fy_l
    ray_z = torch.ones_like(ray_x)

    rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)
    rays_norm = F.normalize(rays, p=2, dim=-1)

    dot_prod = (pts3d_gt_for_loss * rays_norm).sum(dim=-1, keepdim=True)
    proj_point = dot_prod * rays_norm
    dist_3d = torch.norm(pts3d_gt_for_loss - proj_point, p=2, dim=-1)

    print(f"Distance 3D (should be near 0): {dist_3d}")

    if torch.allclose(dist_3d, torch.zeros_like(dist_3d), atol=1e-6):
        print("PASS: Ray reconstruction math is correct.")
    else:
        print("FAILURE: Ray reconstruction math is incorrect.")

    # 4. Check Grid Scaling Logic
    print("\n--- Verifying Grid Coordinate System Consistency ---")

    # Dataset assumes pixels are 0..W-1 (integer coordinates)
    # Loss assumes pixels are generated from GridCache (0..1) * Scale (W)

    grid_w_steps = torch.linspace(0, 1, W)
    scaled_w_steps = grid_w_steps * W
    arange_w = torch.arange(W, dtype=torch.float32)

    diff = (scaled_w_steps - arange_w).abs()
    max_diff = diff.max()
    print(f"Max coordinate discrepancy: {max_diff.item():.6f} pixels")

    if max_diff > 1e-3:
        print("FAILURE: The loss function's grid generation does not match the dataset's pixel coordinate system.")
        print(f"Dataset Range: [0, {W-1}]")
        print(f"Loss Grid Range: [0, {scaled_w_steps[-1].item():.4f}]")
    else:
        print("PASS: Coordinate systems match.")

if __name__ == "__main__":
    verify_geometry()
