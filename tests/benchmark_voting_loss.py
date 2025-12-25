import torch
import time
import pytest
from pose.net import PoseLoss
from pose.loss_optimized import PoseLossOptimized
from config import Config

def benchmark_voting_loss():
    # Setup
    config = Config()
    config.maps = 16 # 8 corners * 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Initialize losses
    loss_orig = PoseLoss(config).to(device)
    loss_opt = PoseLossOptimized(config).to(device)

    # Inputs
    B = 32
    C = 16 # 8 corners * 2
    H = 64
    W = 64
    N = C // 2

    pvmap = torch.randn(B, C, H, W, device=device)
    mask_gt = torch.rand(B, H, W, device=device)
    pts3d_gt = torch.randn(B, N, 3, device=device)
    cam_K = torch.randn(B, 3, 3, device=device)
    # Ensure fx, fy are not zero
    cam_K[:, 0, 0] = 1000.0
    cam_K[:, 1, 1] = 1000.0

    scale_tensor = torch.tensor([W-1.0, H-1.0], device=device).view(1, 1, 2)

    # Warmup
    for _ in range(10):
        _ = loss_orig.voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
        _ = loss_opt.voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)

    # Correctness Check
    out_orig = loss_orig.voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    out_opt = loss_opt.voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)

    diff = torch.abs(out_orig - out_opt).item()
    print(f"Difference: {diff:.6e}")
    if diff > 1e-5:
        print("WARNING: Outputs differ significantly!")
    else:
        print("Outputs match.")

    # Benchmark Original
    start_time = time.time()
    iters = 100
    for _ in range(iters):
        _ = loss_orig.voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    time_orig = (end_time - start_time) / iters
    print(f"Original Time: {time_orig*1000:.4f} ms")

    # Benchmark Optimized
    start_time = time.time()
    for _ in range(iters):
        _ = loss_opt.voting_loss(pvmap, mask_gt, pts3d_gt, cam_K, scale_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    time_opt = (end_time - start_time) / iters
    print(f"Optimized Time: {time_opt*1000:.4f} ms")

    print(f"Speedup: {time_orig / time_opt:.2f}x")

if __name__ == "__main__":
    benchmark_voting_loss()
