
import torch
import torch.nn.functional as F
import time
import sys
import os
import unittest

# Ensure we can import pose modules
sys.path.append(os.getcwd())

from pose.loss_optimized import PoseLossOptimized
from pose.net import PoseLoss
from config import Config

class TestVotingLossOptimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B = 4
        num_corners = 8
        self.C = num_corners * 2
        self.H_feat = 24
        self.W_feat = 24

        # Create dummy data
        self.pvmap = torch.randn(self.B, self.C, self.H_feat, self.W_feat, device=self.device)
        self.mask_gt = torch.rand(self.B, self.H_feat, self.W_feat, device=self.device)
        self.pts3d_gt = torch.randn(self.B, num_corners, 3, device=self.device)

        # Create intrinsic matrix
        fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
        self.cam_K = torch.zeros(self.B, 3, 3, device=self.device)
        self.cam_K[:, 0, 0] = fx
        self.cam_K[:, 1, 1] = fy
        self.cam_K[:, 0, 2] = cx
        self.cam_K[:, 1, 2] = cy
        self.cam_K[:, 2, 2] = 1.0

        self.scale_tensor = torch.tensor((self.W_feat*14, self.H_feat*14), device=self.device).view(1, 1, 2)

        config = Config()
        self.pose_loss_orig = PoseLoss(config).to(self.device)
        self.pose_loss_opt = PoseLossOptimized(config).to(self.device)

    def test_correctness_aggregate_first_true(self):
        res_orig = self.pose_loss_orig.voting_loss(self.pvmap, self.mask_gt, self.pts3d_gt, self.cam_K, self.scale_tensor, aggregate_first=True)
        res_opt = self.pose_loss_opt.voting_loss(self.pvmap, self.mask_gt, self.pts3d_gt, self.cam_K, self.scale_tensor, aggregate_first=True)
        self.assertTrue(torch.allclose(res_orig, res_opt, atol=1e-4), f"Results differ: {torch.abs(res_orig - res_opt).item()}")

    def test_correctness_aggregate_first_false(self):
        res_orig = self.pose_loss_orig.voting_loss(self.pvmap, self.mask_gt, self.pts3d_gt, self.cam_K, self.scale_tensor, aggregate_first=False)
        res_opt = self.pose_loss_opt.voting_loss(self.pvmap, self.mask_gt, self.pts3d_gt, self.cam_K, self.scale_tensor, aggregate_first=False)
        self.assertTrue(torch.allclose(res_orig, res_opt, atol=1e-4), f"Results differ: {torch.abs(res_orig - res_opt).item()}")

    def test_speed(self):
        print("\nBenchmarking Speed...")
        iterations = 100

        # Warmup
        _ = self.pose_loss_orig.voting_loss(self.pvmap, self.mask_gt, self.pts3d_gt, self.cam_K, self.scale_tensor, aggregate_first=True)

        if self.device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = self.pose_loss_orig.voting_loss(self.pvmap, self.mask_gt, self.pts3d_gt, self.cam_K, self.scale_tensor, aggregate_first=True)
        if self.device.type == 'cuda': torch.cuda.synchronize()
        orig_time = time.time() - start

        if self.device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = self.pose_loss_opt.voting_loss(self.pvmap, self.mask_gt, self.pts3d_gt, self.cam_K, self.scale_tensor, aggregate_first=True)
        if self.device.type == 'cuda': torch.cuda.synchronize()
        opt_time = time.time() - start

        print(f"Original Time: {orig_time*1000/iterations:.4f} ms")
        print(f"Optimized Time: {opt_time*1000/iterations:.4f} ms")
        # We don't assert speed improvement as it depends on environment, but we log it.

if __name__ == '__main__':
    unittest.main()
