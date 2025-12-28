import torch
import time
import sys
import os
import unittest

# Add parent directory to path to import pose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose.module import DPTHead
from pose.optimized_dpt import OptimizedDPTHead

class TestDPTHeadOptimization(unittest.TestCase):
    def test_correctness_and_performance(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on {device}")

        inp_dim = 64
        oup_dim = 16
        features = 32
        hidden_ratio = 2.0
        patch_size = 4

        # Instantiate both models
        original_model = DPTHead(
            inp=inp_dim,
            oup=oup_dim,
            hidden_ratio=hidden_ratio,
            features=features,
            patch_size=patch_size,
            pos_emb=True
        ).to(device)

        optimized_model = OptimizedDPTHead(
            inp=inp_dim,
            oup=oup_dim,
            hidden_ratio=hidden_ratio,
            features=features,
            patch_size=patch_size,
            pos_emb=True
        ).to(device)

        # Copy weights from original to optimized to ensure identical behavior
        optimized_model.load_state_dict(original_model.state_dict())

        original_model.eval()
        optimized_model.eval()

        # Create input
        B = 8
        H, W = 64, 64
        num_patches = (H // patch_size) * (W // patch_size)
        token_list = [
            torch.randn(B, num_patches, inp_dim, device=device) for _ in range(4)
        ]
        image_size = (H, W)

        # Correctness check
        with torch.no_grad():
            out_orig = original_model(token_list, image_size)
            out_opt = optimized_model(token_list, image_size)

        # Verify outputs are close
        max_diff = (out_orig - out_opt).abs().max()
        print(f"Max difference between original and optimized: {max_diff.item()}")
        self.assertTrue(torch.allclose(out_orig, out_opt, atol=1e-5), f"Outputs differ! Max diff: {max_diff.item()}")

        # Performance benchmark
        iterations = 50

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                original_model(token_list, image_size)
                optimized_model(token_list, image_size)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                original_model(token_list, image_size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        orig_time = time.time() - start_time

        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                optimized_model(token_list, image_size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        opt_time = time.time() - start_time

        print(f"Original DPTHead time ({iterations} iters): {orig_time:.4f}s")
        print(f"Optimized DPTHead time ({iterations} iters): {opt_time:.4f}s")
        print(f"Speedup: {orig_time / opt_time:.2f}x")

        # We expect some speedup, or at least parity
        self.assertLessEqual(opt_time, orig_time * 1.05, "Optimized version is significantly slower!")

if __name__ == '__main__':
    unittest.main()
