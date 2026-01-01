
import torch
import time
import sys
import os
import unittest

# Add repo root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose.module import DPTHead
from pose.optimized_dpt import OptimizedDPTHead

class TestDPTOptimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B = 4
        self.C = 256
        self.H, self.W = 224, 224
        self.patch_size = 14
        self.inp_dim = self.C
        self.oup_dim = 16
        self.hidden_ratio = 2.0
        self.features = 256

        self.baseline_model = DPTHead(
            inp=self.inp_dim,
            oup=self.oup_dim,
            hidden_ratio=self.hidden_ratio,
            features=self.features,
            patch_size=self.patch_size,
            pos_emb=True,
            use_conf=True
        ).to(self.device)
        self.baseline_model.eval()

        self.optimized_model = OptimizedDPTHead(
            inp=self.inp_dim,
            oup=self.oup_dim,
            hidden_ratio=self.hidden_ratio,
            features=self.features,
            patch_size=self.patch_size,
            pos_emb=True,
            use_conf=True
        ).to(self.device)
        self.optimized_model.load_state_dict(self.baseline_model.state_dict())
        self.optimized_model.eval()

    def test_correctness(self):
        patch_h = self.H // self.patch_size
        patch_w = self.W // self.patch_size
        num_patches = patch_h * patch_w

        token_list = [torch.randn(self.B, num_patches, self.inp_dim).to(self.device) for _ in range(4)]
        image_size = (self.H, self.W)
        frames_chunk_size = 1

        with torch.no_grad():
            out_base = self.baseline_model(token_list, image_size, frames_chunk_size=frames_chunk_size)
            out_opt = self.optimized_model(token_list, image_size, frames_chunk_size=frames_chunk_size)

            diff = torch.abs(out_base - out_opt).max()
            print(f"\nMax difference: {diff.item()}")
            self.assertTrue(diff < 1e-5, f"Outputs do not match. Max diff: {diff.item()}")

    def test_performance(self):
        # Larger sizes for performance test
        B = 8
        H, W = 448, 448
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        num_patches = patch_h * patch_w

        token_list = [torch.randn(B, num_patches, self.inp_dim).to(self.device) for _ in range(4)]
        image_size = (H, W)
        frames_chunk_size = 1

        iterations = 20

        # Warmup
        with torch.no_grad():
             _ = self.baseline_model(token_list, image_size, frames_chunk_size=frames_chunk_size)
             _ = self.optimized_model(token_list, image_size, frames_chunk_size=frames_chunk_size)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.baseline_model(token_list, image_size, frames_chunk_size=frames_chunk_size)
        base_time = time.time() - start_time

        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.optimized_model(token_list, image_size, frames_chunk_size=frames_chunk_size)
        opt_time = time.time() - start_time

        print(f"\nBaseline: {base_time:.4f}s, Optimized: {opt_time:.4f}s")
        print(f"Speedup: {base_time/opt_time:.2f}x")

if __name__ == "__main__":
    unittest.main()
