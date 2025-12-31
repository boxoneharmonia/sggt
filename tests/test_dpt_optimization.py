
import torch
import torch.nn as nn
import time
import unittest
from pose.module import DPTHead
from pose.optimized_dpt import OptimizedDPTHead

class TestDPTOptimization(unittest.TestCase):
    def setUp(self):
        self.B = 8
        self.H, self.W = 128, 128
        self.patch_size = 16
        self.L = (self.H // self.patch_size) * (self.W // self.patch_size)
        self.C = 64
        self.chunk_size = 2

        self.token_list = [
            torch.randn(self.B, self.L, self.C),
            torch.randn(self.B, self.L, self.C),
            torch.randn(self.B, self.L, self.C),
            torch.randn(self.B, self.L, self.C)
        ]

        # Initialize original model
        self.original = DPTHead(
            inp=self.C,
            oup=10,
            features=64,
            patch_size=self.patch_size,
            pos_emb=True,
            use_conf=True
        )
        self.original.eval()

        # Initialize optimized model and copy weights
        self.optimized = OptimizedDPTHead(
            inp=self.C,
            oup=10,
            features=64,
            patch_size=self.patch_size,
            pos_emb=True,
            use_conf=True
        )
        self.optimized.load_state_dict(self.original.state_dict())
        self.optimized.eval()

    def test_correctness(self):
        """Verify that optimized model produces identical output"""
        with torch.no_grad():
            out_orig = self.original(self.token_list, (self.H, self.W), frames_chunk_size=self.chunk_size)
            out_opt = self.optimized(self.token_list, (self.H, self.W), frames_chunk_size=self.chunk_size)

        # Check closeness
        self.assertTrue(torch.allclose(out_orig, out_opt, atol=1e-6))
        print("\nCorrectness Verified: Outputs are identical.")

    def test_performance(self):
        """Benchmark performance difference"""
        # Warmup
        iters = 20
        with torch.no_grad():
            for _ in range(5):
                self.original(self.token_list, (self.H, self.W), frames_chunk_size=self.chunk_size)
                self.optimized(self.token_list, (self.H, self.W), frames_chunk_size=self.chunk_size)

        # Measure Original
        start = time.time()
        for _ in range(iters):
            with torch.no_grad():
                self.original(self.token_list, (self.H, self.W), frames_chunk_size=self.chunk_size)
        orig_time = time.time() - start

        # Measure Optimized
        start = time.time()
        for _ in range(iters):
            with torch.no_grad():
                self.optimized(self.token_list, (self.H, self.W), frames_chunk_size=self.chunk_size)
        opt_time = time.time() - start

        print(f"\nPerformance Benchmark ({iters} iterations):")
        print(f"Original: {orig_time:.4f}s")
        print(f"Optimized: {opt_time:.4f}s")
        print(f"Speedup: {orig_time / opt_time:.2f}x")
        print(f"Reduction: {(orig_time - opt_time) / orig_time * 100:.1f}%")

if __name__ == "__main__":
    unittest.main()
