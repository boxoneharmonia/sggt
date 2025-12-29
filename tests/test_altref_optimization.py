import torch
import time
import unittest
import math
from pose.module import AltRefAttBlock
from pose.optimized_module import OptimizedAltRefAttBlock

class TestAltRefOptimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.B = 4
        self.S = 8 # Sequence length
        self.N = 196 # Patches
        self.dim = 256
        self.num_heads = 8

        # Random input
        self.x = torch.randn(self.B, self.S, self.N, self.dim).to(self.device)

        self.original = AltRefAttBlock(
            dim=self.dim, num_heads=self.num_heads, qkv_bias=True
        ).to(self.device)
        self.original.eval()

        self.optimized = OptimizedAltRefAttBlock(
            dim=self.dim, num_heads=self.num_heads, qkv_bias=True
        ).to(self.device)

        # Copy weights
        self.optimized.load_state_dict(self.original.state_dict())
        self.optimized.eval()

    def test_correctness(self):
        with torch.no_grad():
            out_orig = self.original(self.x)
            out_opt = self.optimized(self.x)

            # Using higher tolerance because float operations order might differ slightly
            # especially with concatenation and reshaping
            torch.testing.assert_close(out_orig, out_opt, rtol=1e-5, atol=1e-5)
            print("Correctness check passed!")

    def test_performance(self):
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                self.original(self.x)
                self.optimized(self.x)

        num_iters = 20

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iters):
                self.original(self.x)
        orig_time = time.time() - start_time

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iters):
                self.optimized(self.x)
        opt_time = time.time() - start_time

        print(f"Original time: {orig_time:.4f}s")
        print(f"Optimized time: {opt_time:.4f}s")
        print(f"Speedup: {orig_time / opt_time:.2f}x")

        self.assertLess(opt_time, orig_time, "Optimized version should be faster")

if __name__ == '__main__':
    unittest.main()
