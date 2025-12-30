import torch
import time
import unittest
import sys
import os

# Add repo root to path so we can import pose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose.module import AltRefAttBlock
from pose.optimized_module import OptimizedAltRefAttBlock

class TestOptimizedAltRefAttBlock(unittest.TestCase):
    def test_correctness_and_performance(self):
        # Setup
        dim = 768
        num_heads = 12
        s = 8
        n = 196
        b = 1 # Decrease batch size to avoid OOM on CPU/small GPU

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Testing on {device} with dim={dim}, s={s}, n={n}")

        # Original model
        model_orig = AltRefAttBlock(dim=dim, num_heads=num_heads).to(device)
        model_orig.eval()

        # Optimized model
        model_opt = OptimizedAltRefAttBlock(dim=dim, num_heads=num_heads).to(device)
        model_opt.load_state_dict(model_orig.state_dict())
        model_opt.eval()

        # Input
        x = torch.randn(b, s, n, dim).to(device)

        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = model_orig(x)
                _ = model_opt(x)

        # Correctness check
        print("Checking correctness...")
        with torch.no_grad():
            out_orig = model_orig(x)
            out_opt = model_opt(x)

        diff = (out_orig - out_opt).abs().max().item()
        print(f"Max difference: {diff}")
        self.assertTrue(diff < 1e-4, f"Output mismatch! Max diff: {diff}")

        # Benchmark
        print("Benchmarking...")
        iterations = 20

        # Original
        if device == 'cuda': torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model_orig(x)
        if device == 'cuda': torch.cuda.synchronize()
        time_orig = time.time() - start

        # Optimized
        if device == 'cuda': torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model_opt(x)
        if device == 'cuda': torch.cuda.synchronize()
        time_opt = time.time() - start

        print(f"Original time: {time_orig:.4f}s")
        print(f"Optimized time: {time_opt:.4f}s")
        print(f"Speedup: {time_orig / time_opt:.2f}x")

if __name__ == '__main__':
    unittest.main()
