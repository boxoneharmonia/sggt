import torch
import time
import unittest
from pose.module import DPTHead
from pose.dpt_optimized import DPTHeadOptimized

class TestDPTOptimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.B = 4
        self.N = 576
        self.C = 768
        self.H, self.W = 336, 336

        self.inp = self.C
        self.oup = 8
        self.hidden_ratio = 2.0
        self.features = 256
        self.patch_size = 14

        self.model_orig = DPTHead(
            inp=self.inp, oup=self.oup, hidden_ratio=self.hidden_ratio,
            features=self.features, patch_size=self.patch_size, pos_emb=True, use_conf=True
        ).to(self.device).eval()

        self.model_opt = DPTHeadOptimized(
            inp=self.inp, oup=self.oup, hidden_ratio=self.hidden_ratio,
            features=self.features, patch_size=self.patch_size, pos_emb=True, use_conf=True
        ).to(self.device).eval()

        # Sync weights for correctness check
        self.model_opt.load_state_dict(self.model_orig.state_dict(), strict=False)

        self.input_tokens = [
            torch.randn(self.B, self.N, self.C).to(self.device) for _ in range(4)
        ]

    def test_correctness(self):
        print("Testing correctness...")
        with torch.no_grad():
            out_orig = self.model_orig(self.input_tokens, (self.H, self.W))
            out_opt = self.model_opt(self.input_tokens, (self.H, self.W))

        diff = (out_orig - out_opt).abs().max()
        print(f"Max difference: {diff.item()}")
        self.assertTrue(torch.allclose(out_orig, out_opt, atol=1e-5), f"Outputs mismatch! Max diff: {diff}")

    def test_performance_smoke(self):
        print("Running smoke performance test...")
        # Warmup
        for _ in range(2):
            _ = self.model_orig(self.input_tokens, (self.H, self.W))
            _ = self.model_opt(self.input_tokens, (self.H, self.W))

        start = time.time()
        for _ in range(5):
            _ = self.model_opt(self.input_tokens, (self.H, self.W))
        dur = time.time() - start
        print(f"Optimized model took {dur:.4f}s for 5 runs")
        self.assertLess(dur, 20.0, "Model is too slow!")

if __name__ == '__main__':
    unittest.main()
