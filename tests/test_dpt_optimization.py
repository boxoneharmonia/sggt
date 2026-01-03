import torch
import torch.nn as nn
import time
import copy
from einops import rearrange
from pose.module import DPTHead
from pose.optimized_dpt import OptimizedDPTHead

def benchmark():
    print("Benchmarking DPTHead optimization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup config
    B = 16 # Increased batch size
    N_patches = 196 # 14x14
    C = 384
    H, W = 224, 224
    patch_size = 16

    # Create model
    original_model = DPTHead(inp=C, oup=16, hidden_ratio=2.0, features=256, patch_size=patch_size, use_conf=True).to(device)
    optimized_model = OptimizedDPTHead(inp=C, oup=16, hidden_ratio=2.0, features=256, patch_size=patch_size, use_conf=True).to(device)

    # Ensure weights match
    optimized_model.load_state_dict(original_model.state_dict())

    # Create dummy input
    token_list = [torch.randn(B, N_patches, C, device=device) for _ in range(4)]

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(2):
            _ = original_model(token_list, (H, W))
            _ = optimized_model(token_list, (H, W))

    # Benchmark Original
    iterations = 20
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            out_orig = original_model(token_list, (H, W))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    orig_time = time.time() - start_time
    print(f"Original DPTHead time: {orig_time:.4f}s")

    # Benchmark Optimized
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            out_opt = optimized_model(token_list, (H, W))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    opt_time = time.time() - start_time
    print(f"Optimized DPTHead time: {opt_time:.4f}s")

    # Check correctness
    diff = (out_orig - out_opt).abs().max()
    print(f"Max difference: {diff.item()}")

    if diff.item() < 1e-5:
        print("✅ Outputs match!")
    else:
        print("❌ Outputs do not match!")

    speedup = orig_time / opt_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
