
import torch
import time
import sys
import os

# Add the parent directory to sys.path to allow importing from pose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose.module import DPTHead
from pose.optimized_dpt import OptimizedDPTHead

def benchmark_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Setup parameters
    B = 16
    H, W = 224, 224
    patch_size = 16
    inp_dim = 256
    oup_dim = 64

    # Calculate N
    patch_h = H // patch_size
    patch_w = W // patch_size
    N = patch_h * patch_w

    print(f"Batch size: {B}, Resolution: {H}x{W}, Patch size: {patch_size}, N: {N}")

    # Create models
    model_orig = DPTHead(inp=inp_dim, oup=oup_dim, patch_size=patch_size, pos_emb=True).to(device)
    model_opt = OptimizedDPTHead(inp=inp_dim, oup=oup_dim, patch_size=patch_size, pos_emb=True).to(device)

    # Ensure weights are same for correctness check (though we just check shape/runnability mostly,
    # but let's copy state dict to be sure outputs match)
    model_opt.load_state_dict(model_orig.state_dict())

    model_orig.eval()
    model_opt.eval()

    # Create inputs
    token_list = [torch.randn(B, N, inp_dim, device=device) for _ in range(4)]
    image_size = (H, W)

    # 1. Verification
    print("\nVerifying outputs...")
    with torch.no_grad():
        out_orig = model_orig(token_list, image_size)
        out_opt = model_opt(token_list, image_size)

    diff = (out_orig - out_opt).abs().max()
    print(f"Max difference between outputs: {diff.item()}")
    if diff.item() > 1e-5:
        print("WARNING: Outputs do not match!")
    else:
        print("Outputs match.")

    # 2. Benchmark Original
    print("\nBenchmarking Original DPTHead...")
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_orig(token_list, image_size)

    num_iters = 20
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model_orig(token_list, image_size)

    end_time = time.time()
    avg_time_orig = (end_time - start_time) / num_iters
    print(f"Original DPTHead Average time: {avg_time_orig*1000:.4f} ms")

    # 3. Benchmark Optimized
    print("\nBenchmarking Optimized DPTHead...")
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_opt(token_list, image_size)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model_opt(token_list, image_size)

    end_time = time.time()
    avg_time_opt = (end_time - start_time) / num_iters
    print(f"Optimized DPTHead Average time: {avg_time_opt*1000:.4f} ms")

    speedup = (avg_time_orig - avg_time_opt) / avg_time_orig * 100
    print(f"\nSpeedup: {speedup:.2f}%")

if __name__ == "__main__":
    benchmark_comparison()
