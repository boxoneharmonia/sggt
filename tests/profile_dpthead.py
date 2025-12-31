
import torch
import torch.nn as nn
from einops import rearrange
from pose.module import DPTHead
import time
import sys

# Monkey patch to count calls
original_apply_pos_embed = DPTHead._apply_pos_embed
call_count = 0

def mocked_apply_pos_embed(self, x, W, H):
    global call_count
    call_count += 1
    return original_apply_pos_embed(self, x, W, H)

DPTHead._apply_pos_embed = mocked_apply_pos_embed

def benchmark_dpthead():
    global call_count
    B = 8
    # Use sizes divisible by sufficient powers of 2
    # patch_size=16. H=128 -> patch_h=8.
    # resize[3] does downsample / 2. 8 -> 4.
    # FeatureFusionBlock upsamples * 2. 4 -> 8. Matches.
    H, W = 128, 128
    patch_size = 16
    L = (H // patch_size) * (W // patch_size) # 64
    C = 64

    # Create inputs
    token_list = [
        torch.randn(B, L, C),
        torch.randn(B, L, C),
        torch.randn(B, L, C),
        torch.randn(B, L, C)
    ]

    # Instantiate model
    model = DPTHead(inp=C, oup=10, features=64, patch_size=patch_size, pos_emb=True)
    model.eval()

    chunk_size = 2

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            model(token_list, (H, W), frames_chunk_size=chunk_size)

    call_count = 0
    # Measure
    start_time = time.time()
    iters = 20
    for _ in range(iters):
        with torch.no_grad():
            model(token_list, (H, W), frames_chunk_size=chunk_size)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f}s")
    print(f"Total calls to _apply_pos_embed: {call_count}")
    expected_calls_per_iter = (B // chunk_size) * 4 # 4 scales
    print(f"Calls per iteration: {call_count / iters}")
    print(f"Expected calls per iteration: {expected_calls_per_iter}")

if __name__ == "__main__":
    benchmark_dpthead()
