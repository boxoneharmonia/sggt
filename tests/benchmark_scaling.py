import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add repo root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose.module import AltAttBlock, AltRefAttBlock

def benchmark_speed(model, input_tensor, iterations=20, warmup=5):
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    return avg_time * 1000  # Convert to ms

def run_scaling_benchmark():
    print("=== Benchmarking Scaling with Sequence Length (CPU) ===")

    # Configuration
    dim = 256
    num_heads = 8
    batch_size = 1
    num_tokens = 64  # Keep N fixed, vary S
    seq_lengths = [4, 8, 16, 32, 64]

    times_alt = []
    times_ref = []

    print(f"Config: Dim={dim}, Heads={num_heads}, Batch={batch_size}, Tokens={num_tokens}")
    print(f"{'Seq Len':<10} | {'AltAtt (ms)':<15} | {'AltRefAtt (ms)':<15}")
    print("-" * 45)

    for s in seq_lengths:
        input_tensor = torch.randn(batch_size, s, num_tokens, dim)

        # Instantiate Models (Re-init to be safe, though not strictly necessary)
        model_alt = AltAttBlock(dim=dim, num_heads=num_heads)
        model_ref = AltRefAttBlock(dim=dim, num_heads=num_heads)

        t_alt = benchmark_speed(model_alt, input_tensor)
        t_ref = benchmark_speed(model_ref, input_tensor)

        times_alt.append(t_alt)
        times_ref.append(t_ref)

        print(f"{s:<10} | {t_alt:<15.2f} | {t_ref:<15.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, times_alt, marker='o', label='AltAtt (Global Attention)', linewidth=2)
    plt.plot(seq_lengths, times_ref, marker='s', label='AltRefAtt (Ref Pair Attention)', linewidth=2)

    plt.title('Inference Speed vs Sequence Length')
    plt.xlabel('Sequence Length (S)')
    plt.ylabel('Inference Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add trend annotations
    plt.annotate('Quadratic Growth O(S^2)', xy=(seq_lengths[-2], times_alt[-2]),
                 xytext=(seq_lengths[-2], times_alt[-2] + 50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.annotate('Linear Growth O(S)', xy=(seq_lengths[-2], times_ref[-2]),
                 xytext=(seq_lengths[-2], times_ref[-2] - 50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    output_path = os.path.join(os.path.dirname(__file__), 'scaling_comparison.png')
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_scaling_benchmark()
