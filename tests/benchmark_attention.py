import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add repo root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose.module import AltAttBlock, AltRefAttBlock

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_speed(model, input_tensor, iterations=100, warmup=10):
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

def run_benchmark():
    print("=== Benchmarking AltAtt vs AltRefAtt (CPU) ===")

    # Configuration
    dim = 256
    num_heads = 8
    batch_size = 4
    seq_len = 16
    num_tokens = 196 # e.g., 14x14 patches

    # Input tensor: (B, S, N, C)
    input_tensor = torch.randn(batch_size, seq_len, num_tokens, dim)

    print(f"Config: Dim={dim}, Heads={num_heads}, Batch={batch_size}, Seq={seq_len}, Tokens={num_tokens}")

    # Instantiate Models
    model_alt = AltAttBlock(dim=dim, num_heads=num_heads)
    model_ref = AltRefAttBlock(dim=dim, num_heads=num_heads)

    # Count Params
    params_alt = count_parameters(model_alt)
    params_ref = count_parameters(model_ref)

    print(f"AltAtt Params:    {params_alt:,}")
    print(f"AltRefAtt Params: {params_ref:,}")

    # Benchmark Speed
    print("Running speed test...")
    time_alt = benchmark_speed(model_alt, input_tensor)
    time_ref = benchmark_speed(model_ref, input_tensor)

    print(f"AltAtt Time:      {time_alt:.2f} ms")
    print(f"AltRefAtt Time:   {time_ref:.2f} ms")

    # Plotting
    labels = ['AltAtt', 'AltRefAtt']
    times = [time_alt, time_ref]
    params = [params_alt / 1e6, params_ref / 1e6] # Convert to Millions

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Speed
    ax[0].bar(labels, times, color=['blue', 'orange'])
    ax[0].set_title('Inference Speed (CPU)')
    ax[0].set_ylabel('Time per Batch (ms)')
    for i, v in enumerate(times):
        ax[0].text(i, v, f'{v:.1f} ms', ha='center', va='bottom')

    # Plot 2: Params
    ax[1].bar(labels, params, color=['blue', 'orange'])
    ax[1].set_title('Parameter Size')
    ax[1].set_ylabel('Parameters (Millions)')
    for i, v in enumerate(params):
        ax[1].text(i, v, f'{v:.2f} M', ha='center', va='bottom')

    plt.suptitle(f'Benchmark: Dim={dim}, Seq={seq_len}, Tokens={num_tokens}, Batch={batch_size}')
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
