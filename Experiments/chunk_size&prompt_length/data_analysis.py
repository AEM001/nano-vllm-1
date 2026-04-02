"""
Data visualization for chunk_size & prompt_length experiment.
Analyzes TTFT and throughput metrics across different configurations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_data(filepath):
    """Load experiment results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_prompt_length(prompt_file):
    """Extract numeric prompt length from filename like '256_prompts.json'."""
    return int(prompt_file.split('_')[0])


def organize_data(data):
    """Organize data by chunk_size and prompt_length."""
    chunk_sizes = sorted(set(d['chunk_size'] for d in data))
    prompt_lengths = sorted(set(extract_prompt_length(d['prompt_file']) for d in data))
    
    # Create lookup: (chunk_size, prompt_length) -> metrics
    lookup = {}
    for d in data:
        pl = extract_prompt_length(d['prompt_file'])
        lookup[(d['chunk_size'], pl)] = {
            'ttft': d['avg_ttft'],
            'throughput': d['throughput_tokens_per_sec'],
            'min_ttft': d['min_ttft'],
            'max_ttft': d['max_ttft']
        }
    
    return chunk_sizes, prompt_lengths, lookup


def plot_ttft_by_chunk_size(data, output_dir):
    """Plot TTFT vs chunk size for each prompt length."""
    chunk_sizes, prompt_lengths, lookup = organize_data(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    for i, pl in enumerate(prompt_lengths):
        ttfts = [lookup[(cs, pl)]['ttft'] for cs in chunk_sizes]
        ax.plot(chunk_sizes, ttfts, marker=markers[i], linewidth=2, 
                markersize=8, label=f'{pl} token prompts', color=colors[i])
    
    ax.set_xlabel('Chunk Size', fontsize=12)
    ax.set_ylabel('TTFT (seconds)', fontsize=12)
    ax.set_title('Time to First Token (TTFT) vs Chunk Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(chunk_sizes)
    ax.set_xticklabels(chunk_sizes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ttft_by_chunk_size.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_throughput_by_chunk_size(data, output_dir):
    """Plot throughput vs chunk size for each prompt length."""
    chunk_sizes, prompt_lengths, lookup = organize_data(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    for i, pl in enumerate(prompt_lengths):
        throughputs = [lookup[(cs, pl)]['throughput'] for cs in chunk_sizes]
        ax.plot(chunk_sizes, throughputs, marker=markers[i], linewidth=2,
                markersize=8, label=f'{pl} token prompts', color=colors[i])
    
    ax.set_xlabel('Chunk Size', fontsize=12)
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_title('Throughput vs Chunk Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(chunk_sizes)
    ax.set_xticklabels(chunk_sizes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_by_chunk_size.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap(data, output_dir):
    """Create heatmaps for TTFT and throughput."""
    chunk_sizes, prompt_lengths, lookup = organize_data(data)
    
    # Prepare matrices
    ttft_matrix = np.array([[lookup[(cs, pl)]['ttft'] for pl in prompt_lengths] 
                           for cs in chunk_sizes])
    throughput_matrix = np.array([[lookup[(cs, pl)]['throughput'] for pl in prompt_lengths] 
                                  for cs in chunk_sizes])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # TTFT heatmap
    im1 = ax1.imshow(ttft_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(prompt_lengths)))
    ax1.set_yticks(range(len(chunk_sizes)))
    ax1.set_xticklabels([f'{pl}' for pl in prompt_lengths])
    ax1.set_yticklabels(chunk_sizes)
    ax1.set_xlabel('Prompt Length (tokens)', fontsize=11)
    ax1.set_ylabel('Chunk Size', fontsize=11)
    ax1.set_title('TTFT (seconds)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(chunk_sizes)):
        for j in range(len(prompt_lengths)):
            text = ax1.text(j, i, f'{ttft_matrix[i, j]:.2f}s',
                           ha='center', va='center', fontsize=9)
    
    plt.colorbar(im1, ax=ax1)
    
    # Throughput heatmap
    im2 = ax2.imshow(throughput_matrix, cmap='YlGn', aspect='auto')
    ax2.set_xticks(range(len(prompt_lengths)))
    ax2.set_yticks(range(len(chunk_sizes)))
    ax2.set_xticklabels([f'{pl}' for pl in prompt_lengths])
    ax2.set_yticklabels(chunk_sizes)
    ax2.set_xlabel('Prompt Length (tokens)', fontsize=11)
    ax2.set_ylabel('Chunk Size', fontsize=11)
    ax2.set_title('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(chunk_sizes)):
        for j in range(len(prompt_lengths)):
            text = ax2.text(j, i, f'{throughput_matrix[i, j]:.0f}',
                           ha='center', va='center', fontsize=9)
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bars(data, output_dir):
    """Create grouped bar chart comparing TTFT across configurations."""
    chunk_sizes, prompt_lengths, lookup = organize_data(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(chunk_sizes))
    width = 0.25
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # TTFT bars
    for i, pl in enumerate(prompt_lengths):
        ttfts = [lookup[(cs, pl)]['ttft'] for cs in chunk_sizes]
        ax1.bar(x + i * width, ttfts, width, label=f'{pl} tokens', color=colors[i])
    
    ax1.set_xlabel('Chunk Size', fontsize=11)
    ax1.set_ylabel('TTFT (seconds)', fontsize=11)
    ax1.set_title('TTFT Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(chunk_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Throughput bars
    for i, pl in enumerate(prompt_lengths):
        throughputs = [lookup[(cs, pl)]['throughput'] for cs in chunk_sizes]
        ax2.bar(x + i * width, throughputs, width, label=f'{pl} tokens', color=colors[i])
    
    ax2.set_xlabel('Chunk Size', fontsize=11)
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=11)
    ax2.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(chunk_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_analysis(data):
    """Print statistical analysis of the results."""
    chunk_sizes, prompt_lengths, lookup = organize_data(data)
    
    print("=" * 60)
    print("EXPERIMENT ANALYSIS")
    print("=" * 60)
    
    # Find best configurations
    print("\n1. BEST TTFT BY PROMPT LENGTH:")
    for pl in prompt_lengths:
        best_chunk = min(chunk_sizes, key=lambda cs: lookup[(cs, pl)]['ttft'])
        best_ttft = lookup[(best_chunk, pl)]['ttft']
        print(f"   {pl} tokens: chunk_size={best_chunk} → {best_ttft:.3f}s")
    
    print("\n2. BEST THROUGHPUT BY PROMPT LENGTH:")
    for pl in prompt_lengths:
        best_chunk = max(chunk_sizes, key=lambda cs: lookup[(cs, pl)]['throughput'])
        best_tp = lookup[(best_chunk, pl)]['throughput']
        print(f"   {pl} tokens: chunk_size={best_chunk} → {best_tp:.1f} tok/s")
    
    print("\n3. CHUNK SIZE IMPACT (256 vs 512+ for short prompts):")
    ttft_256 = lookup[(256, 256)]['ttft']
    ttft_512 = lookup[(512, 256)]['ttft']
    improvement = ((ttft_256 - ttft_512) / ttft_256) * 100
    print(f"   256-token prompts: {ttft_256:.3f}s → {ttft_512:.3f}s ({improvement:.1f}% faster)")
    
    print("\n4. SCALING WITH PROMPT LENGTH:")
    for cs in [512, 1024]:
        ttft_256 = lookup[(cs, 256)]['ttft']
        ttft_512 = lookup[(cs, 512)]['ttft']
        ttft_1024 = lookup[(cs, 1024)]['ttft']
        print(f"   chunk_size={cs}: 256t→512t {ttft_512/ttft_256:.1f}x, 256t→1024t {ttft_1024/ttft_256:.1f}x")
    
    print("\n5. RECOMMENDATION:")
    print("   Use chunk_size=768-1024 for best balance across all prompt lengths")
    print("=" * 60)


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_file = script_dir / 'experiment_results.json'
    output_dir = script_dir / 'charts'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_file}...")
    data = load_data(data_file)
    
    # Generate visualizations
    print("Generating charts...")
    plot_ttft_by_chunk_size(data, output_dir)
    plot_throughput_by_chunk_size(data, output_dir)
    plot_heatmap(data, output_dir)
    plot_comparison_bars(data, output_dir)
    
    # Print analysis
    print_analysis(data)
    
    print(f"\nCharts saved to: {output_dir}")
    print("  - ttft_by_chunk_size.png")
    print("  - throughput_by_chunk_size.png")
    print("  - performance_heatmaps.png")
    print("  - comparison_bars.png")


if __name__ == '__main__':
    main()
