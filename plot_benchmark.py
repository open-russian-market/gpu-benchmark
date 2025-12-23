import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import matplotlib.patches as mpatches

def parse_benchmark_file(filename):
    """Parse the benchmark data from the text file."""
    models_data = []

    # Try different encodings
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'cp1252']
    lines = None

    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    if lines is None:
        print("Could not decode file with any known encoding")
        return models_data

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for model name line
        if line.startswith('Model name:'):
            model_name = line.split(':')[1].strip()

            # Find the average time line (skip input size, output size, total time)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('Average time:'):
                i += 1

            if i >= len(lines):
                break

            # Extract average time
            avg_time_line = lines[i].strip()
            avg_time_match = re.search(r'Average time:\s*([\d.]+)\s*ms', avg_time_line)
            if avg_time_match:
                avg_time = float(avg_time_match.group(1))
                models_data.append((model_name, avg_time))

        i += 1

    return models_data

def get_model_family(model_name):
    """Categorize model into architecture families."""
    model_name = model_name.lower()

    # Define family mappings
    families = {
        'AlexNet': ['alexnet'],
        'ConvNeXt': ['convnext'],
        'DenseNet': ['densenet'],
        'EfficientNet': ['efficientnet'],
        'GoogLeNet': ['googlenet'],
        'Inception': ['inception'],
        'MaxViT': ['maxvit'],
        'MNASNet': ['mnasnet'],
        'MobileNet': ['mobilenet'],
        'RegNet': ['regnet'],
        'ResNet': ['resnet'],
        'ResNeXt': ['resnext'],
        'ShuffleNet': ['shufflenet'],
        'SqueezeNet': ['squeezenet'],
        'Swin': ['swin'],
        'VGG': ['vgg'],
        'ViT': ['vit'],
        'Wide ResNet': ['wide_resnet']
    }

    for family, keywords in families.items():
        if any(keyword in model_name for keyword in keywords):
            return family

    return 'Other'

def get_family_color(family):
    """Assign colors to different model families."""
    color_map = {
        'AlexNet': '#FF6B6B',      # Red
        'ConvNeXt': '#4ECDC4',     # Teal
        'DenseNet': '#45B7D1',     # Blue
        'EfficientNet': '#FFA07A', # Light Salmon
        'GoogLeNet': '#98D8C8',    # Mint
        'Inception': '#F7DC6F',    # Yellow
        'MaxViT': '#BB8FCE',       # Light Purple
        'MNASNet': '#85C1E9',      # Light Blue
        'MobileNet': '#F8C471',    # Orange
        'RegNet': '#82E0AA',       # Light Green
        'ResNet': '#E74C3C',       # Dark Red
        'ResNeXt': '#9B59B6',      # Purple
        'ShuffleNet': '#3498DB',   # Blue
        'SqueezeNet': '#E67E22',   # Orange
        'Swin': '#2ECC71',         # Green
        'VGG': '#1ABC9C',          # Turquoise
        'ViT': '#F39C12',          # Orange
        'Wide ResNet': '#D35400',  # Dark Orange
        'Other': '#95A5A6'         # Gray
    }

    return color_map.get(family, '#95A5A6')

def create_benchmark_plot(filename, save_path='benchmark_plot.png'):
    """Create and save a horizontal bar plot of benchmark results grouped by model family."""

    # Parse the data
    models_data = parse_benchmark_file(filename)

    if not models_data:
        print("No data found in file")
        return

    # Group by family
    family_data = defaultdict(list)
    for model_name, avg_time in models_data:
        family = get_model_family(model_name)
        family_data[family].append((model_name, avg_time))

    # Create plot with horizontal layout
    plt.figure(figsize=(14, 24))

    # Plot each family
    legend_elements = []
    current_y = 0
    y_tick_positions = []
    y_tick_labels = []

    # Sort families by average performance for better visualization
    family_avg_times = {}
    for family, models in family_data.items():
        avg_time = np.mean([time for _, time in models])
        family_avg_times[family] = avg_time

    sorted_families = sorted(family_avg_times.items(), key=lambda x: x[1])

    for family, _ in sorted_families:
        models = family_data[family]
        if not models:
            continue

        # Sort models within family by performance
        models.sort(key=lambda x: x[1])

        # Get color for this family
        color = get_family_color(family)

        # Create horizontal bars for this family
        for i, (model_name, avg_time) in enumerate(models):
            plt.barh(current_y + i, avg_time, height=0.8, color=color,
                    alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add model name as text on the bar
            plt.text(avg_time + 0.5, current_y + i, model_name,
                    fontsize=7, ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, edgecolor='none'))

        # Add family separator line
        if current_y > 0:
            plt.axhline(y=current_y - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Add family name as section header
        family_y = current_y + len(models) / 2 - 0.5
        plt.text(-1.0, family_y, family, fontsize=8, ha='right', va='center',
                fontweight='bold', rotation=90)

        # Update legend
        legend_elements.append(mpatches.Patch(color=color, label=f'{family} ({len(models)} models)'))

        current_y += len(models) + 3  # Add extra space between families

    # Customize plot
    plt.xlabel('Average Inference Time (ms)', fontsize=12)
    plt.ylabel('Models (Grouped by Architecture Family)', fontsize=12)
    plt.title('GTX 1070 Model Benchmark Results', fontsize=14, pad=20)

    # Set y-axis ticks and labels (remove them since we have text labels)
    plt.yticks([])

    # Add grid
    plt.grid(True, alpha=0.3, axis='x')

    # Set x-axis limit to ensure all text fits
    max_time = max([time for _, time in models_data])
    plt.xlim(0, max_time * 1.5)  # Give 50% extra space for text

    # Add legend at the bottom
    plt.legend(handles=legend_elements, loc='upper center', fontsize=9,
              bbox_to_anchor=(0.5, -0.05), borderaxespad=0., ncol=4)

    # Adjust layout to make room for legend and prevent text overlaps
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.18)

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {save_path}")

    # Close the plot without showing (no popup window)
    plt.close()

def create_summary_stats_plot(filename, save_path='benchmark_summary.png'):
    """Create a summary plot showing average performance by family."""

    # Parse the data
    models_data = parse_benchmark_file(filename)

    if not models_data:
        print("No data found in file")
        return

    # Group by family and calculate statistics
    family_stats = defaultdict(list)
    for model_name, avg_time in models_data:
        family = get_model_family(model_name)
        family_stats[family].append(avg_time)

    # Calculate summary statistics
    families = []
    avg_times = []
    min_times = []
    max_times = []
    model_counts = []

    for family, times in family_stats.items():
        families.append(family)
        avg_times.append(np.mean(times))
        min_times.append(np.min(times))
        max_times.append(np.max(times))
        model_counts.append(len(times))

    # Sort by average time
    sorted_indices = np.argsort(avg_times)
    families = [families[i] for i in sorted_indices]
    avg_times = [avg_times[i] for i in sorted_indices]
    min_times = [min_times[i] for i in sorted_indices]
    max_times = [max_times[i] for i in sorted_indices]
    model_counts = [model_counts[i] for i in sorted_indices]

    # Create plot
    plt.figure(figsize=(14, 8))

    # Create bars with error bars
    colors = [get_family_color(family) for family in families]
    bars = plt.bar(range(len(families)), avg_times, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=0.5)

    # Add error bars (min-max range)
    error_low = [avg - min_t for avg, min_t in zip(avg_times, min_times)]
    error_high = [max_t - avg for avg, max_t in zip(avg_times, max_times)]
    plt.errorbar(range(len(families)), avg_times, yerr=[error_low, error_high],
                fmt='none', ecolor='black', capsize=3, alpha=0.5)

    # Add value labels on bars
    for i, (bar, avg_time, count) in enumerate(zip(bars, avg_times, model_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{avg_time:.1f}ms\n({count} models)',
                ha='center', va='bottom', fontsize=9)

    # Customize plot
    plt.xlabel('Architecture Family', fontsize=12)
    plt.ylabel('Average Inference Time (ms)', fontsize=12)
    plt.title('GTX 1070 Benchmark Summary by Architecture Family', fontsize=14, pad=20)

    plt.xticks(range(len(families)), families, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add some statistics as text
    total_models = sum(model_counts)
    overall_avg = np.mean([time for _, time in models_data])
    fastest_time = min([time for _, time in models_data])
    slowest_time = max([time for _, time in models_data])

    stats_text = f"""Total Models: {total_models}
Overall Average: {overall_avg:.1f}ms
Fastest: {fastest_time:.1f}ms
Slowest: {slowest_time:.1f}ms"""

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved as: {save_path}")
    plt.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_benchmark.py <benchmark_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'benchmark_plot.png'

    # Create detailed plot
    create_benchmark_plot(input_file, output_file)
