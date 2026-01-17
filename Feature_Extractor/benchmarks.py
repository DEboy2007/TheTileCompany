#!/opt/anaconda3/bin/python
"""
Benchmark attention-based image compression on multiple images.
Runs on all images in benchmark_images/ folder.

Usage:
    python benchmarks.py [--threshold 0.3]
"""

import argparse
import os
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from compress_image import compress, load_model, DEFAULT_THRESHOLD

# Supported image formats
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
BENCHMARK_FOLDER = 'benchmark_images'


def get_image_files(folder):
    """Get all image files from folder"""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


def run_benchmark(threshold=DEFAULT_THRESHOLD):
    """Run benchmark on all images in benchmark_images folder"""

    print("=" * 70)
    print("ATTENTION-BASED IMAGE COMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"Threshold: {threshold:.0%}")
    print(f"Folder: {BENCHMARK_FOLDER}/")
    print("=" * 70)

    # Get image files
    image_files = get_image_files(BENCHMARK_FOLDER)

    if len(image_files) == 0:
        print(f"\n⚠️  No images found in {BENCHMARK_FOLDER}/")
        print(f"   Add some images (.jpg, .png, .webp) to the folder and run again.")
        return None

    print(f"\nFound {len(image_files)} images\n")

    # Preload model
    print("Loading model...")
    load_model()
    print()

    # Run compression on each image
    results = []

    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        print(f"[{i+1}/{len(image_files)}] Processing: {filename}")

        try:
            result = compress(image_path, threshold=threshold, verbose=False)
            stats = result['stats']

            results.append({
                'filename': filename,
                'path': image_path,
                'pixels_kept': stats['pixels_kept'],
                'pixels_removed': stats['pixels_removed'],
                'similarity': stats['similarity'],
                'quality': stats['quality'],
                'original_image': result['original_image'],
                'compressed_image': result['compressed_image'],
                'attention_map': result['attention_map']
            })

            print(f"         Pixels kept: {stats['pixels_kept']:.1f}% | Similarity: {stats['similarity']:.4f} | {stats['quality']}")

        except Exception as e:
            print(f"         ❌ Error: {e}")
            results.append({
                'filename': filename,
                'path': image_path,
                'error': str(e)
            })

    return results


def calculate_stats(results):
    """Calculate aggregate statistics"""

    # Filter successful results
    valid = [r for r in results if 'error' not in r]

    if len(valid) == 0:
        return None

    similarities = [r['similarity'] for r in valid]
    pixels_kept = [r['pixels_kept'] for r in valid]

    stats = {
        'total_images': len(results),
        'successful': len(valid),
        'failed': len(results) - len(valid),
        'avg_similarity': np.mean(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'std_similarity': np.std(similarities),
        'avg_pixels_kept': np.mean(pixels_kept),
        'avg_compression': 100 - np.mean(pixels_kept),
        'excellent_count': sum(1 for r in valid if r['similarity'] > 0.95),
        'good_count': sum(1 for r in valid if 0.90 < r['similarity'] <= 0.95),
        'acceptable_count': sum(1 for r in valid if 0.80 < r['similarity'] <= 0.90),
        'poor_count': sum(1 for r in valid if r['similarity'] <= 0.80),
    }

    return stats


def print_summary(stats, threshold):
    """Print summary statistics"""

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n📊 OVERALL STATS:")
    print(f"   Images tested:     {stats['successful']}/{stats['total_images']}")
    print(f"   Threshold:         {threshold:.0%}")
    print(f"   Avg compression:   {stats['avg_compression']:.1f}% pixels removed")
    print(f"   Avg pixels kept:   {stats['avg_pixels_kept']:.1f}%")

    print(f"\n🎯 SIMILARITY SCORES:")
    print(f"   Average:           {stats['avg_similarity']:.4f}")
    print(f"   Min:               {stats['min_similarity']:.4f}")
    print(f"   Max:               {stats['max_similarity']:.4f}")
    print(f"   Std Dev:           {stats['std_similarity']:.4f}")

    print(f"\n📈 QUALITY BREAKDOWN:")
    print(f"   Excellent (>0.95): {stats['excellent_count']} ({stats['excellent_count']/stats['successful']*100:.1f}%)")
    print(f"   Good (0.90-0.95):  {stats['good_count']} ({stats['good_count']/stats['successful']*100:.1f}%)")
    print(f"   Acceptable (0.80-0.90): {stats['acceptable_count']} ({stats['acceptable_count']/stats['successful']*100:.1f}%)")
    print(f"   Poor (<0.80):      {stats['poor_count']} ({stats['poor_count']/stats['successful']*100:.1f}%)")

    print("=" * 70)


def create_visualizations(results, stats, threshold, output_folder='benchmark_results'):
    """Create visualization plots"""

    os.makedirs(output_folder, exist_ok=True)

    valid = [r for r in results if 'error' not in r]

    if len(valid) == 0:
        return

    # 1. Summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Quality distribution
    categories = ['Excellent\n(>0.95)', 'Good\n(0.90-0.95)', 'Acceptable\n(0.80-0.90)', 'Poor\n(<0.80)']
    counts = [stats['excellent_count'], stats['good_count'], stats['acceptable_count'], stats['poor_count']]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    axes[0].bar(categories, counts, color=colors)
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title(f'Quality Distribution (Threshold: {threshold:.0%})')
    for i, (cat, count) in enumerate(zip(categories, counts)):
        axes[0].text(i, count + 0.1, str(count), ha='center', fontsize=12, fontweight='bold')

    # Similarity distribution
    similarities = [r['similarity'] for r in valid]
    axes[1].hist(similarities, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
    axes[1].axvline(x=0.95, color='#2ecc71', linestyle='--', label='Excellent (0.95)')
    axes[1].axvline(x=0.90, color='#f39c12', linestyle='--', label='Good (0.90)')
    axes[1].axvline(x=0.80, color='#e74c3c', linestyle='--', label='Acceptable (0.80)')
    axes[1].axvline(x=stats['avg_similarity'], color='black', linestyle='-', linewidth=2, label=f"Avg ({stats['avg_similarity']:.3f})")
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Similarity Score Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'benchmark_summary.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_folder}/benchmark_summary.png")

    # 2. Individual results scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    filenames = [r['filename'][:15] + '...' if len(r['filename']) > 15 else r['filename'] for r in valid]
    similarities = [r['similarity'] for r in valid]
    pixels_kept = [r['pixels_kept'] for r in valid]

    scatter = ax.scatter(pixels_kept, similarities, c=similarities, cmap='RdYlGn', s=100, vmin=0.7, vmax=1.0)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Pixels Kept (%)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Compression vs Similarity (Threshold: {threshold:.0%})')
    plt.colorbar(scatter, label='Similarity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'benchmark_scatter.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/benchmark_scatter.png")

    # 3. Grid of compressed images (up to 12)
    n_show = min(12, len(valid))
    cols = 4
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if n_show > 1 else [axes]

    for i in range(n_show):
        r = valid[i]
        axes[i].imshow(r['compressed_image'])
        axes[i].set_title(f"{r['filename'][:12]}...\nSim: {r['similarity']:.3f} | {r['pixels_kept']:.0f}%", fontsize=9)
        axes[i].axis('off')

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'benchmark_samples.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/benchmark_samples.png")

    # 4. Per-image results table
    fig, ax = plt.subplots(figsize=(12, max(4, len(valid) * 0.4)))
    ax.axis('off')

    table_data = []
    for r in valid:
        quality_color = {'Excellent': '🟢', 'Good': '🔵', 'Acceptable': '🟡', 'Poor': '🔴'}
        table_data.append([
            r['filename'][:25],
            f"{r['pixels_kept']:.1f}%",
            f"{100-r['pixels_kept']:.1f}%",
            f"{r['similarity']:.4f}",
            f"{quality_color.get(r['quality'], '')} {r['quality']}"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Image', 'Pixels Kept', 'Compression', 'Similarity', 'Quality'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title(f'Benchmark Results - Threshold: {threshold:.0%}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'benchmark_table.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/benchmark_table.png")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Benchmark attention-based image compression')
    parser.add_argument('--threshold', '-t', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Attention threshold (0.0-1.0), default {DEFAULT_THRESHOLD}')
    parser.add_argument('--output', '-o', type=str, default='benchmark_results',
                        help='Output folder for visualizations')

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(threshold=args.threshold)

    if results is None or len(results) == 0:
        return

    # Calculate stats
    stats = calculate_stats(results)

    if stats is None:
        print("\n❌ All images failed to process")
        return

    # Print summary
    print_summary(stats, args.threshold)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, stats, args.threshold, args.output)

    print(f"\n✅ Benchmark complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
