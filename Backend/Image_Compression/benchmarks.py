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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from run_prompt import (
    get_dino_model, load_image, get_attention_map, compress_image,
    DEFAULT_THRESHOLD, PATCH_SIZE
)

IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
BENCHMARK_FOLDER = 'benchmark_images'


def get_embedding(model, img):
    """Get semantic embedding for similarity comparison"""
    from torchvision import transforms
    import torch

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        return model(img_tensor)


def calculate_similarity(model, img1, img2):
    """Calculate cosine similarity between two images"""
    emb1 = get_embedding(model, img1)
    emb2 = get_embedding(model, img2)
    emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
    emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
    return (emb1 @ emb2.T).item()


def get_image_files(folder):
    """Get all image files from folder"""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


def run_benchmark(threshold=DEFAULT_THRESHOLD):
    """Run benchmark on all images"""
    print("=" * 70)
    print("ATTENTION-BASED IMAGE COMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"Threshold: {threshold:.0%} | Folder: {BENCHMARK_FOLDER}/")

    image_files = get_image_files(BENCHMARK_FOLDER)
    if not image_files:
        print(f"\n⚠️  No images found in {BENCHMARK_FOLDER}/")
        return None

    print(f"\nFound {len(image_files)} images\n")

    model = get_dino_model()
    results = []

    for i, path in enumerate(image_files):
        filename = os.path.basename(path)
        print(f"[{i+1}/{len(image_files)}] {filename}", end=" ")

        try:
            img, _ = load_image(path)
            attention_map, _ = get_attention_map(model, img)
            compressed_img, stats = compress_image(img, attention_map, threshold)
            similarity = calculate_similarity(model, img, compressed_img)

            pixels_kept = (1 - stats['token_reduction_pct'] / 100) * 100
            quality = "Excellent" if similarity > 0.95 else "Good" if similarity > 0.90 else "Acceptable" if similarity > 0.80 else "Poor"

            results.append({
                'filename': filename,
                'pixels_kept': pixels_kept,
                'similarity': similarity,
                'quality': quality,
                'compressed_image': compressed_img
            })
            print(f"→ {pixels_kept:.1f}% kept | Sim: {similarity:.4f} | {quality}")

        except Exception as e:
            print(f"→ ❌ {e}")
            results.append({'filename': filename, 'error': str(e)})

    return results


def calculate_stats(results):
    """Calculate aggregate statistics"""
    valid = [r for r in results if 'error' not in r]
    if not valid:
        return None

    similarities = [r['similarity'] for r in valid]
    pixels_kept = [r['pixels_kept'] for r in valid]

    return {
        'total': len(results),
        'successful': len(valid),
        'avg_similarity': np.mean(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'avg_pixels_kept': np.mean(pixels_kept),
        'excellent': sum(1 for r in valid if r['similarity'] > 0.95),
        'good': sum(1 for r in valid if 0.90 < r['similarity'] <= 0.95),
        'acceptable': sum(1 for r in valid if 0.80 < r['similarity'] <= 0.90),
        'poor': sum(1 for r in valid if r['similarity'] <= 0.80),
    }


def print_summary(stats, threshold):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Images: {stats['successful']}/{stats['total']} | Threshold: {threshold:.0%}")
    print(f"  Avg pixels kept: {stats['avg_pixels_kept']:.1f}%")
    print(f"  Avg similarity: {stats['avg_similarity']:.4f} (min: {stats['min_similarity']:.4f}, max: {stats['max_similarity']:.4f})")
    print(f"\n  Quality: 🟢 Excellent: {stats['excellent']} | 🔵 Good: {stats['good']} | 🟡 Acceptable: {stats['acceptable']} | 🔴 Poor: {stats['poor']}")
    print("=" * 70)


def create_visualizations(results, stats, threshold, output_folder='benchmark_results'):
    """Create visualization plots"""
    os.makedirs(output_folder, exist_ok=True)
    valid = [r for r in results if 'error' not in r]
    if not valid:
        return

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Quality distribution
    categories = ['Excellent', 'Good', 'Acceptable', 'Poor']
    counts = [stats['excellent'], stats['good'], stats['acceptable'], stats['poor']]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    axes[0].bar(categories, counts, color=colors)
    axes[0].set_title(f'Quality Distribution (Threshold: {threshold:.0%})')
    axes[0].set_ylabel('Count')

    # Scatter plot
    axes[1].scatter([r['pixels_kept'] for r in valid], [r['similarity'] for r in valid],
                    c=[r['similarity'] for r in valid], cmap='RdYlGn', s=80, vmin=0.7, vmax=1.0)
    axes[1].axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Pixels Kept (%)')
    axes[1].set_ylabel('Similarity')
    axes[1].set_title('Compression vs Similarity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'benchmark_summary.png'), dpi=150)
    print(f"\n✓ Saved: {output_folder}/benchmark_summary.png")

    # Sample grid
    n_show = min(8, len(valid))
    if n_show > 0:
        cols = 4
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i in range(n_show):
            axes[i].imshow(valid[i]['compressed_image'])
            axes[i].set_title(f"{valid[i]['filename'][:10]}.. | {valid[i]['similarity']:.3f}", fontsize=8)
            axes[i].axis('off')
        for i in range(n_show, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'benchmark_samples.png'), dpi=150)
        print(f"✓ Saved: {output_folder}/benchmark_samples.png")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Benchmark image compression')
    parser.add_argument('--threshold', '-t', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--output', '-o', type=str, default='benchmark_results')
    args = parser.parse_args()

    results = run_benchmark(args.threshold)
    if not results:
        return

    stats = calculate_stats(results)
    if not stats:
        print("\n❌ All images failed")
        return

    print_summary(stats, args.threshold)
    print("\nGenerating visualizations...")
    create_visualizations(results, stats, args.threshold, args.output)
    print(f"\n✅ Done! Results in {args.output}/")


if __name__ == "__main__":
    main()
