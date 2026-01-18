#!/opt/anaconda3/bin/python
"""
Benchmark attention-only seam carving compression (100% DINOv2 attention, no Sobel).

Usage:
    python benchmark.py --folder /path/to/images [--reduction 0.3] [--output output]
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Import compression functions from api.py
from api import (
    load_dino_model,
    get_attention_map,
    create_gray_overlay,
    seam_carve_image,
    IMAGE_SIZE
)

# Constants
DEFAULT_REDUCTION = 0.30  # 30% pixel reduction
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']


def get_image_files(folder):
    """Get all image files from folder"""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def run_benchmark(folder, reduction=DEFAULT_REDUCTION, threshold=0.3):
    """Run benchmark on all images in folder"""
    print("=" * 60)
    print("SEAM CARVING COMPRESSION BENCHMARK")
    print("=" * 60)
    print(f"Folder: {folder}")
    print(f"Target reduction: {reduction:.0%}")

    image_files = get_image_files(folder)
    if not image_files:
        print(f"\nNo images found in {folder}/")
        return None

    print(f"\nFound {len(image_files)} images\n")

    model = load_dino_model()
    results = []

    for img_path in tqdm(image_files, desc="Processing images"):
        filename = os.path.basename(img_path)

        try:
            # Load and resize
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            # Get attention and create visualizations
            attention_map = get_attention_map(model, img_resized)
            gray_overlay = create_gray_overlay(img_resized, attention_map, threshold)
            compressed_img = seam_carve_image(img_resized, attention_map, reduction)

            # Calculate stats
            original_pixels = IMAGE_SIZE * IMAGE_SIZE
            compressed_pixels = compressed_img.width * compressed_img.height
            actual_reduction = (1 - compressed_pixels / original_pixels) * 100

            results.append({
                'filename': filename,
                'original_size': original_size,
                'resized_size': img_resized.size,
                'compressed_size': compressed_img.size,
                'original_pixels': original_pixels,
                'compressed_pixels': compressed_pixels,
                'reduction_pct': actual_reduction,
                'original_image': img_resized,
                'gray_overlay': gray_overlay,
                'compressed_image': compressed_img,
                'attention_map': attention_map
            })

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            results.append({'filename': filename, 'error': str(e)})

    return results


def compute_cosine_similarity(img1, img2):
    """Compute cosine similarity between two images (resized to same dimensions)"""
    # Resize compressed to match original
    img2_resized = img2.resize(img1.size, Image.BILINEAR)

    # Flatten to vectors
    v1 = np.array(img1).flatten().astype(np.float32)
    v2 = np.array(img2_resized).flatten().astype(np.float32)

    # Cosine similarity
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    return dot / (norm1 * norm2 + 1e-8)


def print_summary(results, reduction):
    """Print summary statistics"""
    valid = [r for r in results if 'error' not in r]
    if not valid:
        print("\nAll images failed")
        return

    # Compute cosine similarities if not already done
    for r in valid:
        if 'cosine_sim' not in r:
            r['cosine_sim'] = compute_cosine_similarity(r['original_image'], r['compressed_image'])

    avg_reduction = np.mean([r['reduction_pct'] for r in valid])
    total_original = sum([r['original_pixels'] for r in valid])
    total_compressed = sum([r['compressed_pixels'] for r in valid])
    avg_similarity = np.mean([r['cosine_sim'] for r in valid])
    min_similarity = np.min([r['cosine_sim'] for r in valid])
    max_similarity = np.max([r['cosine_sim'] for r in valid])

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Images processed: {len(valid)}/{len(results)}")
    print(f"Target reduction: {reduction:.0%}")
    print(f"Actual reduction: {avg_reduction:.1f}%")
    print(f"Total pixels: {total_original:,} -> {total_compressed:,}")
    print(f"Cosine similarity: {avg_similarity:.4f} (min: {min_similarity:.4f}, max: {max_similarity:.4f})")
    print("=" * 60)


def save_output_images(results, output_folder='output'):
    """Save compressed images to output folder"""
    os.makedirs(output_folder, exist_ok=True)
    valid = [r for r in results if 'error' not in r]

    for r in valid:
        # Save compressed image
        name = os.path.splitext(r['filename'])[0]
        r['compressed_image'].save(os.path.join(output_folder, f"{name}_compressed.png"))

    print(f"\nSaved {len(valid)} compressed images to {output_folder}/")


def create_visualizations(results, output_folder='Graphs'):
    """Create visualization plots showing original, gray overlay, and compressed"""
    os.makedirs(output_folder, exist_ok=True)
    valid = [r for r in results if 'error' not in r]
    if not valid:
        return

    # Compute cosine similarities
    for r in valid:
        r['cosine_sim'] = compute_cosine_similarity(r['original_image'], r['compressed_image'])

    # Show sample comparisons (3 rows: original, gray overlay, compressed)
    n_show = min(6, len(valid))
    fig, axes = plt.subplots(3, n_show, figsize=(3*n_show, 9))

    for i in range(n_show):
        # Original
        axes[0, i].imshow(valid[i]['original_image'])
        axes[0, i].set_title(f"Original\n{valid[i]['resized_size'][0]}x{valid[i]['resized_size'][1]}", fontsize=9)
        axes[0, i].axis('off')

        # Gray overlay
        axes[1, i].imshow(valid[i]['gray_overlay'])
        axes[1, i].set_title(f"Attention Mask\n(gray=low attention)", fontsize=9)
        axes[1, i].axis('off')

        # Compressed
        axes[2, i].imshow(valid[i]['compressed_image'])
        axes[2, i].set_title(f"Compressed\n{valid[i]['compressed_size'][0]}x{valid[i]['compressed_size'][1]} (sim={valid[i]['cosine_sim']:.3f})", fontsize=9)
        axes[2, i].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Original', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Attention', fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel('Compressed', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'benchmark_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_folder}/benchmark_comparison.png")

    # Create cosine similarity distribution plot
    if len(valid) > 1:
        fig2, ax = plt.subplots(figsize=(8, 5))
        similarities = [r['cosine_sim'] for r in valid]
        ax.hist(similarities, bins=min(20, len(valid)), edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.4f}')
        ax.set_xlabel('Cosine Similarity to Original')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Cosine Similarity (Compressed vs Original)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'cosine_similarity.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {output_folder}/cosine_similarity.png")


def main():
    parser = argparse.ArgumentParser(description='Benchmark seam carving compression on folder of images')
    parser.add_argument('--folder', '-f', type=str, required=True, help='Folder containing images')
    parser.add_argument('--reduction', '-r', type=float, default=DEFAULT_REDUCTION, help=f'Target reduction (0-1, default: {DEFAULT_REDUCTION})')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Attention threshold for gray overlay (default: 0.3)')
    parser.add_argument('--graphs', '-g', type=str, default='Graphs', help='Output folder for visualizations')
    parser.add_argument('--output', '-o', type=str, default='output', help='Output folder for compressed images')

    args = parser.parse_args()

    results = run_benchmark(args.folder, args.reduction, args.threshold)
    if not results:
        return

    print_summary(results, args.reduction)
    print("\nSaving compressed images...")
    save_output_images(results, args.output)
    print("\nGenerating visualizations...")
    create_visualizations(results, args.graphs)
    print(f"\nDone! Graphs in {args.graphs}/, compressed images in {args.output}/")


if __name__ == "__main__":
    main()
