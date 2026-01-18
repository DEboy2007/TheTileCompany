#!/opt/anaconda3/bin/python
"""
Benchmark attention-based image compression on CIFAR-10 dataset.
Tests 3 compression levels: 95%, 80%, and 70% pixel preservation.

Usage:
    python benchmarks.py
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from attention_extractor import (
    load_model as get_dino_model,
    get_attention_maps,
    create_pruned_image,
    get_model_embedding,
    compare_embeddings,
    DEFAULT_THRESHOLD
)

# Constants
PATCH_SIZE = 14
IMAGE_SIZE = 518

CIFAR10_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_raw', 'cifar-10-batches-py')
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
NUM_IMAGES = 100
COMPRESSION_TARGETS = [0.95, 0.80, 0.70]  # Preserve 95%, 80%, 70% of pixels


def get_attention_map(model, img):
    """Get attention map from DINOv2 model (wrapper for compatibility)"""
    attentions, patch_grid = get_attention_maps(model, img)

    # Use last layer attention, average over heads, get CLS attention to patches
    attn = attentions[-1]  # Last layer
    cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # Average over heads, CLS to patches

    # Reshape to spatial dimensions
    w_patches, h_patches = patch_grid
    cls_attn = cls_attn.reshape(h_patches, w_patches).cpu().numpy()

    # Resize to image dimensions
    from PIL import Image as PILImage
    cls_attn_img = PILImage.fromarray(cls_attn)
    cls_attn_resized = np.array(cls_attn_img.resize(img.size, PILImage.BILINEAR))

    # Normalize to 0-1
    cls_attn_resized = (cls_attn_resized - cls_attn_resized.min()) / (cls_attn_resized.max() - cls_attn_resized.min() + 1e-8)

    return cls_attn_resized, patch_grid


def compress_image(img, attention_map, threshold=DEFAULT_THRESHOLD):
    """Compress image by masking low-attention regions (wrapper for compatibility)"""
    compressed_img, mask = create_pruned_image(img, attention_map, threshold)

    # Calculate token savings
    total_patches = (img.size[0] // PATCH_SIZE) * (img.size[1] // PATCH_SIZE)
    h_patches = img.size[1] // PATCH_SIZE
    w_patches = img.size[0] // PATCH_SIZE
    patch_attention = attention_map.reshape(h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean(axis=(1, 3))
    tokens_kept = (patch_attention >= threshold).sum()
    tokens_removed = total_patches - tokens_kept

    return compressed_img, {
        'total_tokens': total_patches,
        'tokens_kept': int(tokens_kept),
        'tokens_removed': int(tokens_removed),
        'token_reduction_pct': tokens_removed / total_patches * 100
    }


def calculate_similarity(model, img1, img2):
    """Calculate cosine similarity between two images"""
    emb1 = get_model_embedding(model, img1)
    emb2 = get_model_embedding(model, img2)
    return compare_embeddings(emb1, emb2)


def load_cifar10_batch(batch_file):
    """Load a single CIFAR-10 batch file"""
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data']
    labels = batch[b'labels']
    # Reshape from (N, 3072) to (N, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_cifar10(num_images=NUM_IMAGES):
    """Load CIFAR-10 images, sampling evenly from all batches"""
    all_images = []
    all_labels = []

    # Load from all 5 training batches
    for i in range(1, 6):
        batch_path = os.path.join(CIFAR10_PATH, f'data_batch_{i}')
        if os.path.exists(batch_path):
            images, labels = load_cifar10_batch(batch_path)
            all_images.append(images)
            all_labels.extend(labels)

    if not all_images:
        return None, None

    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)

    # Sample evenly across dataset
    if num_images < len(all_images):
        indices = np.linspace(0, len(all_images) - 1, num_images, dtype=int)
        all_images = all_images[indices]
        all_labels = all_labels[indices]

    return all_images, all_labels


def calculate_threshold_for_target(attention_map, target_preservation):
    """
    Calculate the attention threshold needed to preserve a target percentage of pixels.

    Args:
        attention_map: 2D numpy array of attention values
        target_preservation: float between 0 and 1 (e.g., 0.95 for 95%)

    Returns:
        threshold value that preserves the target percentage
    """
    # Flatten and sort attention values
    flat_attention = attention_map.flatten()
    sorted_attention = np.sort(flat_attention)

    # Find the threshold at the target percentile
    # Lower attention values will be masked, so we want the (1 - target) percentile
    percentile = (1 - target_preservation) * 100
    threshold = np.percentile(sorted_attention, percentile)

    return threshold


def run_benchmark():
    """Run benchmark on 100 CIFAR-10 images with 3 compression targets"""
    print("=" * 70)
    print("ATTENTION-BASED IMAGE COMPRESSION BENCHMARK (CIFAR-10)")
    print("=" * 70)
    print(f"Images: {NUM_IMAGES}")
    print(f"Compression targets: {', '.join([f'{t:.0%}' for t in COMPRESSION_TARGETS])}")

    images, labels = load_cifar10()
    if images is None:
        print(f"\nNo CIFAR-10 data found in {CIFAR10_PATH}/")
        return None

    print(f"\nLoaded {len(images)} images from CIFAR-10\n")

    model = get_dino_model()

    # Store results for each compression level
    all_results = {target: [] for target in COMPRESSION_TARGETS}

    for i, (img_array, label) in enumerate(zip(images, labels)):
        class_name = CIFAR10_CLASSES[label]
        img_name = f"{i:04d}_{class_name}"
        print(f"[{i+1}/{len(images)}] {img_name}")

        try:
            # Convert to PIL and resize to model input size
            img_small = Image.fromarray(img_array)
            img = img_small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            # Get attention map once
            attention_map, _ = get_attention_map(model, img)

            # Test each compression target
            for target in COMPRESSION_TARGETS:
                # Calculate threshold for this target
                threshold = calculate_threshold_for_target(attention_map, target)

                # Compress image
                compressed_img, stats = compress_image(img, attention_map, threshold)
                similarity = calculate_similarity(model, img, compressed_img)

                pixels_kept = (1 - stats['token_reduction_pct'] / 100) * 100

                all_results[target].append({
                    'filename': img_name,
                    'class': class_name,
                    'target': target,
                    'threshold': threshold,
                    'pixels_kept': pixels_kept,
                    'similarity': similarity,
                    'compressed_image': compressed_img,
                    'original_image': img
                })

                print(f"  {target:.0%} target: {pixels_kept:.1f}% kept | Sim: {similarity:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            for target in COMPRESSION_TARGETS:
                all_results[target].append({
                    'filename': img_name,
                    'class': class_name,
                    'target': target,
                    'error': str(e)
                })

    return all_results


def calculate_stats(results):
    """Calculate aggregate statistics"""
    valid = [r for r in results if 'error' not in r]
    if not valid:
        return None

    similarities = [r['similarity'] for r in valid]
    pixels_kept = [r['pixels_kept'] for r in valid]

    # Per-class statistics
    class_stats = {}
    for class_name in CIFAR10_CLASSES:
        class_results = [r for r in valid if r['class'] == class_name]
        if class_results:
            class_stats[class_name] = {
                'count': len(class_results),
                'avg_similarity': np.mean([r['similarity'] for r in class_results]),
                'avg_pixels_kept': np.mean([r['pixels_kept'] for r in class_results])
            }

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
        'class_stats': class_stats
    }


def print_summary(all_stats):
    """Print summary statistics for all compression targets"""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for target in COMPRESSION_TARGETS:
        stats = all_stats[target]
        print(f"\n{target:.0%} PIXEL PRESERVATION TARGET:")
        print(f"  Images: {stats['successful']}/{stats['total']}")
        print(f"  Actual avg pixels kept: {stats['avg_pixels_kept']:.1f}%")
        print(f"  Avg similarity: {stats['avg_similarity']:.4f}")
        print(f"  Similarity range: {stats['min_similarity']:.4f} - {stats['max_similarity']:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON ACROSS COMPRESSION LEVELS:")
    print("=" * 70)
    print(f"{'Target':<10} {'Avg Pixels':<12} {'Avg Similarity':<15} {'Excellent':<10} {'Good':<10} {'Poor':<10}")
    print("-" * 70)
    for target in COMPRESSION_TARGETS:
        stats = all_stats[target]
        print(f"{target:.0%}        {stats['avg_pixels_kept']:>6.1f}%      {stats['avg_similarity']:>8.4f}        {stats['excellent']:>4}       {stats['good']:>4}       {stats['poor']:>4}")
    print("=" * 70)


def create_visualizations(all_results, all_stats, output_folder='Graphs'):
    """Create comparative visualization plots across compression targets"""
    os.makedirs(output_folder, exist_ok=True)

    # Comparison plot - 3 subplots comparing the compression levels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Similarity comparison across targets
    targets_pct = [f'{t:.0%}' for t in COMPRESSION_TARGETS]
    avg_sims = [all_stats[t]['avg_similarity'] for t in COMPRESSION_TARGETS]
    colors = ['#3498db', '#f39c12', '#e74c3c']

    axes[0].bar(targets_pct, avg_sims, color=colors)
    axes[0].set_ylabel('Average Similarity')
    axes[0].set_xlabel('Pixel Preservation Target')
    axes[0].set_title('Average Similarity by Compression Level')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Excellent')
    axes[0].axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='Good')
    axes[0].legend()

    # 2. Quality distribution across targets
    width = 0.25
    x = np.arange(len(COMPRESSION_TARGETS))

    excellent_counts = [all_stats[t]['excellent'] for t in COMPRESSION_TARGETS]
    good_counts = [all_stats[t]['good'] for t in COMPRESSION_TARGETS]
    poor_counts = [all_stats[t]['poor'] for t in COMPRESSION_TARGETS]

    axes[1].bar(x - width, excellent_counts, width, label='Excellent (>0.95)', color='#2ecc71')
    axes[1].bar(x, good_counts, width, label='Good (0.90-0.95)', color='#3498db')
    axes[1].bar(x + width, poor_counts, width, label='Poor (<0.90)', color='#e74c3c')

    axes[1].set_xlabel('Pixel Preservation Target')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Quality Distribution by Compression Level')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(targets_pct)
    axes[1].legend()

    # 3. Actual compression achieved
    avg_pixels = [all_stats[t]['avg_pixels_kept'] for t in COMPRESSION_TARGETS]
    target_pixels = [t * 100 for t in COMPRESSION_TARGETS]

    x_pos = np.arange(len(COMPRESSION_TARGETS))
    axes[2].bar(x_pos - 0.2, target_pixels, 0.4, label='Target', color='#95a5a6', alpha=0.7)
    axes[2].bar(x_pos + 0.2, avg_pixels, 0.4, label='Actual', color='#3498db')

    axes[2].set_xlabel('Compression Level')
    axes[2].set_ylabel('Pixels Kept (%)')
    axes[2].set_title('Target vs Actual Pixel Preservation')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(targets_pct)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'compression_comparison.png'), dpi=150)
    print(f"\nSaved: {output_folder}/compression_comparison.png")

    # Sample grid showing original vs compressed at all 3 levels
    n_show = min(5, len([r for r in all_results[COMPRESSION_TARGETS[0]] if 'error' not in r]))
    if n_show > 0:
        fig, axes = plt.subplots(4, n_show, figsize=(3 * n_show, 12))

        for i in range(n_show):
            # Original image
            result_95 = all_results[COMPRESSION_TARGETS[0]][i]
            axes[0, i].imshow(result_95['original_image'])
            axes[0, i].set_title(f"{result_95['class']}", fontsize=10)
            axes[0, i].axis('off')

            # Compressed at each level
            for j, target in enumerate(COMPRESSION_TARGETS):
                result = all_results[target][i]
                if 'error' not in result:
                    axes[j+1, i].imshow(result['compressed_image'])
                    axes[j+1, i].set_title(f"Sim: {result['similarity']:.3f}", fontsize=10)
                    axes[j+1, i].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=12, rotation=0, ha='right', va='center')
        for j, target in enumerate(COMPRESSION_TARGETS):
            axes[j+1, 0].set_ylabel(f'{target:.0%}', fontsize=12, rotation=0, ha='right', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'compression_samples.png'), dpi=150)
        print(f"Saved: {output_folder}/compression_samples.png")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description=f'Benchmark image compression on {NUM_IMAGES} CIFAR-10 images '
                    f'with {len(COMPRESSION_TARGETS)} compression targets: '
                    f'{", ".join([f"{t:.0%}" for t in COMPRESSION_TARGETS])}'
    )
    parser.add_argument('--output', '-o', type=str, default='Graphs',
                        help='Output folder for visualizations (default: Graphs)')
    args = parser.parse_args()

    # Run benchmark on all compression targets
    all_results = run_benchmark()
    if not all_results:
        return

    # Calculate stats for each compression target
    all_stats = {}
    for target in COMPRESSION_TARGETS:
        stats = calculate_stats(all_results[target])
        if not stats:
            print(f"\nAll images failed for {target:.0%} target")
            return
        all_stats[target] = stats

    # Print summary
    print_summary(all_stats)

    # Generate visualizations
    print("\nGenerating visualizations...")
    create_visualizations(all_results, all_stats, args.output)
    print(f"\nDone! Results in {args.output}/")


if __name__ == "__main__":
    main()
