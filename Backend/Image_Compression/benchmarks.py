#!/opt/anaconda3/bin/python
"""
Benchmark attention-based image compression on CIFAR-10 dataset.

Usage:
    python benchmarks.py [--threshold 0.3] [--num-images 100]
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from run_prompt import (
    get_dino_model, get_attention_map, compress_image,
    DEFAULT_THRESHOLD, PATCH_SIZE, IMAGE_SIZE
)

CIFAR10_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_raw', 'cifar-10-batches-py')
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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


def load_cifar10_batch(batch_file):
    """Load a single CIFAR-10 batch file"""
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data']
    labels = batch[b'labels']
    # Reshape from (N, 3072) to (N, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_cifar10(num_images=100):
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


def run_benchmark(threshold=DEFAULT_THRESHOLD, num_images=100):
    """Run benchmark on CIFAR-10 images"""
    print("=" * 70)
    print("ATTENTION-BASED IMAGE COMPRESSION BENCHMARK (CIFAR-10)")
    print("=" * 70)
    print(f"Threshold: {threshold:.0%} | Images: {num_images}")

    images, labels = load_cifar10(num_images)
    if images is None:
        print(f"\n  No CIFAR-10 data found in {CIFAR10_PATH}/")
        return None

    print(f"\nLoaded {len(images)} images from CIFAR-10\n")

    model = get_dino_model()
    results = []

    for i, (img_array, label) in enumerate(zip(images, labels)):
        class_name = CIFAR10_CLASSES[label]
        img_name = f"{i:04d}_{class_name}"
        print(f"[{i+1}/{len(images)}] {img_name}", end=" ")

        try:
            # Convert to PIL and resize to model input size
            img_small = Image.fromarray(img_array)
            img = img_small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            attention_map, _ = get_attention_map(model, img)
            compressed_img, stats = compress_image(img, attention_map, threshold)
            similarity = calculate_similarity(model, img, compressed_img)

            pixels_kept = (1 - stats['token_reduction_pct'] / 100) * 100
            quality = "Excellent" if similarity > 0.95 else "Good" if similarity > 0.90 else "Acceptable" if similarity > 0.80 else "Poor"

            results.append({
                'filename': img_name,
                'class': class_name,
                'pixels_kept': pixels_kept,
                'similarity': similarity,
                'quality': quality,
                'compressed_image': compressed_img,
                'original_image': img
            })
            print(f"-> {pixels_kept:.1f}% kept | Sim: {similarity:.4f} | {quality}")

        except Exception as e:
            print(f"-> Error: {e}")
            results.append({'filename': img_name, 'class': class_name, 'error': str(e)})

    return results


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


def print_summary(stats, threshold):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Images: {stats['successful']}/{stats['total']} | Threshold: {threshold:.0%}")
    print(f"  Avg pixels kept: {stats['avg_pixels_kept']:.1f}%")
    print(f"  Avg similarity: {stats['avg_similarity']:.4f} (min: {stats['min_similarity']:.4f}, max: {stats['max_similarity']:.4f})")
    print(f"\n  Quality Distribution:")
    print(f"    Excellent (>0.95): {stats['excellent']}")
    print(f"    Good (0.90-0.95):  {stats['good']}")
    print(f"    Acceptable (0.80-0.90): {stats['acceptable']}")
    print(f"    Poor (<0.80):      {stats['poor']}")

    if stats['class_stats']:
        print(f"\n  Per-Class Results:")
        for class_name, cs in sorted(stats['class_stats'].items(), key=lambda x: -x[1]['avg_similarity']):
            print(f"    {class_name:12s}: Sim={cs['avg_similarity']:.4f} | Kept={cs['avg_pixels_kept']:.1f}% ({cs['count']} imgs)")
    print("=" * 70)


def create_visualizations(results, stats, threshold, output_folder='Graphs'):
    """Create visualization plots"""
    os.makedirs(output_folder, exist_ok=True)
    valid = [r for r in results if 'error' not in r]
    if not valid:
        return

    # Summary plot with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Quality distribution
    categories = ['Excellent', 'Good', 'Acceptable', 'Poor']
    counts = [stats['excellent'], stats['good'], stats['acceptable'], stats['poor']]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    axes[0].bar(categories, counts, color=colors)
    axes[0].set_title(f'Quality Distribution (Threshold: {threshold:.0%})')
    axes[0].set_ylabel('Count')

    # Per-class similarity
    if stats['class_stats']:
        classes = list(stats['class_stats'].keys())
        sims = [stats['class_stats'][c]['avg_similarity'] for c in classes]
        colors_class = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        axes[1].barh(classes, sims, color=colors_class)
        axes[1].set_xlabel('Avg Similarity')
        axes[1].set_title('Similarity by Class')
        axes[1].set_xlim(0.7, 1.0)
        axes[1].axvline(x=0.95, color='green', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0.90, color='orange', linestyle='--', alpha=0.5)

    # Scatter plot
    axes[2].scatter([r['pixels_kept'] for r in valid], [r['similarity'] for r in valid],
                    c=[r['similarity'] for r in valid], cmap='RdYlGn', s=40, vmin=0.7, vmax=1.0, alpha=0.7)
    axes[2].axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
    axes[2].axhline(y=0.90, color='orange', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Pixels Kept (%)')
    axes[2].set_ylabel('Similarity')
    axes[2].set_title('Compression vs Similarity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cifar10_benchmark_summary.png'), dpi=150)
    print(f"\nSaved: {output_folder}/cifar10_benchmark_summary.png")

    # Sample grid showing original vs compressed
    n_show = min(10, len(valid))
    if n_show > 0:
        fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))

        for i in range(n_show):
            axes[0, i].imshow(valid[i]['original_image'])
            axes[0, i].set_title(f"{valid[i]['class']}", fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(valid[i]['compressed_image'])
            axes[1, i].set_title(f"Sim: {valid[i]['similarity']:.3f}", fontsize=8)
            axes[1, i].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=10)
        axes[1, 0].set_ylabel('Compressed', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'cifar10_samples.png'), dpi=150)
        print(f"Saved: {output_folder}/cifar10_samples.png")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Benchmark image compression on CIFAR-10')
    parser.add_argument('--threshold', '-t', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Attention threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--num-images', '-n', type=int, default=100,
                        help='Number of images to test (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='Graphs',
                        help='Output folder for visualizations (default: Graphs)')
    args = parser.parse_args()

    results = run_benchmark(args.threshold, args.num_images)
    if not results:
        return

    stats = calculate_stats(results)
    if not stats:
        print("\nAll images failed")
        return

    print_summary(stats, args.threshold)
    print("\nGenerating visualizations...")
    create_visualizations(results, stats, args.threshold, args.output)
    print(f"\nDone! Results in {args.output}/")


if __name__ == "__main__":
    main()
