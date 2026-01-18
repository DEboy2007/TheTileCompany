"""
Compression evaluation and benchmarking utilities.
"""

import time
from typing import List, Dict, Any
import numpy as np

from compression.image_compression import (
    load_image,
    get_attention_map,
    compress_image,
    get_dino_model
)
from compression.prompt_compression import compress_prompt


def evaluate_compression(image_path_or_url, prompt, threshold=0.3, aggressiveness=0.5):
    """
    Evaluate compression performance with detailed metrics.

    Args:
        image_path_or_url: Path or URL to image
        prompt: Text prompt
        threshold: Attention threshold for image compression
        aggressiveness: Text compression aggressiveness

    Returns:
        dict with performance metrics
    """
    metrics = {}

    # Image compression metrics
    start_time = time.time()
    img, original_size = load_image(image_path_or_url)
    model = get_dino_model()
    attention_map, _ = get_attention_map(model, img)
    compressed_img, image_stats = compress_image(img, attention_map, threshold)
    image_time = time.time() - start_time

    metrics['image'] = {
        **image_stats,
        'compression_time_s': image_time,
        'original_size': original_size,
        'resized_size': img.size
    }

    # Prompt compression metrics
    start_time = time.time()
    compressed_prompt, prompt_stats = compress_prompt(prompt, aggressiveness)
    prompt_time = time.time() - start_time

    metrics['prompt'] = {
        **prompt_stats,
        'compression_time_s': prompt_time
    }

    # Total metrics
    metrics['total'] = {
        'total_time_s': image_time + prompt_time,
        'estimated_token_reduction_pct': (
            (image_stats['tokens_removed'] +
             (prompt_stats['original_tokens_est'] - prompt_stats['compressed_tokens_est'])) /
            (image_stats['total_tokens'] + prompt_stats['original_tokens_est']) * 100
        )
    }

    return metrics


def benchmark_thresholds(image_path_or_url, thresholds: List[float] = None):
    """
    Benchmark image compression across different threshold values.

    Args:
        image_path_or_url: Path or URL to image
        thresholds: List of threshold values to test

    Returns:
        dict with results for each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    img, original_size = load_image(image_path_or_url)
    model = get_dino_model()
    attention_map, _ = get_attention_map(model, img)

    results = {}
    for threshold in thresholds:
        start_time = time.time()
        compressed_img, stats = compress_image(img, attention_map, threshold)
        compression_time = time.time() - start_time

        results[f'threshold_{threshold}'] = {
            **stats,
            'compression_time_s': compression_time,
            'threshold': threshold
        }

    return results


def benchmark_aggressiveness(prompt, aggressiveness_levels: List[float] = None):
    """
    Benchmark prompt compression across different aggressiveness values.

    Args:
        prompt: Text prompt to compress
        aggressiveness_levels: List of aggressiveness values to test

    Returns:
        dict with results for each aggressiveness level
    """
    if aggressiveness_levels is None:
        aggressiveness_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = {}
    for aggressiveness in aggressiveness_levels:
        start_time = time.time()
        compressed_prompt, stats = compress_prompt(prompt, aggressiveness)
        compression_time = time.time() - start_time

        results[f'aggressiveness_{aggressiveness}'] = {
            **stats,
            'compression_time_s': compression_time,
            'aggressiveness': aggressiveness,
            'compressed_text': compressed_prompt
        }

    return results


def generate_report(metrics: Dict[str, Any]) -> str:
    """
    Generate a human-readable report from compression metrics.

    Args:
        metrics: Metrics dict from evaluate_compression

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("COMPRESSION EVALUATION REPORT")
    report.append("=" * 70)

    # Image metrics
    img = metrics['image']
    report.append("\nIMAGE COMPRESSION:")
    report.append(f"  Original tokens:    {img['total_tokens']}")
    report.append(f"  Compressed tokens:  {img['tokens_kept']}")
    report.append(f"  Tokens saved:       {img['tokens_removed']} ({img['token_reduction_pct']:.1f}%)")
    report.append(f"  Compression time:   {img['compression_time_s']:.3f}s")

    # Prompt metrics
    pmt = metrics['prompt']
    report.append("\nPROMPT COMPRESSION:")
    report.append(f"  Original chars:     {pmt['original_chars']}")
    report.append(f"  Compressed chars:   {pmt['compressed_chars']}")
    report.append(f"  Chars saved:        {pmt['original_chars'] - pmt['compressed_chars']} ({pmt['char_reduction_pct']:.1f}%)")
    report.append(f"  Compression time:   {pmt['compression_time_s']:.3f}s")

    # Total metrics
    total = metrics['total']
    report.append("\nTOTAL:")
    report.append(f"  Total time:         {total['total_time_s']:.3f}s")
    report.append(f"  Est. token reduction: {total['estimated_token_reduction_pct']:.1f}%")

    report.append("=" * 70)

    return "\n".join(report)
