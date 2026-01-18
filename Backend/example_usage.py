#!/usr/bin/env python
"""
Example usage of the refactored compression modules.
Demonstrates how to use compression and benchmarking separately.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


def example_compression():
    """Example: Basic compression usage"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Compression")
    print("=" * 70)

    from compression import compress_image, compress_prompt, load_image, get_dino_model, get_attention_map

    # Load and compress image
    print("\nCompressing image...")
    img, _ = load_image("https://images.unsplash.com/photo-1506905925346-21bda4d32df4")
    model = get_dino_model()
    attention_map, _ = get_attention_map(model, img)
    compressed_img, image_stats = compress_image(img, attention_map, threshold=0.3)

    print(f"Image compression: {image_stats['tokens_removed']} tokens removed ({image_stats['token_reduction_pct']:.1f}%)")

    # Compress prompt
    print("\nCompressing prompt...")
    prompt = "This is a beautiful mountain landscape with snow-capped peaks and clear blue skies."
    compressed_text, prompt_stats = compress_prompt(prompt, aggressiveness=0.5)

    print(f"Prompt compression: {prompt_stats['original_chars']} → {prompt_stats['compressed_chars']} chars ({prompt_stats['char_reduction_pct']:.1f}% saved)")
    print(f"Compressed: {compressed_text}")


def example_benchmarking():
    """Example: Benchmarking compression"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Benchmarking Compression")
    print("=" * 70)

    from benchmarks import evaluate_compression, benchmark_thresholds, generate_report

    # Evaluate compression with detailed metrics
    print("\nEvaluating compression performance...")
    metrics = evaluate_compression(
        image_path_or_url="https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
        prompt="What is in this image? Describe the scene in detail.",
        threshold=0.3,
        aggressiveness=0.5
    )

    # Generate and print report
    report = generate_report(metrics)
    print(report)

    # Benchmark multiple thresholds
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Threshold Benchmarking")
    print("=" * 70)

    print("\nTesting multiple thresholds...")
    results = benchmark_thresholds(
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
        thresholds=[0.2, 0.4, 0.6]
    )

    for key, stats in results.items():
        print(f"{key}: {stats['token_reduction_pct']:.1f}% reduction ({stats['compression_time_s']:.3f}s)")


def example_api_service():
    """Example: Using the API service layer"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: API Service Layer")
    print("=" * 70)

    from api import run_compression

    print("\nRunning full compression + LLM query...")
    result = run_compression(
        image_path_or_url="https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
        prompt="What is in this image?",
        threshold=0.3,
        aggressiveness=0.5
    )

    print(f"\nAnswer: {result['answer'][:200]}...")
    print(f"\nImage stats: {result['image_stats']['token_reduction_pct']:.1f}% reduction")
    print(f"Prompt stats: {result['prompt_stats']['char_reduction_pct']:.1f}% reduction")


if __name__ == "__main__":
    import sys

    # Check which example to run
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "compression":
            example_compression()
        elif example == "benchmark":
            example_benchmarking()
        elif example == "api":
            example_api_service()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python example_usage.py [compression|benchmark|api]")
    else:
        # Run all examples
        example_compression()
        example_benchmarking()
        # Uncomment to run API example (requires LLM API key)
        # example_api_service()
