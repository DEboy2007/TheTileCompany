"""
Basic usage example for Tile SDK.

This example demonstrates how to use the TileClient to compress images.
"""

import sys
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_sdk import TileClient


def main():
    """Run basic compression example."""
    # Initialize the client
    client = TileClient()

    print("Tile SDK Example - Image Compression")
    print("=" * 50)

    # Example 1: Basic compression
    print("\n1. Basic Compression:")
    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed.jpg"
        )
        print(f"   ✓ Saved to: {result['output_path']}")
        print(f"   ✓ Stats: {result['image_stats']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 2: Compression with custom threshold
    print("\n2. Compression with Custom Threshold:")
    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed_30.jpg",
            threshold=0.3
        )
        print(f"   ✓ Saved to: {result['output_path']}")
        print(f"   ✓ Stats: {result['image_stats']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 3: High compression
    print("\n3. High Compression:")
    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed_high.jpg",
            threshold=0.7,
            aggressiveness=3
        )
        print(f"   ✓ Saved to: {result['output_path']}")
        print(f"   ✓ Stats: {result['image_stats']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")


if __name__ == "__main__":
    main()
