"""
Basic usage example for Tile SDK.

This example demonstrates how to use the TileClient to compress images.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python_sdk import TileClient


def main():
    """Run basic compression example."""
    # Initialize the client
    client = TileClient()

    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed.jpg"
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed_30.jpg",
            threshold=0.3
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed_high.jpg",
            threshold=0.7,
            aggressiveness=3
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")


if __name__ == "__main__":
    main()
