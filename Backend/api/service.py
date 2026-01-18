"""
Image compression service - attention-guided seam carving.
"""

import sys
import os

# Add parent directory to import compression_api
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from compression_api import compress, image_to_base64


def compress_image(image_path_or_url, reduction=0.3, threshold=0.3):
    """
    Compress image using attention-guided seam carving.

    Args:
        image_path_or_url: Path or URL to image
        reduction: Target pixel reduction (0-1)
        threshold: Attention threshold for gray overlay

    Returns:
        dict with:
            - reduction_pct: Percentage of pixels saved
            - gray_overlay_base64: Base64 encoded gray overlay image
            - compressed_image_base64: Base64 encoded compressed image
            - stats: Additional statistics
    """
    result = compress(
        image_path_or_url=image_path_or_url,
        reduction=reduction,
        threshold=threshold
    )

    return {
        'reduction_pct': result['reduction_pct'],
        'gray_overlay_base64': image_to_base64(result['gray_overlay']),
        'compressed_image_base64': image_to_base64(result['compressed_image']),
        'stats': result['stats']
    }
