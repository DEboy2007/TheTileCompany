"""
Image compression module using attention-based pruning with DINOv2.
"""

from .compressor import (
    compress_image,
    load_image,
    get_attention_map,
    image_to_base64
)
from .model import get_dino_model

__all__ = [
    'compress_image',
    'load_image',
    'get_attention_map',
    'image_to_base64',
    'get_dino_model'
]
