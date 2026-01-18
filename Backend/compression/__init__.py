"""
Compression package containing image and prompt compression modules.
"""

from .image_compression import (
    compress_image,
    load_image,
    get_attention_map,
    image_to_base64,
    get_dino_model
)
from .prompt_compression import compress_prompt, get_token_client

__all__ = [
    'compress_image',
    'load_image',
    'get_attention_map',
    'image_to_base64',
    'get_dino_model',
    'compress_prompt',
    'get_token_client'
]
