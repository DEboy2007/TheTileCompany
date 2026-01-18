"""
API module for image compression.
Uses attention-guided seam carving.
"""

from .routes import create_app
from .service import compress_image

__all__ = [
    'create_app',
    'compress_image'
]
