"""
API module for compression endpoints.
"""

from .routes import create_app
from .service import run_compression

__all__ = [
    'create_app',
    'run_compression'
]
