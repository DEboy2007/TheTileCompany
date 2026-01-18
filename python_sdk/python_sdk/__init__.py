"""Tile SDK for image compression."""

from .tile_client import (
    TileClient,
    TileClientError,
    TileAPIError,
    TileConnectionError,
)

__version__ = "0.1.0"
__all__ = [
    "TileClient",
    "TileClientError",
    "TileAPIError",
    "TileConnectionError",
]
