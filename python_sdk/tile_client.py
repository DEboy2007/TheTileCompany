"""Simple TileClient SDK for image compression API."""

import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any


class TileClient:
    """Client for interacting with the Tile compression API."""

    def __init__(self):
        """Initialize the TileClient."""
        self.base_url = "http://localhost:5000"

    def compress_image(
        self,
        image_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compress an image and save the result.

        Args:
            image_path: Path or URL to the input image
            output_path: Path where the compressed image will be saved
            **kwargs: Additional parameters (e.g., threshold, aggressiveness)

        Returns:
            Dict containing the response data including image_stats
        """
        # Prepare request payload
        payload = {
            "image": image_path,
            **kwargs
        }

        # Send request
        response = requests.post(
            f"{self.base_url}/compress-image",
            json=payload
        )
        response.raise_for_status()

        # Parse response
        data = response.json()

        if data.get("status") != 0:
            raise Exception(f"API error: {data.get('message', 'Unknown error')}")

        # Decode and save the compressed image
        compressed_image_b64 = data["compressed_image"]
        image_bytes = base64.b64decode(compressed_image_b64)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        return {
            "output_path": str(output_path),
            "image_stats": data.get("image_stats", {})
        }
