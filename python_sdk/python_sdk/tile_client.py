"""Simple TileClient SDK for image compression API."""

import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any


class TileClientError(Exception):
    """Base exception for TileClient errors."""
    pass


class TileAPIError(TileClientError):
    """Exception raised when the Tile API returns an error."""
    pass


class TileConnectionError(TileClientError):
    """Exception raised when connection to the API fails."""
    pass


class TileClient:
    """Client for interacting with the Tile compression API."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the TileClient.

        Args:
            base_url: Base URL for the Tile API (default: http://localhost:5000)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url or "http://localhost:5000"
        self.timeout = timeout
        self._session = requests.Session()

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

        Raises:
            TileConnectionError: If connection to the API fails
            TileAPIError: If the API returns an error
            ValueError: If required fields are missing from the response
        """
        # Prepare request payload
        payload = {
            "image": image_path,
            **kwargs
        }

        # Send request with error handling
        try:
            response = self._session.post(
                f"{self.base_url}/compress-image",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise TileConnectionError(
                f"Failed to connect to Tile API at {self.base_url}: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise TileConnectionError(
                f"Request to Tile API timed out after {self.timeout}s: {e}"
            )
        except requests.exceptions.HTTPError as e:
            raise TileAPIError(f"HTTP error from Tile API: {e}")
        except requests.exceptions.RequestException as e:
            raise TileConnectionError(f"Request to Tile API failed: {e}")

        # Parse response
        try:
            data = response.json()
        except ValueError as e:
            raise TileAPIError(f"Invalid JSON response from API: {e}")

        # Check API status
        if data.get("status") != 0:
            error_msg = data.get('message', 'Unknown error')
            raise TileAPIError(f"API returned error: {error_msg}")

        # Validate response data
        if "compressed_image" not in data:
            raise ValueError("Response missing 'compressed_image' field")

        # Decode and save the compressed image
        try:
            compressed_image_b64 = data["compressed_image"]
            image_bytes = base64.b64decode(compressed_image_b64)
        except (KeyError, base64.binascii.Error) as e:
            raise TileAPIError(f"Failed to decode compressed image: {e}")

        # Save the file
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
        except IOError as e:
            raise TileClientError(f"Failed to save compressed image: {e}")

        return {
            "output_path": str(output_path),
            "image_stats": data.get("image_stats", {})
        }

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
