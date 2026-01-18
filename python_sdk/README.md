# Tile Python SDK

Simple SDK for the Tile image compression API.

## Installation

```bash
pip install requests
```

## Usage

```python
from python_sdk import TileClient

# Initialize client
client = TileClient()

# Compress an image
result = client.compress_image(
    image_path="path/to/image.jpg",
    output_path="output/compressed.jpg",
    threshold=0.3  # optional
)

print(f"Saved to: {result['output_path']}")
print(f"Stats: {result['image_stats']}")
```
