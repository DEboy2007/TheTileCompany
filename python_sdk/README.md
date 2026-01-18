# Tile Python SDK

Python SDK for the Tile image compression API. This library provides a simple interface to compress images using advanced compression algorithms.

## Installation

### From Source (Local Development)

```bash
# Clone the repository (if not already done)
cd python_sdk

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### From Built Distribution

```bash
# Build the package
python -m pip install build
python -m build

# Install from the built wheel
pip install dist/tile_sdk-0.1.0-py3-none-any.whl
```

## Requirements

- Python >= 3.8
- requests >= 2.25.0

## Usage

### Basic Example

```python
from python_sdk import TileClient

# Initialize client (connects to localhost:5000 by default)
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

### Advanced Usage

```python
from python_sdk import TileClient

client = TileClient()

# Compress with custom parameters
result = client.compress_image(
    image_path="input.png",
    output_path="output.png",
    threshold=0.5,
    aggressiveness=2
)

# Access compression statistics
stats = result['image_stats']
print(f"Original size: {stats.get('original_size')}")
print(f"Compressed size: {stats.get('compressed_size')}")
print(f"Compression ratio: {stats.get('ratio')}")
```

## API Reference

### TileClient

#### `__init__()`

Initialize the Tile client.

#### `compress_image(image_path, output_path, **kwargs)`

Compress an image and save the result.

**Arguments:**
- `image_path` (str): Path or URL to the input image
- `output_path` (str): Path where the compressed image will be saved
- `**kwargs`: Additional parameters passed to the compression API
  - `threshold` (float): Compression threshold (0.0 - 1.0)
  - `aggressiveness` (int): Compression aggressiveness level

**Returns:**
- `dict`: Response data containing:
  - `output_path` (str): Path to the saved compressed image
  - `image_stats` (dict): Compression statistics

**Raises:**
- `requests.exceptions.HTTPError`: If the API request fails
- `Exception`: If the API returns an error status

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=python_sdk --cov-report=html
```

### Code Formatting

```bash
# Format code with black
black python_sdk

# Check with flake8
flake8 python_sdk

# Type checking with mypy
mypy python_sdk
```

## License

This project is licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).

- Personal and educational use is permitted
- Commercial use is prohibited
- Modifications and derivatives are not permitted for distribution
- Attribution to NexHacks Team is required

For commercial licensing inquiries, please contact NexHacks Team.

See LICENSE file for full details.
