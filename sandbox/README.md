# Tile SDK Sandbox

```bash
source venv/bin/activate
python test_sdk.py
python example_usage.py
```

## Usage

```python
from python_sdk import TileClient

client = TileClient()
result = client.compress_image(
    image_path="input.jpg",
    output_path="output.jpg",
    threshold=0.3
)
```
