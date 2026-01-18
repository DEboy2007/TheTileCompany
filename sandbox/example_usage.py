from python_sdk import TileClient
from python_sdk.tile_client import TileConnectionError, TileAPIError


def basic_compression():
    client = TileClient()
    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed.jpg"
        )
        return result
    except (TileConnectionError, TileAPIError, Exception) as e:
        raise


def custom_parameters():
    client = TileClient()
    try:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed_custom.jpg",
            threshold=0.5,
            aggressiveness=2
        )
        return result
    except (TileConnectionError, TileAPIError, Exception) as e:
        raise


def custom_endpoint():
    client = TileClient(base_url="http://custom-server:8080", timeout=60)
    return client


def context_manager():
    with TileClient() as client:
        result = client.compress_image(
            image_path="input.jpg",
            output_path="output/compressed_ctx.jpg"
        )
        return result


if __name__ == "__main__":
    basic_compression()
