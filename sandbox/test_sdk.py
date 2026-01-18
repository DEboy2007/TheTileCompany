def test_import():
    try:
        from python_sdk import TileClient
        return True
    except ImportError:
        return False


def test_client_creation():
    try:
        from python_sdk import TileClient
        client = TileClient()
        return True
    except Exception:
        return False


def test_custom_client():
    try:
        from python_sdk import TileClient
        client = TileClient(base_url="http://example.com:8080", timeout=60)
        return True
    except Exception:
        return False


def test_error_classes():
    try:
        from python_sdk.tile_client import (
            TileClientError,
            TileAPIError,
            TileConnectionError
        )
        return True
    except ImportError:
        return False


def main():
    results = []
    results.append(test_import())
    results.append(test_client_creation())
    results.append(test_custom_client())
    results.append(test_error_classes())
    return 0 if all(results) else 1


if __name__ == "__main__":
    exit(main())
