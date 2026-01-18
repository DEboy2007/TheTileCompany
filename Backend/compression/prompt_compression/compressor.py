"""
Prompt compression logic using Bear-1/TokenCompany.
"""

import os
import tokenc

# Constants
DEFAULT_AGGRESSIVENESS = 0.5

# Lazy-loaded global
_token_client = None


def get_token_client():
    """Get TokenCompany client (cached)"""
    global _token_client
    if _token_client is None:
        _token_client = tokenc.TokenClient(api_key=os.getenv("TOKENCOMPANY"))
    return _token_client


def compress_prompt(prompt, aggressiveness=DEFAULT_AGGRESSIVENESS):
    """Compress prompt using Bear-1"""
    client = get_token_client()

    response = client.compress_input(
        input=prompt,
        aggressiveness=aggressiveness,
    )

    original_tokens = len(prompt.split())  # Rough token estimate
    compressed_tokens = len(response.output.split())

    return response.output, {
        'original_chars': len(prompt),
        'compressed_chars': len(response.output),
        'original_tokens_est': original_tokens,
        'compressed_tokens_est': compressed_tokens,
        'char_reduction_pct': (1 - len(response.output) / len(prompt)) * 100
    }
