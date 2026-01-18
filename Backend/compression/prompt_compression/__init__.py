"""
Prompt compression module using Bear-1/TokenCompany.
"""

from .compressor import compress_prompt, get_token_client

__all__ = [
    'compress_prompt',
    'get_token_client'
]
