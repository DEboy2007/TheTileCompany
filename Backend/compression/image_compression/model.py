"""
DINOv2 model loading and management.
"""

import torch

# Lazy-loaded global
_dino_model = None


def get_dino_model():
    """Load DINOv2 model (cached)"""
    global _dino_model
    if _dino_model is None:
        print("Loading DINOv2 model...")
        _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _dino_model.eval()
    return _dino_model
