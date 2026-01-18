"""
Image compression logic using attention-based pruning.
"""

import base64
from io import BytesIO
import requests
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Constants
DEFAULT_THRESHOLD = 0.3
PATCH_SIZE = 14
IMAGE_SIZE = 518


def load_image(image_path_or_url):
    """Load image from URL or local path"""
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(image_path_or_url).convert('RGB')

    original_size = img.size
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return img_resized, original_size


def get_attention_map(model, img):
    """Extract attention map from DINOv2"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0)
    w, h = img.size
    w_patches = w // PATCH_SIZE
    h_patches = h // PATCH_SIZE

    with torch.no_grad():
        attentions = []
        x = model.patch_embed(img_tensor)
        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = model.pos_embed + x

        for blk in model.blocks:
            attn_module = blk.attn
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = (C // attn_module.num_heads) ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)
            attentions.append(attn_weights.detach())
            attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
            attn_output = attn_module.proj(attn_output)
            x = x + attn_output
            x = x + blk.mlp(blk.norm2(x))

    # Use last layer attention
    attn = attentions[-1]
    cls_attn = attn[0, :, 0, 1:].mean(dim=0)
    cls_attn = cls_attn.reshape(h_patches, w_patches).numpy()
    cls_attn = np.array(Image.fromarray(cls_attn).resize(img.size, Image.BILINEAR))
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    return cls_attn, (w_patches, h_patches)


def compress_image(img, attention_map, threshold=DEFAULT_THRESHOLD):
    """Compress image by masking low-attention regions"""
    img_array = np.array(img)
    mask = attention_map < threshold

    compressed_array = img_array.copy()
    compressed_array[mask] = (128, 128, 128)  # Gray fill

    compressed_img = Image.fromarray(compressed_array)

    # Calculate token savings
    total_patches = (img.size[0] // PATCH_SIZE) * (img.size[1] // PATCH_SIZE)
    h_patches = img.size[1] // PATCH_SIZE
    w_patches = img.size[0] // PATCH_SIZE
    patch_attention = attention_map.reshape(h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean(axis=(1, 3))
    tokens_kept = (patch_attention >= threshold).sum()
    tokens_removed = total_patches - tokens_kept

    return compressed_img, {
        'total_tokens': total_patches,
        'tokens_kept': int(tokens_kept),
        'tokens_removed': int(tokens_removed),
        'token_reduction_pct': tokens_removed / total_patches * 100
    }


def image_to_base64(img):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
