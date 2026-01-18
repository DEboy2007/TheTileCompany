#!/opt/anaconda3/bin/python
"""
Image Compression API - Attention-guided seam carving for vision model token reduction.

Usage:
    python api.py --image <path_or_url> [--reduction 0.3] [--output compressed.png]
"""

import argparse
import numpy as np
import torch
import requests
from io import BytesIO
from PIL import Image
import base64

# Constants
DEFAULT_REDUCTION = 0.30  # 30% pixel reduction
IMAGE_SIZE = 518
PATCH_SIZE = 14

# Global model cache
_dino_model = None


def load_dino_model():
    """Load DINOv2 model (cached)"""
    global _dino_model
    if _dino_model is None:
        print("Loading DINOv2 model...")
        _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _dino_model.eval()
        if torch.cuda.is_available():
            _dino_model = _dino_model.cuda()
    return _dino_model


def get_attention_map(model, img):
    """Extract attention map from DINOv2"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    w, h = img.size
    w_patches = w // PATCH_SIZE
    h_patches = h // PATCH_SIZE

    with torch.no_grad():
        # Use built-in forward for attention - much faster
        features = model.forward_features(img_tensor)
        # Get attention from last block
        x = model.patch_embed(img_tensor)
        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        if torch.cuda.is_available():
            cls_tokens = cls_tokens.cuda()
        x = torch.cat((cls_tokens, x), dim=1)
        x = model.pos_embed + x

        # Only compute attention for last block
        for blk in model.blocks[:-1]:
            x = blk(x)

        # Last block - extract attention
        blk = model.blocks[-1]
        attn_module = blk.attn
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = (C // attn_module.num_heads) ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)

    # Use last layer attention
    cls_attn = attn_weights[0, :, 0, 1:].mean(dim=0)
    cls_attn = cls_attn.reshape(h_patches, w_patches).cpu().numpy()

    # Resize to image dimensions using scipy for speed
    from scipy.ndimage import zoom
    scale_h = h / h_patches
    scale_w = w / w_patches
    cls_attn_resized = zoom(cls_attn, (scale_h, scale_w), order=1)

    # Normalize to 0-1
    cls_attn_resized = (cls_attn_resized - cls_attn_resized.min()) / (cls_attn_resized.max() - cls_attn_resized.min() + 1e-8)

    return cls_attn_resized


def create_gray_overlay(img, attention_map, threshold=0.3):
    """Create image with low-attention areas grayed out."""
    img_array = np.array(img, dtype=np.float32)
    mask = (attention_map >= threshold).astype(np.float32)[:, :, np.newaxis]
    gray = np.mean(img_array, axis=2, keepdims=True)
    result = img_array * mask + gray * (1 - mask)
    return Image.fromarray(result.astype(np.uint8))


def find_vertical_seam(energy):
    """Find minimum energy vertical seam - VECTORIZED"""
    h, w = energy.shape
    M = energy.copy().astype(np.float64)
    backtrack = np.zeros((h, w), dtype=np.int32)

    # Pad for easier indexing
    for i in range(1, h):
        # Get the three possible predecessors for each pixel
        left = np.concatenate([[np.inf], M[i-1, :-1]])
        middle = M[i-1, :]
        right = np.concatenate([M[i-1, 1:], [np.inf]])

        # Stack and find minimum
        stacked = np.vstack([left, middle, right])
        min_idx = np.argmin(stacked, axis=0)
        min_val = np.min(stacked, axis=0)

        M[i] += min_val
        backtrack[i] = np.arange(w) + min_idx - 1
        backtrack[i] = np.clip(backtrack[i], 0, w - 1)

    # Backtrack
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]

    return seam


def remove_vertical_seam(img_array, seam):
    """Remove vertical seam - VECTORIZED"""
    h, w = img_array.shape[:2]
    if img_array.ndim == 3:
        c = img_array.shape[2]
        mask = np.ones((h, w), dtype=bool)
        mask[np.arange(h), seam] = False
        return img_array[mask].reshape(h, w - 1, c)
    else:
        mask = np.ones((h, w), dtype=bool)
        mask[np.arange(h), seam] = False
        return img_array[mask].reshape(h, w - 1)


def remove_horizontal_seam(img_array, seam):
    """Remove horizontal seam"""
    return np.transpose(remove_vertical_seam(np.transpose(img_array, (1, 0, 2)) if img_array.ndim == 3 else img_array.T, seam), (1, 0, 2) if img_array.ndim == 3 else (1, 0))


def seam_carve_image(img, attention_map, target_reduction):
    """
    Perform seam carving to reduce image size.
    Optimized version with vectorized operations.
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    scale_factor = np.sqrt(1 - target_reduction)
    target_width = int(w * scale_factor)
    target_height = int(h * scale_factor)

    num_vertical_seams = w - target_width
    num_horizontal_seams = h - target_height

    current_attention = attention_map.copy()

    # Remove vertical seams
    for _ in range(num_vertical_seams):
        energy = 1.0 - current_attention
        seam = find_vertical_seam(energy)
        img_array = remove_vertical_seam(img_array, seam)
        current_attention = remove_vertical_seam(current_attention, seam)

    # Remove horizontal seams
    for _ in range(num_horizontal_seams):
        energy = 1.0 - current_attention
        seam = find_vertical_seam(energy.T)
        img_array = remove_horizontal_seam(img_array, seam)
        current_attention = remove_horizontal_seam(current_attention[:, :, np.newaxis], seam).squeeze() if current_attention.ndim == 2 else remove_horizontal_seam(current_attention, seam)

    return Image.fromarray(img_array)


def image_to_base64(img):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def load_image(image_path_or_url):
    """Load image from URL or local path"""
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(image_path_or_url).convert('RGB')

    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return img_resized


def compress(image_path_or_url, reduction=DEFAULT_REDUCTION, threshold=0.3, model=None):
    """
    Compress image using attention-guided seam carving.

    Args:
        image_path_or_url: Path or URL to image
        reduction: Target pixel/token reduction (0-1)
        threshold: Attention threshold for gray overlay visualization
        model: Optional pre-loaded DINOv2 model

    Returns:
        dict with original_image, gray_overlay, compressed_image, reduction_pct, stats
    """
    print("=" * 60)
    print("IMAGE COMPRESSION API")
    print("=" * 60)

    # Load image
    print("\n[1/3] Loading image...")
    img = load_image(image_path_or_url)
    original_pixels = img.width * img.height
    print(f"      Size: {img.size} ({original_pixels:,} pixels)")

    # Extract attention
    print("\n[2/3] Extracting attention map (DINOv2)...")
    if model is None:
        model = load_dino_model()
    attention_map = get_attention_map(model, img)

    # Create gray overlay
    gray_overlay = create_gray_overlay(img, attention_map, threshold)
    print(f"      Gray overlay created (threshold={threshold})")

    # Seam carve
    print("\n[3/3] Seam carving...")
    compressed_img = seam_carve_image(img, attention_map, reduction)
    compressed_pixels = compressed_img.width * compressed_img.height
    actual_reduction = (1 - compressed_pixels / original_pixels) * 100
    print(f"      Carved: {compressed_img.size} ({compressed_pixels:,} pixels)")

    # Summary
    print("\n" + "=" * 60)
    print("COMPRESSION RESULT")
    print("=" * 60)
    print(f"  Original:   {img.size[0]}x{img.size[1]} = {original_pixels:,} pixels")
    print(f"  Compressed: {compressed_img.size[0]}x{compressed_img.size[1]} = {compressed_pixels:,} pixels")
    print(f"  Saved:      {actual_reduction:.1f}% of pixels")
    print("=" * 60)

    return {
        'original_image': img,
        'gray_overlay': gray_overlay,
        'compressed_image': compressed_img,
        'reduction_pct': actual_reduction,
        'stats': {
            'original_size': img.size,
            'compressed_size': compressed_img.size,
            'original_pixels': original_pixels,
            'compressed_pixels': compressed_pixels,
            'pixels_saved': original_pixels - compressed_pixels
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Compress image using attention-guided seam carving')
    parser.add_argument('--image', '-i', type=str, required=True, help='Image URL or local path')
    parser.add_argument('--reduction', '-r', type=float, default=DEFAULT_REDUCTION, help=f'Target reduction (0-1, default: {DEFAULT_REDUCTION})')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Attention threshold for gray overlay (default: 0.3)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for compressed image')

    args = parser.parse_args()

    result = compress(
        image_path_or_url=args.image,
        reduction=args.reduction,
        threshold=args.threshold
    )

    if args.output:
        result['compressed_image'].save(args.output)
        print(f"\nSaved compressed image to: {args.output}")


if __name__ == "__main__":
    main()
