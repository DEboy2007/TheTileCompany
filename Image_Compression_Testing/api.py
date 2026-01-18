#!/opt/anaconda3/bin/python
"""
Image Compression API - Attention-guided seam carving for vision model token reduction.

Usage:
    python api.py --image <path_or_url> [--reduction 0.3] [--output compressed.png]
"""

import argparse
import os
import numpy as np
import torch
import requests
from io import BytesIO
from PIL import Image
import base64

# Constants
DEFAULT_REDUCTION = 0.30  # 30% token reduction
IMAGE_SIZE = 518
PATCH_SIZE = 14


def load_dino_model():
    """Load DINOv2 model"""
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model


def get_attention_map(model, img):
    """Extract attention map from DINOv2"""
    from torchvision import transforms

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
    cls_attn = cls_attn.reshape(h_patches, w_patches).cpu().numpy()

    # Resize to image dimensions
    cls_attn_img = Image.fromarray(cls_attn)
    cls_attn_resized = np.array(cls_attn_img.resize(img.size, Image.BILINEAR))

    # Normalize to 0-1
    cls_attn_resized = (cls_attn_resized - cls_attn_resized.min()) / (cls_attn_resized.max() - cls_attn_resized.min() + 1e-8)

    return cls_attn_resized


def create_gray_overlay(img, attention_map, threshold=0.3):
    """
    Create image with low-attention areas grayed out.

    Args:
        img: PIL Image
        attention_map: 2D numpy array of attention values (0-1)
        threshold: Attention threshold below which pixels are grayed

    Returns:
        PIL Image with gray overlay on low-attention regions
    """
    img_array = np.array(img).astype(np.float32)

    # Create mask: 1 where attention is high, 0 where low
    mask = (attention_map >= threshold).astype(np.float32)
    mask = np.expand_dims(mask, axis=2)

    # Convert low-attention areas to grayscale
    gray = np.mean(img_array, axis=2, keepdims=True)
    gray = np.repeat(gray, 3, axis=2)

    # Blend: keep color where attention is high, gray where low
    result = img_array * mask + gray * (1 - mask)

    return Image.fromarray(result.astype(np.uint8))


def compute_seam_energy(img_array, attention_map):
    """Compute energy for seam carving using inverted attention"""
    energy = 1.0 - attention_map
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    energy = energy * 255
    return energy


def find_vertical_seam(energy):
    """Find minimum energy vertical seam using dynamic programming"""
    h, w = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, h):
        for j in range(w):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = j + idx
                min_energy = M[i-1, j + idx]
            elif j == w - 1:
                idx = np.argmin(M[i-1, j-1:j+1])
                backtrack[i, j] = j - 1 + idx
                min_energy = M[i-1, j - 1 + idx]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = j - 1 + idx
                min_energy = M[i-1, j - 1 + idx]
            M[i, j] += min_energy

    seam = np.zeros(h, dtype=int)
    seam[-1] = np.argmin(M[-1])
    for i in range(h-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]

    return seam


def remove_vertical_seam(img_array, seam):
    """Remove a vertical seam from the image"""
    h, w, c = img_array.shape
    output = np.zeros((h, w-1, c), dtype=img_array.dtype)
    for i in range(h):
        col = seam[i]
        output[i, :, :] = np.delete(img_array[i, :, :], col, axis=0)
    return output


def remove_horizontal_seam(img_array, seam):
    """Remove a horizontal seam from the image"""
    img_t = np.transpose(img_array, (1, 0, 2))
    img_t = remove_vertical_seam(img_t, seam)
    return np.transpose(img_t, (1, 0, 2))


def seam_carve_image(img, attention_map, target_reduction):
    """
    Perform seam carving to reduce image size by target_reduction.

    Args:
        img: PIL Image
        attention_map: 2D numpy array of attention values
        target_reduction: float between 0 and 1 (e.g., 0.3 for 30% reduction)

    Returns:
        PIL Image (carved)
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # Calculate target dimensions
    scale_factor = np.sqrt(1 - target_reduction)
    target_width = int(w * scale_factor)
    target_height = int(h * scale_factor)

    num_vertical_seams = w - target_width
    num_horizontal_seams = h - target_height

    current_attention = attention_map.copy()

    # Remove vertical seams
    for i in range(num_vertical_seams):
        h_curr, w_curr = img_array.shape[:2]
        attention_resized = np.array(Image.fromarray(current_attention).resize((w_curr, h_curr), Image.BILINEAR))
        energy = compute_seam_energy(img_array, attention_resized)
        seam = find_vertical_seam(energy)
        img_array = remove_vertical_seam(img_array, seam)
        current_attention = remove_vertical_seam(np.expand_dims(attention_resized, axis=2), seam).squeeze()

    # Remove horizontal seams
    for i in range(num_horizontal_seams):
        h_curr, w_curr = img_array.shape[:2]
        attention_resized = np.array(Image.fromarray(current_attention).resize((w_curr, h_curr), Image.BILINEAR))
        energy = compute_seam_energy(img_array, attention_resized)
        energy_t = energy.T
        seam = find_vertical_seam(energy_t)
        img_array = remove_horizontal_seam(img_array, seam)
        current_attention = remove_horizontal_seam(np.expand_dims(attention_resized, axis=2), seam).squeeze()

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

    # Resize to DINOv2 size
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return img_resized


def compress(image_path_or_url, reduction=DEFAULT_REDUCTION, threshold=0.3):
    """
    Compress image using attention-guided seam carving.

    Args:
        image_path_or_url: Path or URL to image
        reduction: Target pixel/token reduction (0-1)
        threshold: Attention threshold for gray overlay visualization

    Returns:
        dict with:
            - original_image: PIL Image
            - gray_overlay: PIL Image with low-attention areas grayed
            - compressed_image: PIL Image (seam carved)
            - reduction_pct: Actual percentage of pixels saved
            - stats: Additional statistics
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
    model = load_dino_model()
    attention_map = get_attention_map(model, img)

    # Create gray overlay visualization
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
