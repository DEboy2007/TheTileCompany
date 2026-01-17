#!/opt/anaconda3/bin/python
"""
Compress an image using attention-based pruning.
Removes low-attention regions while preserving semantic meaning.

Usage:
    python compress_image.py <image_path> [--threshold 0.3] [--output compressed.png]
"""

import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os

# Default settings
DEFAULT_THRESHOLD = 0.3
MODEL = None  # Lazy load


def load_model():
    """Load DINOv2 model (cached)"""
    global MODEL
    if MODEL is None:
        print("Loading DINOv2 model...")
        MODEL = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        MODEL.eval()
        print("✓ Model loaded!")
    return MODEL


def load_image(image_path, size=518):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # Resize to model's expected size (must be divisible by patch size 14)
    img_resized = img.resize((size, size))
    return img_resized, original_size


def get_attention_map(model, img):
    """Extract attention map from the model"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0)

    patch_size = 14
    w, h = img.size
    w_patches = w // patch_size
    h_patches = h // patch_size

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

    return cls_attn


def get_embedding(model, img):
    """Get semantic embedding for an image"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img_tensor)

    return features


def compress_image(img, attention_map, threshold=DEFAULT_THRESHOLD, fill_color=(128, 128, 128)):
    """Compress image by removing low-attention regions"""

    img_array = np.array(img)
    mask = attention_map < threshold

    compressed_array = img_array.copy()
    compressed_array[mask] = fill_color

    compressed_img = Image.fromarray(compressed_array)

    # Calculate stats
    pixels_removed = mask.sum() / mask.size * 100
    pixels_kept = 100 - pixels_removed

    return compressed_img, {
        'pixels_kept': pixels_kept,
        'pixels_removed': pixels_removed,
        'threshold': threshold
    }


def calculate_similarity(model, original_img, compressed_img):
    """Calculate cosine similarity between original and compressed embeddings"""

    emb_original = get_embedding(model, original_img)
    emb_compressed = get_embedding(model, compressed_img)

    # Normalize
    emb_original = emb_original / emb_original.norm(dim=-1, keepdim=True)
    emb_compressed = emb_compressed / emb_compressed.norm(dim=-1, keepdim=True)

    similarity = (emb_original @ emb_compressed.T).item()

    return similarity


def compress(image_path, threshold=DEFAULT_THRESHOLD, output_path=None, verbose=True):
    """
    Main compression function.

    Args:
        image_path: Path to input image
        threshold: Attention threshold (0.0-1.0), default 0.3
        output_path: Path for compressed output (optional)
        verbose: Print progress

    Returns:
        dict with compressed_image, similarity, and stats
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPRESSING: {os.path.basename(image_path)}")
        print(f"{'='*60}")

    # Load model
    model = load_model()

    # Load image
    if verbose:
        print(f"Loading image...")
    img, original_size = load_image(image_path)
    if verbose:
        print(f"✓ Image loaded: {img.size}")

    # Get attention map
    if verbose:
        print(f"Extracting attention map...")
    attention_map = get_attention_map(model, img)
    if verbose:
        print(f"✓ Attention map extracted")

    # Compress
    if verbose:
        print(f"Compressing with threshold={threshold:.0%}...")
    compressed_img, stats = compress_image(img, attention_map, threshold=threshold)
    if verbose:
        print(f"✓ Compressed: {stats['pixels_kept']:.1f}% pixels kept")

    # Calculate similarity
    if verbose:
        print(f"Calculating similarity...")
    similarity = calculate_similarity(model, img, compressed_img)
    stats['similarity'] = similarity
    if verbose:
        print(f"✓ Cosine similarity: {similarity:.4f}")

    # Determine quality
    if similarity > 0.95:
        quality = "Excellent"
    elif similarity > 0.90:
        quality = "Good"
    elif similarity > 0.80:
        quality = "Acceptable"
    else:
        quality = "Poor"
    stats['quality'] = quality

    # Save if output path provided
    if output_path:
        compressed_img.save(output_path)
        if verbose:
            print(f"✓ Saved to: {output_path}")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Threshold:      {threshold:.0%}")
        print(f"  Pixels kept:    {stats['pixels_kept']:.1f}%")
        print(f"  Pixels removed: {stats['pixels_removed']:.1f}%")
        print(f"  Similarity:     {similarity:.4f}")
        print(f"  Quality:        {quality}")
        print(f"{'='*60}\n")

    return {
        'original_image': img,
        'compressed_image': compressed_img,
        'attention_map': attention_map,
        'stats': stats
    }


def main():
    parser = argparse.ArgumentParser(description='Compress image using attention-based pruning')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--threshold', '-t', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Attention threshold (0.0-1.0), default {DEFAULT_THRESHOLD}')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for compressed image')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    # Default output path
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_compressed{ext}"

    # Compress
    result = compress(args.image, threshold=args.threshold, output_path=args.output)


if __name__ == "__main__":
    main()
