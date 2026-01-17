#!/opt/anaconda3/bin/python
"""
Extract and visualize attention maps from DINOv2
Shows which regions of an image receive the most attention
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
from torchvision import transforms


def load_model():
    """Load DINOv2-small model"""
    print("Loading DINOv2-small model (~80MB on first run)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    print("✓ Model loaded!")
    return model


def load_image(image_path_or_url, size=518):
    """Load and preprocess image"""
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(image_path_or_url).convert('RGB')

    # Resize to model's expected size (must be divisible by patch size 14)
    img = img.resize((size, size))
    return img


def get_attention_maps(model, img):
    """Extract attention maps using hooks"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0)

    patch_size = 14
    w, h = img.size
    w_patches = w // patch_size
    h_patches = h // patch_size

    print(f"Image size: {w}x{h}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Number of patches: {w_patches}x{h_patches} = {w_patches * h_patches}")

    # Storage for attention maps
    attention_maps = []

    def get_attention_hook(module, input, output):
        """Hook to capture attention weights"""
        # output is the attention output, we need to compute attention separately
        pass

    # Use DINOv2's built-in method to get attention
    # DINOv2 models have a method to get the last attention
    with torch.no_grad():
        # Get intermediate layers with attention
        # DINOv2 vits14 has 12 blocks
        attentions = []

        # Manually compute attention for each block
        x = model.patch_embed(img_tensor)

        # Add CLS token
        cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding
        x = model.pos_embed + x

        # Go through each block and extract attention
        for i, blk in enumerate(model.blocks):
            # Get the attention module
            attn_module = blk.attn

            # Compute Q, K, V
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # Compute attention weights
            scale = (C // attn_module.num_heads) ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)

            attentions.append(attn_weights.detach())

            # Complete the forward pass for this block
            # Apply attention to values
            attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
            attn_output = attn_module.proj(attn_output)

            # Residual + MLP
            x = x + attn_output
            x = x + blk.mlp(blk.norm2(x))

    return attentions, (w_patches, h_patches)


def visualize_attention(img, attentions, patch_grid, layer=-1, head=None):
    """Visualize attention maps"""

    w_patches, h_patches = patch_grid

    # Get attention from specified layer
    attn = attentions[layer]  # Shape: (batch, heads, tokens, tokens)

    # Get CLS token attention to all patches (what the model "looks at")
    cls_attn = attn[0, :, 0, 1:]  # Shape: (heads, num_patches)

    if head is not None:
        cls_attn = cls_attn[head:head+1]

    # Average across heads
    cls_attn = cls_attn.mean(dim=0)  # Shape: (num_patches,)

    # Reshape to 2D grid
    cls_attn = cls_attn.reshape(h_patches, w_patches).numpy()

    # Upsample to image size
    cls_attn = np.array(Image.fromarray(cls_attn).resize(img.size, Image.BILINEAR))

    # Normalize
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    return cls_attn


def plot_results(img, attention_map, output_path='attention_output.png'):
    """Plot original image and attention heatmap"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention heatmap
    im = axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title('Attention Map (Where Model Looks)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attention_map, cmap='hot', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.show()


def plot_all_layers(img, attentions, patch_grid, output_path='attention_all_layers.png'):
    """Plot attention maps from all layers"""

    num_layers = len(attentions)
    cols = 4
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i in range(num_layers):
        attn_map = visualize_attention(img, attentions, patch_grid, layer=i)
        axes[i].imshow(img)
        axes[i].imshow(attn_map, cmap='hot', alpha=0.6)
        axes[i].set_title(f'Layer {i+1}')
        axes[i].axis('off')

    # Hide empty subplots
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved all layers visualization to: {output_path}")


def get_high_attention_regions(attention_map, threshold=0.5):
    """Get regions with highest attention"""

    high_attn = attention_map > threshold
    high_attn_percent = high_attn.sum() / attention_map.size * 100

    print(f"\n📊 Attention Analysis:")
    print(f"   Pixels above {threshold:.0%} attention: {high_attn_percent:.1f}%")
    print(f"   Potential token reduction: {100 - high_attn_percent:.1f}%")

    return high_attn


def main():
    print("=" * 60)
    print("ATTENTION MAP EXTRACTION - DINOv2")
    print("=" * 60)

    # Load model
    model = load_model()

    # Test image
    test_url = "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=600"

    print(f"\nLoading test image...")
    img = load_image(test_url)
    print(f"✓ Image loaded: {img.size}")

    # Extract attention
    print("\nExtracting attention maps...")
    attentions, patch_grid = get_attention_maps(model, img)
    print(f"✓ Extracted attention from {len(attentions)} layers")

    # Visualize last layer attention (most semantic)
    print("\nGenerating visualization...")
    attention_map = visualize_attention(img, attentions, patch_grid, layer=-1)

    # Analyze high attention regions
    get_high_attention_regions(attention_map, threshold=0.3)
    get_high_attention_regions(attention_map, threshold=0.5)
    get_high_attention_regions(attention_map, threshold=0.7)

    # Plot results
    plot_results(img, attention_map)

    # Plot all layers
    print("\nGenerating all-layers visualization...")
    plot_all_layers(img, attentions, patch_grid)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
