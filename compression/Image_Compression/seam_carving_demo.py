#!/opt/anaconda3/bin/python
"""
Attention-guided seam carving demo.
Shows original, grayed-out, and seam-carved images side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from attention_extractor import load_model, get_attention_maps

# Parameters
THRESHOLD = 0.20
TARGET_WIDTH_RATIO = 0.836  # Reduce to 83.6% of original width (~30% token reduction)
TARGET_HEIGHT_RATIO = 0.836  # Reduce to 83.6% of original height


def get_attention_map(model, img):
    """Get attention map from DINOv2"""
    attentions, patch_grid = get_attention_maps(model, img)

    # Use last layer attention, average over heads
    attn = attentions[-1]
    cls_attn = attn[0, :, 0, 1:].mean(dim=0)

    # Reshape to spatial dimensions
    w_patches, h_patches = patch_grid
    cls_attn = cls_attn.reshape(h_patches, w_patches).cpu().numpy()

    # Resize to image dimensions
    cls_attn_img = Image.fromarray(cls_attn)
    cls_attn_resized = np.array(cls_attn_img.resize(img.size, Image.BILINEAR))

    # Normalize to 0-1
    cls_attn_resized = (cls_attn_resized - cls_attn_resized.min()) / (cls_attn_resized.max() - cls_attn_resized.min() + 1e-8)

    return cls_attn_resized


def create_grayed_image(img, attention_map, threshold):
    """Create image with low-attention areas grayed out"""
    img_array = np.array(img)
    mask = attention_map < threshold

    grayed = img_array.copy()
    grayed[mask] = (128, 128, 128)

    return Image.fromarray(grayed)


def compute_seam_energy(img_array, attention_map):
    """
    Compute energy for seam carving using INVERTED attention.
    Low attention = high energy = more likely to be removed.
    """
    # Invert attention: we want to remove LOW attention areas
    energy = 1.0 - attention_map

    # Normalize to 0-255 range for consistency
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    energy = energy * 255

    return energy


def find_vertical_seam(energy):
    """
    Find the minimum energy vertical seam using dynamic programming.
    Returns array of column indices for each row.
    """
    h, w = energy.shape

    # M[i,j] = minimum energy to reach pixel (i,j) from top
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    # Fill the cumulative minimum energy map
    for i in range(1, h):
        for j in range(w):
            # Check three possible predecessors
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

    # Backtrack to find the seam
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
    # Transpose, remove vertical seam, transpose back
    img_t = np.transpose(img_array, (1, 0, 2))
    img_t = remove_vertical_seam(img_t, seam)
    return np.transpose(img_t, (1, 0, 2))


def seam_carve(img, attention_map, target_width, target_height):
    """
    Perform seam carving to reduce image to target dimensions.
    Uses attention map as energy (inverted - low attention = high removal priority).
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    num_vertical_seams = w - target_width
    num_horizontal_seams = h - target_height

    print(f"Removing {num_vertical_seams} vertical seams and {num_horizontal_seams} horizontal seams")

    # Resize attention map to match current image size
    current_attention = attention_map.copy()

    # Remove vertical seams
    for i in range(num_vertical_seams):
        if i % 10 == 0:
            print(f"  Vertical seam {i+1}/{num_vertical_seams}")

        # Resize attention map to current image size
        h_curr, w_curr = img_array.shape[:2]
        attention_resized = np.array(Image.fromarray(current_attention).resize((w_curr, h_curr), Image.BILINEAR))

        energy = compute_seam_energy(img_array, attention_resized)
        seam = find_vertical_seam(energy)
        img_array = remove_vertical_seam(img_array, seam)

        # Also remove seam from attention map
        current_attention = remove_vertical_seam(
            np.expand_dims(attention_resized, axis=2), seam
        ).squeeze()

    # Remove horizontal seams
    for i in range(num_horizontal_seams):
        if i % 10 == 0:
            print(f"  Horizontal seam {i+1}/{num_horizontal_seams}")

        # Resize attention map to current image size
        h_curr, w_curr = img_array.shape[:2]
        attention_resized = np.array(Image.fromarray(current_attention).resize((w_curr, h_curr), Image.BILINEAR))

        energy = compute_seam_energy(img_array, attention_resized)
        energy_t = energy.T
        seam = find_vertical_seam(energy_t)
        img_array = remove_horizontal_seam(img_array, seam)

        # Also remove seam from attention map
        current_attention = remove_horizontal_seam(
            np.expand_dims(attention_resized, axis=2), seam
        ).squeeze()

    return Image.fromarray(img_array)


def main():
    # Load high-res image from URL
    import requests
    from io import BytesIO

    # High-quality image URL - a lighthouse with ocean background
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200"

    print(f"Downloading image from: {image_url}")
    response = requests.get(image_url)
    original_img = Image.open(BytesIO(response.content)).convert('RGB')

    print(f"Original image size: {original_img.size}")

    # Resize to 518x518 for DINOv2
    img = original_img.resize((518, 518), Image.BILINEAR)

    print("Loading DINOv2 model...")
    model = load_model()

    print("\nExtracting attention map...")
    attention_map = get_attention_map(model, img)

    print(f"\nCreating grayed image (threshold={THRESHOLD:.0%})...")
    grayed_img = create_grayed_image(img, attention_map, THRESHOLD)

    print(f"\nPerforming seam carving...")
    target_width = int(img.width * TARGET_WIDTH_RATIO)
    target_height = int(img.height * TARGET_HEIGHT_RATIO)
    carved_img = seam_carve(img, attention_map, target_width, target_height)

    # Calculate token reduction
    original_patches = (518 // 14) * (518 // 14)
    carved_patches = (target_width // 14) * (target_height // 14)
    reduction = (1 - carved_patches / original_patches) * 100

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Original size:     {img.size}")
    print(f"Carved size:       {carved_img.size}")
    print(f"Original patches:  {original_patches}")
    print(f"Carved patches:    {carved_patches}")
    print(f"Token reduction:   {reduction:.1f}%")
    print(f"{'='*70}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title(f'Original\n{img.size[0]}x{img.size[1]} ({original_patches} patches)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(grayed_img)
    axes[1].set_title(f'Grayed (threshold={THRESHOLD:.0%})\n{grayed_img.size[0]}x{grayed_img.size[1]} ({original_patches} patches)', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(carved_img)
    axes[2].set_title(f'Seam Carved\n{carved_img.size[0]}x{carved_img.size[1]} ({carved_patches} patches)\n{reduction:.1f}% token reduction', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('Graphs/seam_carving_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: Graphs/seam_carving_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
