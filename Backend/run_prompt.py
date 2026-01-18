#!/opt/anaconda3/bin/python
"""
Run a prompt with compressed image and compressed text.
Combines attention-based image pruning with Bear-1 text compression.

Usage:
    python run_prompt.py --image <url_or_path> --prompt "What is in this image?"
"""

import argparse
import os
import sys
import base64
from io import BytesIO

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'API_TESTS', 'Compression'))

import torch
import requests
import tokenc
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from torchvision import transforms
import numpy as np

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Constants
DEFAULT_THRESHOLD = 0.3
DEFAULT_AGGRESSIVENESS = 0.5
PATCH_SIZE = 14
IMAGE_SIZE = 518

# Lazy-loaded globals
_dino_model = None
_token_client = None
_llm = None


def get_dino_model():
    """Load DINOv2 model (cached)"""
    global _dino_model
    if _dino_model is None:
        print("Loading DINOv2 model...")
        _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _dino_model.eval()
    return _dino_model


def get_token_client():
    """Get TokenCompany client (cached)"""
    global _token_client
    if _token_client is None:
        _token_client = tokenc.TokenClient(api_key=os.getenv("TOKENCOMPANY"))
    return _token_client


def get_llm():
    """Get Gemini LLM (cached)"""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
    return _llm


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


def image_to_base64(img):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def run_prompt(image_path_or_url, prompt, threshold=DEFAULT_THRESHOLD, aggressiveness=DEFAULT_AGGRESSIVENESS, verbose=True):
    """
    Run a prompt with compressed image and text.

    Args:
        image_path_or_url: Path or URL to image
        prompt: Text prompt to ask about the image
        threshold: Attention threshold for image compression (0-1)
        aggressiveness: Text compression aggressiveness (0-1)
        verbose: Print detailed output

    Returns:
        dict with answer and compression stats
    """

    if verbose:
        print("=" * 70)
        print("COMPRESSED VISION-LANGUAGE QUERY")
        print("=" * 70)

    # Step 1: Load and compress image
    if verbose:
        print("\n[1/4] Loading image...")
    img, original_size = load_image(image_path_or_url)
    if verbose:
        print(f"      Image loaded: {img.size}")

    if verbose:
        print("\n[2/4] Compressing image (attention-based pruning)...")
    model = get_dino_model()
    attention_map, _ = get_attention_map(model, img)
    compressed_img, image_stats = compress_image(img, attention_map, threshold)
    if verbose:
        print(f"      Tokens: {image_stats['total_tokens']} → {image_stats['tokens_kept']} ({image_stats['token_reduction_pct']:.1f}% saved)")

    # Step 2: Compress prompt
    if verbose:
        print("\n[3/4] Compressing prompt (Bear-1)...")
    compressed_prompt, prompt_stats = compress_prompt(prompt, aggressiveness)
    if verbose:
        print(f"      Chars: {prompt_stats['original_chars']} → {prompt_stats['compressed_chars']} ({prompt_stats['char_reduction_pct']:.1f}% saved)")
        print(f"      Compressed: \"{compressed_prompt[:80]}{'...' if len(compressed_prompt) > 80 else ''}\"")

    # Step 3: Send to Gemini
    if verbose:
        print("\n[4/4] Querying Gemini with compressed inputs...")

    llm = get_llm()

    # Create message with image
    img_base64 = image_to_base64(compressed_img)

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            },
            {
                "type": "text",
                "text": compressed_prompt
            }
        ]
    )

    response = llm.invoke([message])
    answer = response.content

    # Results
    if verbose:
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)

        print("\n" + "=" * 70)
        print("COMPRESSION STATS")
        print("=" * 70)
        print(f"\n📷 IMAGE COMPRESSION:")
        print(f"   Original tokens:    {image_stats['total_tokens']}")
        print(f"   Compressed tokens:  {image_stats['tokens_kept']}")
        print(f"   Tokens saved:       {image_stats['tokens_removed']} ({image_stats['token_reduction_pct']:.1f}%)")

        print(f"\n💬 PROMPT COMPRESSION:")
        print(f"   Original chars:     {prompt_stats['original_chars']}")
        print(f"   Compressed chars:   {prompt_stats['compressed_chars']}")
        print(f"   Chars saved:        {prompt_stats['original_chars'] - prompt_stats['compressed_chars']} ({prompt_stats['char_reduction_pct']:.1f}%)")

        print(f"\n🎯 TOTAL SAVINGS:")
        total_original = image_stats['total_tokens'] + prompt_stats['original_tokens_est']
        total_compressed = image_stats['tokens_kept'] + prompt_stats['compressed_tokens_est']
        total_saved_pct = (1 - total_compressed / total_original) * 100
        print(f"   Estimated total reduction: {total_saved_pct:.1f}%")
        print("=" * 70)

    return {
        'answer': answer,
        'compressed_image': compressed_img,
        'compressed_prompt': compressed_prompt,
        'image_stats': image_stats,
        'prompt_stats': prompt_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Run prompt with compressed image and text')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Image URL or local path')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                        help='Prompt to ask about the image')
    parser.add_argument('--threshold', '-t', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Image attention threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--aggressiveness', '-a', type=float, default=DEFAULT_AGGRESSIVENESS,
                        help=f'Text compression aggressiveness (default: {DEFAULT_AGGRESSIVENESS})')

    args = parser.parse_args()

    run_prompt(
        image_path_or_url=args.image,
        prompt=args.prompt,
        threshold=args.threshold,
        aggressiveness=args.aggressiveness
    )


if __name__ == "__main__":
    main()
