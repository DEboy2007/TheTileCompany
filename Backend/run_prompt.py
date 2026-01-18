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
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Import compression modules
from compression.image_compression import (
    load_image,
    get_attention_map,
    compress_image,
    image_to_base64,
    get_dino_model
)
from compression.prompt_compression import compress_prompt

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Constants
DEFAULT_THRESHOLD = 0.3
DEFAULT_AGGRESSIVENESS = 0.5

# Lazy-loaded LLM
_llm = None


def get_llm():
    """Get Gemini LLM (cached)"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
    return _llm


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

    from langchain_core.messages import HumanMessage
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
