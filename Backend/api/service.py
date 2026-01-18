"""
Compression service layer - orchestrates image and prompt compression.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from compression.image_compression import (
    load_image,
    get_attention_map,
    compress_image,
    image_to_base64,
    get_dino_model
)
from compression.prompt_compression import compress_prompt

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Lazy-loaded LLM
_llm = None


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


def run_compression(image_path_or_url, prompt, threshold=0.3, aggressiveness=0.5):
    """
    Run a prompt with compressed image and text.

    Args:
        image_path_or_url: Path or URL to image
        prompt: Text prompt to ask about the image
        threshold: Attention threshold for image compression (0-1)
        aggressiveness: Text compression aggressiveness (0-1)

    Returns:
        dict with answer and compression stats
    """
    # Step 1: Load and compress image
    img, original_size = load_image(image_path_or_url)

    model = get_dino_model()
    attention_map, _ = get_attention_map(model, img)
    compressed_img, image_stats = compress_image(img, attention_map, threshold)

    # Step 2: Compress prompt
    compressed_prompt, prompt_stats = compress_prompt(prompt, aggressiveness)

    # Step 3: Send to Gemini
    llm = get_llm()
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

    return {
        'answer': answer,
        'compressed_image': compressed_img,
        'compressed_prompt': compressed_prompt,
        'image_stats': image_stats,
        'prompt_stats': prompt_stats
    }
