"""
Streamlit Image Compression & Vision Chatbot
===============================================
Upload images, compress them using attention-guided seam carving,
and query them using a Vision Transformer-based chatbot.
"""

import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import os
import json
import hashlib
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:5001"
CACHE_DIR = Path("cache/compressed_images")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def check_backend_health():
    """Check if backend server is running and accessible."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def get_image_hash(image_bytes):
    """Generate hash for image to use as cache key."""
    return hashlib.md5(image_bytes).hexdigest()


def compress_image_api(image_bytes, reduction=0.3, threshold=0.3):
    """
    Send image to backend API for compression.

    Args:
        image_bytes: Image bytes
        reduction: Compression reduction ratio (0-1)
        threshold: Attention threshold

    Returns:
        dict with compressed_image, gray_overlay, stats
    """
    # Check cache first
    image_hash = get_image_hash(image_bytes)
    cache_file = CACHE_DIR / f"{image_hash}_{reduction}_{threshold}.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            return {
                'compressed_image': base64_to_image(cached_data['compressed_image_base64']),
                'gray_overlay': base64_to_image(cached_data['gray_overlay_base64']),
                'stats': cached_data['stats'],
                'reduction_pct': cached_data['reduction_pct'],
                'from_cache': True
            }

    # Save image temporarily with absolute path
    temp_path = CACHE_DIR / f"temp_{image_hash}.png"
    with open(temp_path, 'wb') as f:
        f.write(image_bytes)

    try:
        # Call API with absolute path
        response = requests.post(
            f"{BACKEND_URL}/compress",
            json={
                "image": str(temp_path.absolute()),
                "reduction": reduction,
                "threshold": threshold
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()

            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump({
                    'compressed_image_base64': result['compressed_image_base64'],
                    'gray_overlay_base64': result['gray_overlay_base64'],
                    'stats': result['stats'],
                    'reduction_pct': result['reduction_pct']
                }, f)

            return {
                'compressed_image': base64_to_image(result['compressed_image_base64']),
                'gray_overlay': base64_to_image(result['gray_overlay_base64']),
                'stats': result['stats'],
                'reduction_pct': result['reduction_pct'],
                'from_cache': False
            }
        else:
            st.error(f"API error: {response.json().get('message', 'Unknown error')}")
            return None

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def base64_to_image(base64_str):
    """Convert base64 string to PIL Image."""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes))


def image_to_bytes(image):
    """Convert PIL Image to bytes."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()


@st.cache_resource
def load_vision_models():
    """Load Vision models for embedding and captioning."""
    from transformers import BlipProcessor, BlipForConditionalGeneration, ViTModel, ViTImageProcessor

    # Load BLIP for captioning (decoding)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load ViT for embeddings
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    return {
        'blip_processor': blip_processor,
        'blip_model': blip_model,
        'vit_processor': vit_processor,
        'vit_model': vit_model
    }


def generate_embedding(image, vit_processor, vit_model):
    """Generate embedding from image using ViT."""
    import torch

    inputs = vit_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = vit_model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()

    return embedding


def generate_caption(image, blip_processor, blip_model):
    """Generate caption from image using BLIP."""
    import torch

    inputs = blip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = blip_model.generate(**inputs, max_length=50)

    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def compute_embedding_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    import torch
    import torch.nn.functional as F

    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    return similarity.item()


def main():
    st.set_page_config(
        page_title="Image Compression & Vision Chatbot",
        page_icon="🖼️",
        layout="wide"
    )

    st.title("🖼️ Image Compression & Vision Chatbot")
    st.markdown("""
    Upload images to compress them using attention-guided seam carving,
    then query them with a Vision Transformer-based chatbot.
    """)

    # Check if backend server is running
    if not check_backend_health():
        st.error(f"""
        **Backend server is not running!**

        Please start the backend server before using this app:

        ```bash
        cd backend
        python app.py
        ```

        The backend should be running at: {BACKEND_URL}
        """)
        st.stop()

    # Show backend status in sidebar
    st.sidebar.success(f"✓ Backend connected at {BACKEND_URL}")

    # Sidebar controls
    st.sidebar.header("Compression Settings")
    reduction = st.sidebar.slider(
        "Compression Reduction",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Target pixel reduction ratio"
    )
    threshold = st.sidebar.slider(
        "Attention Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Threshold for gray overlay visualization"
    )

    # Initialize session state
    if 'compressed_images' not in st.session_state:
        st.session_state.compressed_images = []
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}

    # Image upload section
    st.header("1. Upload Images")
    uploaded_files = st.file_uploader(
        "Choose images to compress",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image_bytes = uploaded_file.read()
            original_image = Image.open(BytesIO(image_bytes))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original")
                st.image(original_image, use_container_width=True)

            if st.button(f"Compress {uploaded_file.name}", key=f"compress_{uploaded_file.name}"):
                with st.spinner("Compressing image..."):
                    result = compress_image_api(image_bytes, reduction, threshold)

                    if result:
                        with col2:
                            st.subheader("Gray Overlay")
                            st.image(result['gray_overlay'], use_container_width=True)

                        with col3:
                            st.subheader("Compressed")
                            st.image(result['compressed_image'], use_container_width=True)

                        # Display stats
                        st.success(f"✓ Compressed! {result['reduction_pct']:.1f}% reduction")
                        if result.get('from_cache'):
                            st.info("Loaded from cache")

                        # Add to session state for chatbot
                        st.session_state.compressed_images.append({
                            'name': uploaded_file.name,
                            'original': original_image,
                            'compressed': result['compressed_image'],
                            'stats': result['stats']
                        })

            st.divider()

    # Embedding Comparison section
    if st.session_state.compressed_images:
        st.header("2. Embedding & Semantic Comparison")
        st.markdown("Compare embeddings and decoded captions from original vs compressed images to demonstrate compression effectiveness.")

        # Load vision models
        with st.spinner("Loading vision models (ViT + BLIP)..."):
            models = load_vision_models()

        # Select image to analyze
        selected_image_name = st.selectbox(
            "Select image to analyze",
            [img['name'] for img in st.session_state.compressed_images]
        )

        selected_image = next(
            img for img in st.session_state.compressed_images
            if img['name'] == selected_image_name
        )

        # Generate embeddings and captions button
        if st.button("Generate Embeddings & Captions", key="generate_embeddings"):
            with st.spinner("Generating embeddings and captions..."):
                # Generate embeddings
                original_embedding = generate_embedding(
                    selected_image['original'],
                    models['vit_processor'],
                    models['vit_model']
                )
                compressed_embedding = generate_embedding(
                    selected_image['compressed'],
                    models['vit_processor'],
                    models['vit_model']
                )

                # Generate captions (decode embeddings)
                original_caption = generate_caption(
                    selected_image['original'],
                    models['blip_processor'],
                    models['blip_model']
                )
                compressed_caption = generate_caption(
                    selected_image['compressed'],
                    models['blip_processor'],
                    models['blip_model']
                )

                # Compute similarity
                similarity = compute_embedding_similarity(original_embedding, compressed_embedding)

                # Store in session state
                st.session_state.embeddings_cache[selected_image_name] = {
                    'original_embedding': original_embedding,
                    'compressed_embedding': compressed_embedding,
                    'original_caption': original_caption,
                    'compressed_caption': compressed_caption,
                    'similarity': similarity
                }

        # Display results if available
        if selected_image_name in st.session_state.embeddings_cache:
            cached = st.session_state.embeddings_cache[selected_image_name]

            # Display images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(selected_image['original'], use_container_width=True)
                st.metric("Size", f"{selected_image['stats']['original_pixels']:,} pixels")
                st.markdown("**Decoded Caption:**")
                st.info(cached['original_caption'])

            with col2:
                st.subheader("Compressed Image")
                st.image(selected_image['compressed'], use_container_width=True)
                st.metric("Size", f"{selected_image['stats']['compressed_pixels']:,} pixels")
                st.markdown("**Decoded Caption:**")
                st.info(cached['compressed_caption'])

            # Embedding similarity
            st.divider()
            st.subheader("Embedding Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding Similarity", f"{cached['similarity']:.4f}")
            with col2:
                st.metric("Original Embedding Dim", f"{cached['original_embedding'].shape[0]}")
            with col3:
                st.metric("Compressed Embedding Dim", f"{cached['compressed_embedding'].shape[0]}")

            # Interpretation
            if cached['similarity'] > 0.95:
                st.success("✓ **Excellent semantic preservation!** The compressed image maintains nearly identical semantic content.")
            elif cached['similarity'] > 0.90:
                st.success("✓ **Good semantic preservation.** The compressed image maintains strong semantic similarity.")
            elif cached['similarity'] > 0.85:
                st.warning("⚠ **Moderate semantic preservation.** Some semantic information may be lost.")
            else:
                st.error("✗ **Poor semantic preservation.** Significant semantic information lost during compression.")

            # Show embedding stats
            with st.expander("View Embedding Statistics"):
                import torch
                st.write("**Original Embedding Stats:**")
                st.write(f"- Mean: {cached['original_embedding'].mean().item():.4f}")
                st.write(f"- Std: {cached['original_embedding'].std().item():.4f}")
                st.write(f"- Min: {cached['original_embedding'].min().item():.4f}")
                st.write(f"- Max: {cached['original_embedding'].max().item():.4f}")

                st.write("\n**Compressed Embedding Stats:**")
                st.write(f"- Mean: {cached['compressed_embedding'].mean().item():.4f}")
                st.write(f"- Std: {cached['compressed_embedding'].std().item():.4f}")
                st.write(f"- Min: {cached['compressed_embedding'].min().item():.4f}")
                st.write(f"- Max: {cached['compressed_embedding'].max().item():.4f}")

    # Statistics
    if st.session_state.compressed_images:
        st.header("3. Statistics")
        total_original = sum(img['stats']['original_pixels'] for img in st.session_state.compressed_images)
        total_compressed = sum(img['stats']['compressed_pixels'] for img in st.session_state.compressed_images)
        total_saved = total_original - total_compressed

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Original Pixels", f"{total_original:,}")
        col2.metric("Total Compressed Pixels", f"{total_compressed:,}")
        col3.metric("Total Pixels Saved", f"{total_saved:,} ({100*total_saved/total_original:.1f}%)")


if __name__ == "__main__":
    main()
