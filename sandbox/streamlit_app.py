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


def generate_caption(image, blip_processor, blip_model, temperature=0.3):
    """Generate caption from image using BLIP with temperature control."""
    import torch

    inputs = blip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = blip_model.generate(
            **inputs,
            max_length=50,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9
        )

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
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


    # Image upload section
    st.header("1. Upload Images")
    uploaded_files = st.file_uploader(
        "Choose images to compress (auto-processes on upload)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Load vision models once
        with st.spinner("Loading vision models (ViT + BLIP)..."):
            models = load_vision_models()

        for uploaded_file in uploaded_files:
            # Skip if already processed
            if uploaded_file.name in st.session_state.processed_files:
                continue

            image_bytes = uploaded_file.read()
            original_image = Image.open(BytesIO(image_bytes))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original")
                st.image(original_image, use_container_width=True)

            # Auto-compress
            with st.spinner(f"Compressing {uploaded_file.name}..."):
                result = compress_image_api(image_bytes, reduction, threshold)

                if result:
                    with col2:
                        st.subheader("Gray Overlay")
                        st.image(result['gray_overlay'], use_container_width=True)

                    with col3:
                        st.subheader("Compressed")
                        st.image(result['compressed_image'], use_container_width=True)

                    # Display stats
                    st.success(f"✓ Compressed: {result['reduction_pct']:.1f}% reduction")

                    # Add to session state
                    st.session_state.compressed_images.append({
                        'name': uploaded_file.name,
                        'original': original_image,
                        'compressed': result['compressed_image'],
                        'stats': result['stats']
                    })

                    # Auto-generate embeddings and captions
                    with st.spinner(f"Generating embeddings and captions for {uploaded_file.name}..."):
                        # Generate embeddings
                        original_embedding = generate_embedding(
                            original_image,
                            models['vit_processor'],
                            models['vit_model']
                        )
                        compressed_embedding = generate_embedding(
                            result['compressed_image'],
                            models['vit_processor'],
                            models['vit_model']
                        )

                        # Generate captions with low temperature
                        original_caption = generate_caption(
                            original_image,
                            models['blip_processor'],
                            models['blip_model'],
                            temperature=0.3
                        )
                        compressed_caption = generate_caption(
                            result['compressed_image'],
                            models['blip_processor'],
                            models['blip_model'],
                            temperature=0.3
                        )

                        # Compute similarity
                        similarity = compute_embedding_similarity(original_embedding, compressed_embedding)

                        # Store in session state
                        st.session_state.embeddings_cache[uploaded_file.name] = {
                            'original_embedding': original_embedding,
                            'compressed_embedding': compressed_embedding,
                            'original_caption': original_caption,
                            'compressed_caption': compressed_caption,
                            'similarity': similarity
                        }

                    # Mark as processed
                    st.session_state.processed_files.add(uploaded_file.name)

            # st.divider()

    # Embedding Comparison section
    if st.session_state.compressed_images:
        # st.header("2. Embedding & Semantic Comparison")

        # Use most recent image
        selected_image = st.session_state.compressed_images[-1]
        selected_image_name = selected_image['name']
        st.markdown(f"**Viewing:** {selected_image_name}")

        # Display results (auto-generated during upload)
        if selected_image_name in st.session_state.embeddings_cache:
            cached = st.session_state.embeddings_cache[selected_image_name]

            # Display images side by side at same size
            col1, col2 = st.columns(2)

            # with col1:
            #     st.subheader("Original Image")
            #     st.image(selected_image['original'], use_container_width=True)
            #     st.metric("Size", f"{selected_image['stats']['original_pixels']:,} pixels")

            # with col2:
            #     st.subheader("Compressed Image")
            #     st.image(selected_image['compressed'], use_container_width=True)
            #     st.metric("Size", f"{selected_image['stats']['compressed_pixels']:,} pixels")

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
    
        if st.session_state.compressed_images:
            st.subheader("Statistics")
            total_original = sum(img['stats']['original_pixels'] for img in st.session_state.compressed_images)
            total_compressed = sum(img['stats']['compressed_pixels'] for img in st.session_state.compressed_images)
            total_saved = total_original - total_compressed

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Original Pixels", f"{total_original:,}")
            col2.metric("Total Compressed Pixels", f"{total_compressed:,}")
            col3.metric("Total Pixels Saved", f"{total_saved:,} ({100*total_saved/total_original:.1f}%)")

            with st.expander("View Embedding Statistics"):
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

    st.divider()

    if st.session_state.compressed_images:
        st.header("Image Query Chat")

        # Load VQA model
        @st.cache_resource
        def load_vqa_model():
            from transformers import BlipProcessor, BlipForQuestionAnswering
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            return processor, model

        vqa_processor, vqa_model = load_vqa_model()

        # Use most recent image
        chat_image = st.session_state.compressed_images[-1]
        chat_image_name = chat_image['name']
        # st.markdown(f"**Querying:** {chat_image_name}")

        # Display both images
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.image(chat_image['original'], caption="Original Image", use_container_width=True, width=400)
        # with col2:
        #     st.image(chat_image['compressed'], caption="Compressed Image", use_container_width=True, width=400)

        # Chat input
        question = st.text_input("Ask a question about the images:", key="chat_question")

        if question:
            with st.spinner("Querying both images..."):
                import torch

                # Query original image
                inputs_original = vqa_processor(chat_image['original'], question, return_tensors="pt")
                with torch.no_grad():
                    outputs_original = vqa_model.generate(**inputs_original)
                answer_original = vqa_processor.decode(outputs_original[0], skip_special_tokens=True)

                # Query compressed image
                inputs_compressed = vqa_processor(chat_image['compressed'], question, return_tensors="pt")
                with torch.no_grad():
                    outputs_compressed = vqa_model.generate(**inputs_compressed)
                answer_compressed = vqa_processor.decode(outputs_compressed[0], skip_special_tokens=True)

                # Add to chat history
                st.session_state.chat_history.append({
                    'image': chat_image_name,
                    'question': question,
                    'answer_original': answer_original,
                    'answer_compressed': answer_compressed,
                    'answers_match': answer_original.lower().strip() == answer_compressed.lower().strip()
                })

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
                with st.container():
                    st.markdown(f"**Q ({chat['image']}):** {chat['question']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Answer:**")
                        st.info(chat['answer_original'])
                    with col2:
                        st.markdown("**Compressed Answer:**")
                        st.info(chat['answer_compressed'])

                    st.divider()



if __name__ == "__main__":
    main()
