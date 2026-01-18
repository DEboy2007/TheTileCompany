# Image Compression & Vision Chatbot

A Streamlit application that enables users to upload images, compress them using attention-guided seam carving, and query them using a Vision Transformer-based chatbot.

## Features

1. **Image Upload**: Upload multiple images (PNG, JPG, JPEG)
2. **Intelligent Compression**: Uses DINOv2 attention maps for content-aware seam carving
3. **Caching**: Compressed images are cached locally for faster re-access
4. **Vision Chatbot**: Ask questions about your images using BLIP Vision-Language model
5. **Auto-Backend**: Backend server starts automatically and unobtrusively

## Installation

```bash
# Install Streamlit app dependencies
pip install -r requirements_streamlit.txt

# Install backend dependencies (if not already installed)
cd backend
pip install -r requirements.txt
cd ..
```

## Usage

Simply run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The backend server will start automatically in the background.

## How It Works

### 1. Upload Images
- Click "Browse files" to upload one or more images
- Images are displayed in their original form

### 2. Compress Images
- Adjust compression settings in the sidebar:
  - **Compression Reduction**: How much to reduce (10%-50%)
  - **Attention Threshold**: Threshold for visualization (0.1-0.9)
- Click "Compress [filename]" to process each image
- View the original, gray overlay (attention map), and compressed result

### 3. Query with Chatbot
- After compressing images, they appear in the chatbot section
- Select an image from the dropdown
- Type questions like:
  - "What objects are in this image?"
  - "What color is the sky?"
  - "Is there a person in this image?"
  - "What is the main subject?"
- The BLIP model will analyze the compressed image and answer

### 4. View Statistics
- See total pixels saved across all compressed images
- Track compression efficiency

## Technical Details

### Compression Pipeline
1. Image uploaded to Streamlit
2. Sent to Flask backend API (`/compress`)
3. Backend uses DINOv2 to extract attention maps
4. Seam carving removes low-attention areas
5. Compressed image returned as base64
6. Results cached locally by image hash

### Vision Chatbot
- Uses **BLIP-VQA** (Visual Question Answering) model
- Model: `Salesforce/blip-vqa-base`
- Processes compressed images for efficient inference
- Maintains chat history for reference

### Caching
- Compressed images cached in `cache/compressed_images/`
- Cache key: `md5(image)_reduction_threshold.json`
- Automatically loaded on subsequent requests

## Architecture

```
streamlit_app.py (Frontend)
    ↓ HTTP POST
backend/app.py (Flask API)
    ↓
backend/compression_api.py (DINOv2 + Seam Carving)
    ↓
Compressed Image (cached)
    ↓
BLIP-VQA Model (Chatbot)
```

## API Endpoints

The backend provides:

- `GET /health`: Health check
- `POST /compress`: Compress image
  ```json
  {
    "image": "path_or_url",
    "reduction": 0.3,
    "threshold": 0.3
  }
  ```

## Notes

- First run will download models (DINOv2, BLIP)
- Backend runs on `http://localhost:5001`
- Backend automatically terminates when Streamlit closes
- Cache persists between runs for faster loading

## Troubleshooting

**Backend won't start:**
- Check if port 5001 is already in use
- Manually start backend: `cd backend && python app.py`

**Model download issues:**
- Ensure internet connection for first run
- Models cached in `~/.cache/huggingface/`

**Out of memory:**
- Reduce number of images processed simultaneously
- Use smaller compression reduction values
- Clear cache: `rm -rf cache/`
