# Backend Refactored Structure

The backend has been refactored to separate concerns into distinct, modular components.

## Directory Structure

```
backend/
├── api/                          # Flask API routes and service layer
│   ├── __init__.py
│   ├── routes.py                 # Flask route definitions
│   └── service.py                # Business logic orchestration
│
├── compression/                  # Core compression logic
│   ├── __init__.py
│   ├── image_compression/        # Image compression module
│   │   ├── __init__.py
│   │   ├── model.py              # DINOv2 model loading
│   │   └── compressor.py         # Image compression logic
│   └── prompt_compression/       # Prompt compression module
│       ├── __init__.py
│       └── compressor.py         # Text compression logic
│
├── benchmarks/                   # Evaluation and benchmarking (separate from core)
│   ├── __init__.py
│   └── evaluator.py              # Performance evaluation utilities
│
├── app.py                        # Flask application entry point
└── run_prompt.py                 # CLI tool for running compressed prompts
```

## Modules

### 1. API Module (`api/`)
- **Purpose**: Flask REST API for compression services
- **Key Files**:
  - `routes.py`: Defines Flask endpoints (`/compress`, `/process`)
  - `service.py`: Orchestrates compression workflow and LLM queries

### 2. Compression Module (`compression/`)
Core compression logic, independent of API or benchmarking.

#### Image Compression (`compression/image_compression/`)
- **model.py**: DINOv2 model loading and caching
- **compressor.py**: Attention-based image compression
  - `load_image()`: Load from URL or local path
  - `get_attention_map()`: Extract attention from DINOv2
  - `compress_image()`: Apply attention-based pruning
  - `image_to_base64()`: Convert image to base64

#### Prompt Compression (`compression/prompt_compression/`)
- **compressor.py**: Text compression using Bear-1/TokenCompany
  - `get_token_client()`: TokenCompany client management
  - `compress_prompt()`: Compress text with aggressiveness control

### 3. Benchmarks Module (`benchmarks/`)
- **Purpose**: Evaluation and testing, **separate from core compression logic**
- **Key Functions**:
  - `evaluate_compression()`: Comprehensive performance metrics
  - `benchmark_thresholds()`: Test multiple image compression thresholds
  - `benchmark_aggressiveness()`: Test multiple prompt compression levels
  - `generate_report()`: Human-readable performance reports

### 4. CLI Tool (`run_prompt.py`)
- Command-line interface for running compressed vision-language queries
- Uses compression modules directly
- Verbose output with detailed stats

### 5. Flask App (`app.py`)
- Entry point for Flask application
- Imports and runs the app created by `api.create_app()`

## Usage

### Running the Flask API

```bash
cd backend
python app.py
```

API Endpoints:
- `POST /compress`: Compress image + prompt and query LLM
  ```json
  {
    "image": "url_or_path",
    "prompt": "What is in this image?",
    "threshold": 0.3,
    "aggressiveness": 0.5
  }
  ```

- `POST /process`: Legacy text processing endpoint

### Running the CLI Tool

```bash
python run_prompt.py --image <url_or_path> --prompt "What is in this image?" --threshold 0.3 --aggressiveness 0.5
```

### Using Compression Modules Programmatically

```python
from compression import compress_image, compress_prompt, load_image, get_dino_model, get_attention_map

# Image compression
img, _ = load_image("path/to/image.jpg")
model = get_dino_model()
attention_map, _ = get_attention_map(model, img)
compressed_img, stats = compress_image(img, attention_map, threshold=0.3)

# Prompt compression
compressed_text, stats = compress_prompt("Your long prompt here", aggressiveness=0.5)
```

### Running Benchmarks

```python
from benchmarks import evaluate_compression, benchmark_thresholds, generate_report

# Evaluate compression performance
metrics = evaluate_compression(
    image_path_or_url="path/to/image.jpg",
    prompt="What is in this image?",
    threshold=0.3,
    aggressiveness=0.5
)

# Generate report
report = generate_report(metrics)
print(report)

# Benchmark multiple thresholds
results = benchmark_thresholds("path/to/image.jpg", thresholds=[0.1, 0.3, 0.5, 0.7])
```

## Design Principles

1. **Separation of Concerns**: Core compression logic is independent of API and benchmarking
2. **Modularity**: Each module has a single, well-defined responsibility
3. **Reusability**: Compression modules can be used in API, CLI, or benchmarks
4. **Lazy Loading**: Models are loaded only when needed and cached for reuse
5. **Clean Dependencies**: Benchmarking depends on compression, but compression is standalone

## Benefits

- **Maintainability**: Clear structure makes code easy to understand and modify
- **Testability**: Each module can be tested independently
- **Scalability**: Easy to add new compression methods or API endpoints
- **Performance**: Benchmarking tools are separate and don't bloat the core API
- **Flexibility**: Core compression logic can be used in different contexts (API, CLI, notebooks, etc.)
