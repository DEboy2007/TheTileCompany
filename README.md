# TheTileCompany
### Project for NexHacks 2026

**Semantics-preserving image compression for vision transformers**

## Inspiration

We were inspired by The Token Company and its mission to compress text in LLM prompts. As large language models continue to advance, an increasing proportion of human–model interaction involves images and other multimodal inputs. Alongside the growing sophistication of LLMs such as ChatGPT, there is an increasing reliance on vision models.

With inference complexity directly proportional to image size, input compression is crucial to reducing costs. Efficiently compressing image inputs can reduce energy costs, decrease latency, and enable deployment in resource-constrained environments, all while leaving performance in downstream vision tasks virtually unchanged.

## What it does

The same way The Token Company is able to reduce input size within LLMs and text models, TheTileCompany is able to do the same for images and vision transformers. The algorithm we developed provides a semantics-preserving method for decreasing inference costs within vision models, while our library deploys it in a seamless, non-intrusive package.

Our benchmarks prove TheTileCompany algorithm is able to reliably achieve a 30% reduction in image size while enabling the reproduction of indistinguishable embeddings. In other words, we are able to reduce vision model inference costs over 30% while producing images that are, for machine learning purposes, indistinguishable from the original.

## How we built it

We built our approach around dissecting a lightweight Vision Transformer (ViT), leveraging its attention mechanism to implement our algorithm. Specifically, we use the ViT's attention scores as a proxy for pixel importance. This attention-driven representation allows us to identify low-importance paths: sequences of pixels that minimally impact semantic content. By prioritizing pixel pruning along these paths, our algorithm removes pixels that have a negligible effect on vision model performance while preserving structure and meaning. Notably, our lightweight ViT is much smaller and cheaper than running a full deployment model: enabling us to efficiently identify least important pixels.

To validate our results, we compared embedding cosine similarity scores and checked the visual accuracy of reproduced images. These evaluations confirmed that our pruning maintains semantic fidelity while significantly reducing input size, giving us confidence to move toward deployment.

On top of this library, we built a REST API using Flask, enabling integration with a full-stack NextJS website. We also deployed an experimental Streamlit application to showcase the library in action, including interaction with a chatbot, providing a hands-on demonstration of how attention-guided pixel pruning can streamline vision model inputs.

## Installation & Usage

### Backend (Flask API)

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

The API will run on `http://localhost:5001`.

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

The website will be available at `http://localhost:3000`.

### Streamlit Demo

For experimenting with passing compressed images into a chatbot:

```bash
cd sandbox
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

### Python SDK

```bash
cd python_sdk
pip install -e .
```

```python
from python_sdk import TileClient

client = TileClient(api_url="http://localhost:5001")
result = client.compress_image("path/to/image.jpg", reduction=0.3)
```

## Project Structure

```
TheTileCompany/
├── Backend/                # Flask API with DINOv2 compression
├── frontend/               # Next.js web application
├── sandbox/                # Streamlit demo application
├── python_sdk/             # Python client library
└── Image_Compression_Testing/  # Benchmarks and test images
```

## Challenges we ran into

Our first challenge was figuring out how to extract attention maps from the vision transformer partway through the model. Our first (and very time-consuming) attempt was to use an arXiv paper called HiPrune to prune the least important pixels after the attention layer. Their model was written for Windows, and we went as far as rewriting the setup scripts for macOS before deciding it wasn't worth the fight. That was when we pivoted to a local vision transformer model, which gave us full access to intermediate layers and let us actually work with the internal attention scores.

But even with a local model, smoothly discarding irrelevant pixels turned out to be much harder than we expected. Simply deleting them created a jagged mess of misaligned objects and often made the image unreadable. We tried Gaussian blur (hoping it would smooth out the attention map and reduce sharp discontinuities) and even a Sobel filter (using color gradients to preserve edges while compressing the rest), but both approaches failed and still produced heavy distortion.

The final solution that worked was a seam-carving algorithm guided by the attention matrix. Instead of deleting arbitrary low-importance pixels, seam carving removes continuous low-energy paths, which keeps the structure intact. This ended up working extremely well: we were able to remove over 30% of unimportant pixels while keeping the image virtually undistorted.

## Accomplishments that we're proud of

We're proud that we were actually able to extract pixel-level attention from a Vision Transformer and turn it into something useful. There isn't much documentation on accessing intermediate layers, so getting stable attention maps in the first place was a big milestone. We're also proud of how we creatively combined algorithms that normally don't appear together, like transformer attention scores with classical seam-carving, into a single pipeline that exceeded all expectations. The method not only removes a huge chunk of redundant pixels while preserving image semantics, but it also runs very fast. During testing, the entire process averaged around two seconds, and in many cases, you can barely tell anything was removed (despite massive savings).

## What we learned

Calvin got his first real exposure to ReactJS and web development and handled the frontend-backend integration way better than any of us expected on a first attempt. For both Dima and Calvin, this was their first hackathon, so making a full working project in 24 hours was a tremendously rewarding experience.

For all of us, this was the first time we went this deep into model internals instead of just treating LLMs as black boxes. We learned how much you can actually do when you explore intermediate layers directly rather than relying on prompt engineering alone. We also learned the importance of knowing when to pivot. We ran into multiple dead ends, and the only reason the final product exists is because we weren't afraid to abandon ideas that clearly weren't working.

## What's next for TheTileCompany

Our next step is extending the system to videos: we believe there are significant algorithmic innovations and systems level optimizations that can be performed. Processing videos will require pushing our processing speed even further and keeping frame-to-frame consistency so the pruning doesn't introduce flicker. We also want to experiment with additional approaches to process the attention layer data, including using CNN-based classifiers or image segmentation to help distinguish relevant and irrelevant regions. Finally, we plan to make the tool easier to deploy and use by offering multiple interfaces (API, CLI, Python package) and eventually exploring integrations with mainstream multimodal platforms like ChatGPT or Claude, where automatic pixel pruning could directly reduce token usage and inference cost for everyday users.

## Links

- **Devpost:** [https://devpost.com/software/visplaceholder](https://devpost.com/software/visplaceholder)
