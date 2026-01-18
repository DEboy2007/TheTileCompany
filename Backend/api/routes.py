"""
Flask API routes for compression service.
"""

from flask import Flask, request, jsonify
from .service import run_compression
from .compression.image_compression import (
    load_image,
    get_attention_map,
    compress_image,
    image_to_base64,
    get_dino_model
)


def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)

    @app.route('/compress', methods=['POST'])
    def compress():
        """
        Compress image and prompt, then query LLM.

        Request body:
        {
            "image": "url_or_path",
            "prompt": "What is in this image?",
            "threshold": 0.3,  # optional, default 0.3
            "aggressiveness": 0.5  # optional, default 0.5
        }
        """
        try:
            data = request.get_json()

            image = data.get('image')
            prompt = data.get('prompt')
            threshold = data.get('threshold', 0.3)
            aggressiveness = data.get('aggressiveness', 0.5)

            if not image or not prompt:
                return jsonify({
                    "status": 1,
                    "message": "Missing required fields: image and prompt"
                }), 400

            result = run_compression(
                image_path_or_url=image,
                prompt=prompt,
                threshold=threshold,
                aggressiveness=aggressiveness
            )

            return jsonify({
                "status": 0,
                "answer": result['answer'],
                "compressed_prompt": result['compressed_prompt'],
                "image_stats": result['image_stats'],
                "prompt_stats": result['prompt_stats']
            })

        except Exception as e:
            return jsonify({
                "status": 1,
                "message": str(e)
            }), 500

    @app.route('/compress-image', methods=['POST'])
    def compress_image_only():
        """
        Compress image and return the compressed image.

        Request body:
        {
            "image": "url_or_path",
            "threshold": 0.3  # optional, default 0.3
        }

        Response:
        {
            "status": 0,
            "compressed_image": "base64_encoded_image",
            "image_stats": {
                "original_size": [width, height],
                "compressed_size": [width, height],
                "compression_ratio": float
            }
        }
        """
        try:
            data = request.get_json()

            image = data.get('image')
            threshold = data.get('threshold', 0.3)

            if not image:
                return jsonify({
                    "status": 1,
                    "message": "Missing required field: image"
                }), 400

            # Load and compress image
            img, original_size = load_image(image)

            model = get_dino_model()
            attention_map, _ = get_attention_map(model, img)
            compressed_img, image_stats = compress_image(img, attention_map, threshold)

            # Convert compressed image to base64
            img_base64 = image_to_base64(compressed_img)

            return jsonify({
                "status": 0,
                "compressed_image": img_base64,
                "image_stats": image_stats
            })

        except Exception as e:
            return jsonify({
                "status": 1,
                "message": str(e)
            }), 500

    # @app.route('/process', methods=['POST'])
    # def process_text():
    #     """
    #     Simple text processing endpoint (legacy compatibility).

    #     Request body:
    #     {
    #         "body": "text to process",
    #         "compression_level": 1
    #     }
    #     """
    #     try:
    #         data = request.get_json()
    #         body = data.get('body')
    #         compression = data.get('compression_level')

    #         if not isinstance(body, str) or not isinstance(compression, int):
    #             return jsonify({
    #                 "content": "",
    #                 "status": 1,
    #                 "message": "Incorrect inputs"
    #             }), 400

    #         content = body

    #         return jsonify({
    #             "content": content,
    #             "status": 0
    #         })

    #     except Exception as e:
    #         return jsonify({
    #             "content": str(e),
    #             "status": 1,
    #             "message": str(e)
    #         }), 500

    return app
