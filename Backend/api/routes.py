"""
Flask API routes for image compression service.
"""

from flask import Flask, request, jsonify
from .service import compress_image


def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)

    @app.route('/compress', methods=['POST'])
    def compress():
        """
        Compress image using attention-guided seam carving.

        Request body:
        {
            "image": "url_or_path",
            "reduction": 0.3,  # optional, default 0.3
            "threshold": 0.3  # optional, default 0.3
        }

        Returns:
        {
            "status": 0,
            "reduction_pct": 30.5,
            "gray_overlay_base64": "...",
            "compressed_image_base64": "...",
            "stats": {...}
        }
        """
        try:
            data = request.get_json()

            image = data.get('image')
            reduction = data.get('reduction', 0.3)
            threshold = data.get('threshold', 0.3)

            if not image:
                return jsonify({
                    "status": 1,
                    "message": "Missing required field: image"
                }), 400

            result = compress_image(
                image_path_or_url=image,
                reduction=reduction,
                threshold=threshold
            )

            return jsonify({
                "status": 0,
                "reduction_pct": result['reduction_pct'],
                "gray_overlay_base64": result['gray_overlay_base64'],
                "compressed_image_base64": result['compressed_image_base64'],
                "stats": result['stats']
            })

        except Exception as e:
            return jsonify({
                "status": 1,
                "message": str(e)
            }), 500

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({"status": "ok"})

    return app
