from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        body = data.get('body')
        compression = data.get('compression_level')


        if not isinstance(body, str) or not isinstance(compression, int):
            return jsonify({
                "content": "",
                "status": 1,  # error
                "message": "Incorrect inputs"
            }), 400

        content = body

        return jsonify({
            "content": content,
            "status": 0  # success
        })

    except Exception as e:
        return jsonify({
            "content": str(e),
            "status": 1,
            "message": e
        }), 500

if __name__ == '__main__':
    app.run(debug=True)