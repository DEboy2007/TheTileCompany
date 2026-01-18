"""
Flask application entry point.
Uses the API module for all routes.
"""

from api import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)