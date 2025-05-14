from flask import Flask
from flask_cors import CORS
from .main import app


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
    CORS(app)

    # Enregistrement du blueprint
    app.register_blueprint(main)

    return app
