"""Application factory for the Prompt Optimizer Flask service."""

from flask import Flask

from app.api.routes import api_blueprint
from app.core.config import Config


def create_app(config_object: type[Config] = Config) -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(api_blueprint)
    return app
