from __future__ import annotations

# Import the Flask class from the flask module. It is used to create the Flask application instance.
from flask import Flask

# Import the MAX_CONTENT_LENGTH and APP_SECRET_KEY constants from the config file
from config import MAX_CONTENT_LENGTH, APP_SECRET_KEY
# Import the configure_logging function from the logging_config module
from logging_config import configure_logging
# Import the bp blueprint from the routes module
from routes import bp
# Import the ensure_runtime_directories and initialize_metrics_file functions from the utils module
from utils import ensure_runtime_directories, initialize_metrics_file


def create_app() -> Flask:
    # Create the Flask application instance
    app = Flask(__name__)
    # Set the secret key for the application
    app.config['SECRET_KEY'] = APP_SECRET_KEY
    # Set the maximum content length for the application
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

    # Ensure runtime directories exist
    ensure_runtime_directories()
    # Initialize the metrics file
    initialize_metrics_file()
    # Configure logging
    configure_logging(app)

    # Register the blueprint. The purpose of blue prints is to register routes to the application in a modular way.
    app.register_blueprint(bp)
    # Log that the application has been initialized successfully
    app.logger.info('Flask application initialized successfully.')
    # Return the application instance
    return app
