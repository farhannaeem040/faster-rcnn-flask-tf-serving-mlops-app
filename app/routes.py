# ──────────────────────────────────────────────────────────────
# routes.py  –  Every URL the web app responds to lives here
# ──────────────────────────────────────────────────────────────
# Flask uses "routes" to connect a URL (like /predict) to a Python
# function that decides what to do and what HTML page to show.
# We group all routes into a Blueprint called 'bp' so the factory
# in app_factory.py can register them all at once cleanly.

from __future__ import annotations

import logging
from pathlib import Path

# Flask helpers:
#   Blueprint       – groups related routes together
#   current_app     – lets us access the running app inside a route
#   flash           – sends a one-time message to show on the next page
#   redirect        – sends the browser to a different URL
#   render_template – fills in an HTML template and returns it
#   request         – holds everything the browser sent (files, form data, etc.)
#   send_from_directory – safely serves a file from a folder
#   url_for         – builds a URL from a route function name (avoids hard-coding URLs)
from flask import Blueprint, current_app, flash, redirect, render_template, request, send_from_directory, url_for

# Folders where uploaded and processed images are stored.
from config import ORIGINALS_FOLDER, PROCESSED_FOLDER, LOG_FILE

# Functions that talk to the TensorFlow Serving inference server.
from tf_serving_client import TFServingError, get_model_status, predict_image

# Our utility helpers for validation, image processing, metrics, and model deployment.
from utils import (
    allowed_file,
    draw_boxes_on_image,
    load_metrics,
    normalize_detections,
    sanitize_filename,
    update_metrics,
    extract_and_deploy_model
)

# tempfile lets us create a safe temporary file for the uploaded model ZIP.
import tempfile

# Create the Blueprint.  'main' is just the internal name we use with url_for().
bp = Blueprint('main', __name__)

# Each module gets its own named logger so we can filter log output by module.
logger = logging.getLogger(__name__)


# ── Home page ─────────────────────────────────────────────────────────────────
@bp.route('/', methods=['GET'])
def index():
    """Show the home/upload page with a summary of overall statistics."""
    logger.info('Rendering index page.')
    # Load the latest metrics (request count, success rate, etc.) to display on the page.
    metrics_data = load_metrics()
    return render_template('index.html', metrics=metrics_data)


# ── Upload & predict ──────────────────────────────────────────────────────────
@bp.route('/predict', methods=['POST'])
def predict():
    """Accept an uploaded image, run object detection, and show the results.

    This is the main "action" of the whole application.
    Steps:
      1. Validate the uploaded file.
      2. Save the original image to disk.
      3. Send the image to TF Serving for inference.
      4. Draw coloured bounding boxes on the image.
      5. Update the metrics log.
      6. Show the results page.
    """
    logger.info('Prediction request received.')

    # ── Step 1: Validate the upload ───────────────────────────
    # Check that the form actually included a file field called 'image'.
    if 'image' not in request.files:
        logger.warning('Prediction aborted: no image field in request.')
        flash('No image was uploaded.', 'error')
        return redirect(url_for('main.index'))

    file = request.files['image']

    # A blank filename means the user clicked Submit without choosing a file.
    if file.filename == '':
        logger.warning('Prediction aborted: empty filename.')
        flash('Please choose an image file first.', 'error')
        return redirect(url_for('main.index'))

    # Make sure the file extension is one we accept (.jpg, .jpeg, .png).
    if not allowed_file(file.filename):
        logger.warning('Prediction aborted: invalid file extension for %s', file.filename)
        flash('Invalid file type. Only .jpg, .jpeg and .png are allowed.', 'error')
        return redirect(url_for('main.index'))

    # ── Step 2: Save the original file ────────────────────────
    # Clean the filename to remove any dangerous characters (e.g. "../").
    filename = sanitize_filename(file.filename)
    original_path  = ORIGINALS_FOLDER / filename
    processed_filename = f'processed_{filename}'
    processed_path = PROCESSED_FOLDER / processed_filename

    # Write the raw upload to the originals folder.
    file.save(original_path)
    logger.info('Original image saved to %s', original_path)

    # ── Steps 3–5: Run inference, draw boxes, record metrics ──
    try:
        # Ask TF Serving to detect objects in the image.
        # raw_prediction contains the model's output; latency_ms is how long it took.
        raw_prediction, latency_ms = predict_image(original_path)

        # Convert the raw model output into a clean list of detections
        # (each item has label, confidence score, and bounding box coordinates).
        detections = normalize_detections(raw_prediction)

        # Draw red bounding boxes (and labels) on a copy of the image.
        draw_boxes_on_image(original_path, processed_path, detections)

        # Record this successful request in our metrics file.
        update_metrics(latency_ms=latency_ms, success=True, detection_count=len(detections))

        logger.info(
            'Prediction completed successfully for %s with %s detections.',
            filename, len(detections)
        )

    except TFServingError as exc:
        # TF Serving explicitly returned an error (e.g. model not loaded).
        logger.exception('TensorFlow Serving inference failed for %s.', filename)
        update_metrics(latency_ms=0.0, success=False, error_message=str(exc))
        flash(str(exc), 'error')
        return redirect(url_for('main.index'))

    except Exception as exc:
        # Something else went wrong (network issue, disk full, etc.).
        logger.exception('Unexpected error during prediction for %s.', filename)
        update_metrics(latency_ms=0.0, success=False, error_message=str(exc))
        flash(f'Unexpected error: {exc}', 'error')
        return redirect(url_for('main.index'))

    # ── Step 6: Show the results page ─────────────────────────
    # Pass the image URLs and detection list to the results template.
    return render_template(
        'results.html',
        image_name=filename,
        original_image=url_for('main.original_file', filename=filename),
        processed_image=url_for('main.processed_file', filename=processed_filename),
        detections=detections,
        detection_count=len(detections),
    )


# ── Metrics dashboard ─────────────────────────────────────────────────────────
@bp.route('/metrics', methods=['GET'])
def metrics():
    """Show the performance metrics page (request counts, latency, history)."""
    logger.info('Rendering metrics page.')
    metrics_data = load_metrics()
    return render_template('metrics.html', metrics=metrics_data)


# ── Image gallery ─────────────────────────────────────────────────────────────
@bp.route('/gallery', methods=['GET'])
def gallery():
    """Show a gallery of all previously processed (annotated) images."""
    logger.info('Rendering gallery page.')
    try:
        if PROCESSED_FOLDER.exists():
            # Find every image file in the processed folder.
            files = [f for f in PROCESSED_FOLDER.iterdir() if f.is_file() and allowed_file(f.name)]
            # Sort newest first so the most recent predictions appear at the top.
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            image_filenames = [f.name for f in files]
        else:
            image_filenames = []
    except Exception as exc:
        logger.exception('Failed to construct gallery list: %s', exc)
        image_filenames = []
        flash('Could not load gallery.', 'error')

    return render_template('gallery.html', images=image_filenames)


# ── Log viewer ────────────────────────────────────────────────────────────────
@bp.route('/logs', methods=['GET'])
def view_logs():
    """Read the application log file and display the most recent 1 000 lines."""
    logger.info('Rendering logs page.')
    log_content = ""

    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Only keep the last 1 000 lines to avoid sending a huge payload
                # to the browser when the log file grows very large.
                log_content = "".join(lines[-1000:])
        except Exception as e:
            log_content = f"Error reading logs: {e}"
            logger.error(log_content)
    else:
        log_content = "Log file does not exist yet."

    return render_template('logs.html', log_content=log_content)


# ── Model settings / upload ───────────────────────────────────────────────────
@bp.route('/settings', methods=['GET', 'POST'])
def settings():
    """Handle the settings page where a new model version can be uploaded.

    GET  → show the settings form with a list of deployed model versions.
    POST → validate the uploaded ZIP, extract it, and deploy the new version.
    """
    if request.method == 'POST':
        logger.info('Model update request received.')

        # Make sure the form included a file field called 'model_zip'.
        if 'model_zip' not in request.files:
            flash('No compression file uploaded.', 'error')
            return redirect(url_for('main.settings'))

        file = request.files['model_zip']

        # Reject blank submissions (the user didn't pick a file).
        if file.filename == '':
            flash('Empty filename given.', 'error')
            return redirect(url_for('main.settings'))

        # Only ZIP files are accepted — the model must be compressed.
        if not file.filename.lower().endswith('.zip'):
            flash('Invalid format! Models must be packed into a .zip container housing the .pb file.', 'error')
            return redirect(url_for('main.settings'))

        try:
            # Save the ZIP to a temporary file first.
            # Using a temp file is safer than holding the whole upload in memory.
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                file.save(tmp.name)
                tmp_path = Path(tmp.name)

            # Extract the ZIP and move the model files into the correct versioned folder.
            new_v = extract_and_deploy_model(tmp_path)

            # Delete the temporary ZIP now that we're done with it.
            tmp_path.unlink()

            flash(
                f'Model upgrade deployed successfully! '
                f'TensorFlow Serving is cycling to isolated Engine Version {new_v}.',
                'success'
            )
        except Exception as e:
            flash(f'Model structure decoding failed: {e}', 'error')

        # Always redirect after a POST to prevent duplicate submissions if the
        # user presses the browser's Back button.
        return redirect(url_for('main.settings'))

    # ── GET: show the settings page ───────────────────────────
    # Build a list of all version numbers that have been deployed so far.
    versions = []
    from config import MODEL_DIR
    if MODEL_DIR.exists():
        # Folder names that are pure numbers (1, 2, 3 …) are model versions.
        # Sort them highest-first so the newest version appears at the top.
        versions = sorted(
            [d.name for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
            key=int,
            reverse=True
        )

    return render_template('settings.html', versions=versions)


# ── Health check ──────────────────────────────────────────────────────────────
@bp.route('/health', methods=['GET'])
def health():
    """Check whether TF Serving is up and the model is loaded.

    Returns HTTP 200 if everything is fine, HTTP 500 if TF Serving is down.
    Useful for Docker health checks and monitoring dashboards.
    """
    try:
        status = get_model_status()
        logger.info('Health check succeeded.')
        return render_template('health.html', status='ok', tf_serving=status)
    except Exception as exc:
        logger.exception('Health check failed.')
        return render_template('health.html', status='error', error_message=str(exc)), 500


# ── Generic error page ────────────────────────────────────────────────────────
@bp.route('/error', methods=['GET'])
def error_page():
    """Render a user-friendly error page.

    The error code and message are passed as URL query parameters, e.g.:
      /error?code=403&message=Forbidden
    """
    code    = request.args.get('code', 'Error')
    message = request.args.get('message', 'An unexpected error occurred.')
    return render_template('error.html', code=code, message=message)


# ── Static file serving ───────────────────────────────────────────────────────
@bp.route('/originals/<path:filename>')
def original_file(filename: str):
    """Serve an original (unmodified) uploaded image to the browser."""
    return send_from_directory(ORIGINALS_FOLDER, filename)


@bp.route('/processed/<path:filename>')
def processed_file(filename: str):
    """Serve a processed image (with bounding boxes drawn) to the browser."""
    return send_from_directory(PROCESSED_FOLDER, filename)


# ── Custom HTTP error handlers ────────────────────────────────────────────────
@bp.app_errorhandler(404)
def page_not_found(error):
    """Show a friendly 'page not found' message instead of Flask's default."""
    return render_template('error.html', code=404, message='The requested page could not be found.'), 404


@bp.app_errorhandler(500)
def internal_server_error(error):
    """Show a friendly 'something went wrong' message for unexpected server errors."""
    return render_template('error.html', code=500, message='An internal server error occurred.'), 500
