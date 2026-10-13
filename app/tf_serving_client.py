"""Client wrapper for TensorFlow Serving REST endpoints."""
# ──────────────────────────────────────────────────────────────
# tf_serving_client.py  –  Talks to the TensorFlow Serving container
# ──────────────────────────────────────────────────────────────
# TensorFlow Serving (TF Serving) is a separate process (or Docker
# container) that holds our trained model and runs inference.
# This file is the only place in the code that communicates with it.
# All HTTP calls go through the two functions below, keeping the
# rest of the app blissfully unaware of networking details.

from __future__ import annotations

import logging
# Path makes it easy to pass file locations around without string gymnastics.
from pathlib import Path
# Type hints — just here to make the code more readable and catch bugs.
from typing import Any, Dict, Tuple

# requests is the standard Python library for making HTTP calls.
import requests

# Our central config: the URLs for TF Serving and the timeout setting.
from config import REQUEST_TIMEOUT_SECONDS, TF_MODEL_STATUS_URL, TF_SERVING_URL

# Helper that converts a raw image file into the JSON format TF Serving expects.
from utils import image_to_serving_payload

# Every module gets its own named logger so log messages show where they came from.
logger = logging.getLogger(__name__)


# ── Custom exception ──────────────────────────────────────────────────────────
class TFServingError(RuntimeError):
    """Raised whenever TensorFlow Serving refuses or fails a request.

    By using our own exception type, the rest of the app can catch
    TFServingError specifically instead of a generic Exception, which makes
    error handling much cleaner.
    """


# ── Health / status check ─────────────────────────────────────────────────────
def get_model_status(timeout_seconds: int = 15) -> Dict[str, Any]:
    """Ask TF Serving whether the model is loaded and ready.

    Returns the raw JSON status dictionary from TF Serving.
    Raises an exception if TF Serving is unreachable or returns an error.
    """
    logger.info('Checking TensorFlow Serving model status at %s', TF_MODEL_STATUS_URL)

    # Send a simple GET request to the status endpoint.
    response = requests.get(TF_MODEL_STATUS_URL, timeout=timeout_seconds)

    # If TF Serving returned a 4xx or 5xx status code, this will throw
    # an exception automatically — no need to check manually.
    response.raise_for_status()

    # Convert the response body from JSON text into a Python dictionary.
    return response.json()


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_image(image_path: Path, timeout_seconds: int = REQUEST_TIMEOUT_SECONDS) -> Tuple[Dict[str, Any], float]:
    """Send an image to TF Serving and get back the detection results.

    Returns a tuple of:
      - predictions:  a dictionary with bounding boxes, scores, and class IDs
      - latency_ms:   how long TF Serving took to respond, in milliseconds
    """
    logger.info('Preparing payload for image: %s', image_path.name)

    # Turn the image file into the specific JSON structure TF Serving needs.
    # (An RGB pixel array wrapped in an 'instances' key.)
    payload = image_to_serving_payload(image_path)

    logger.info('Sending inference request to TensorFlow Serving: %s', TF_SERVING_URL)

    # POST the payload to TF Serving and wait up to timeout_seconds for a reply.
    response = requests.post(TF_SERVING_URL, json=payload, timeout=timeout_seconds)

    # Calculate how long the whole round-trip took (in milliseconds).
    # This gets stored in our metrics so we can track model speed over time.
    latency_ms = response.elapsed.total_seconds() * 1000

    logger.info(
        'TensorFlow Serving responded in %.2f ms with status code %s',
        latency_ms,
        response.status_code
    )

    # Any status code other than 200 means something went wrong.
    # We raise our custom error so the calling code can show a friendly message.
    if response.status_code != 200:
        logger.error('TensorFlow Serving request failed. Body: %s', response.text)
        raise TFServingError(
            f'TensorFlow Serving request failed with status {response.status_code}: {response.text}'
        )

    # Return the prediction dictionary AND the latency so the caller can log both.
    return response.json(), latency_ms
