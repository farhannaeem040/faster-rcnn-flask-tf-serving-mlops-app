# ──────────────────────────────────────────────────────────────
# config.py  –  Central settings for the whole application
# ──────────────────────────────────────────────────────────────
# Instead of scattering magic numbers and folder paths all over
# the code, we keep every setting here.  Any other file can just
# import what it needs from this one place.
# os.getenv() lets you override any value using an environment
# variable — super handy when running inside Docker.

from __future__ import annotations

import os
from pathlib import Path

# ── Folder layout ─────────────────────────────────────────────

# Figure out where THIS file lives on disk.
# All other folder paths are built relative to this location,
# so the app works no matter where it is installed.
BASE_DIR = Path(__file__).resolve().parent

# Where uploaded (unmodified) images are stored.
ORIGINALS_FOLDER = BASE_DIR / 'originals'

# Where images with bounding boxes drawn on them are stored.
PROCESSED_FOLDER = BASE_DIR / 'processed'

# Where performance metric JSON files live.
METRICS_DIR = BASE_DIR / 'metrics'

# Where log files are written.
LOGS_DIR = BASE_DIR / 'logs'

# ── File paths ─────────────────────────────────────────────────

# The single JSON file that stores request/accuracy statistics.
METRICS_FILE = METRICS_DIR / 'metrics.json'

# ── Upload rules ───────────────────────────────────────────────

# Only these image types are accepted when a user uploads a photo.
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Only ZIP files are accepted when uploading a new model.
ALLOWED_MODEL_EXTENSIONS = {'.zip'}

# ── Detection tuning ───────────────────────────────────────────

# Any detection with a confidence score below this value is thrown away.
# 0.30 means "at least 30% confident before we show the result".
# You can raise this (e.g. 0.50) to only see high-confidence detections.
MIN_SCORE_THRESHOLD = float(os.getenv('MIN_SCORE_THRESHOLD', '0.30'))

# ── Upload size limit ──────────────────────────────────────────

# Maximum size (in bytes) of a file that can be uploaded.
# Defaults to 2 GB which is large enough for most model ZIP files.
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', str(2048 * 1024 * 1024)))

# ── TensorFlow Serving connection ─────────────────────────────

# Path to the folder on disk that holds numbered model versions
# (e.g.  model/object_detection/1/,  model/object_detection/2/).
# When running inside Docker this is usually a shared volume.
MODEL_DIR = Path(os.getenv('MODEL_DIR', str(BASE_DIR.parent / 'model' / 'object_detection')))

# The REST endpoint that TF Serving exposes for running predictions.
TF_SERVING_URL = os.getenv(
    'TF_SERVING_URL',
    'http://tf-serving:8501/v1/models/object_detection:predict'
)

# The REST endpoint used to check whether the model is loaded and ready.
TF_MODEL_STATUS_URL = os.getenv(
    'TF_MODEL_STATUS_URL',
    'http://tf-serving:8501/v1/models/object_detection'
)

# How many seconds to wait for TF Serving before giving up.
REQUEST_TIMEOUT_SECONDS = int(os.getenv('REQUEST_TIMEOUT_SECONDS', '120'))

# ── Security ───────────────────────────────────────────────────

# Flask uses this secret key to sign session cookies.
# In production you MUST set this to a long, random string via environment variable.
APP_SECRET_KEY = os.getenv('APP_SECRET_KEY', 'development-secret-key')

# ── Logging ────────────────────────────────────────────────────

# Controls how verbose the logs are.
# Common values: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# The file where application log messages are written.
LOG_FILE = LOGS_DIR / 'application.log'
