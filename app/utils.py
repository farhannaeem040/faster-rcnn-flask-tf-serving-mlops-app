"""Utility helpers for validation, image handling, rendering, metrics and parsing."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    LOGS_DIR,
    METRICS_FILE,
    METRICS_DIR,
    MIN_SCORE_THRESHOLD,
    ORIGINALS_FOLDER,
    PROCESSED_FOLDER,
)

logger = logging.getLogger(__name__)

# Compact COCO label map for Faster R-CNN COCO models.
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# Ensure runtime directories exist
def ensure_runtime_directories() -> None:
    for folder in [ORIGINALS_FOLDER, PROCESSED_FOLDER, METRICS_DIR, LOGS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

# Default metrics
def default_metrics() -> Dict[str, Any]:
    return {
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'average_latency_ms': 0.0,
        'last_latency_ms': None,
        'last_request_at': None,
        'last_error': None,
        'total_detections': 0,
        'history': [],
    }

# Initialize metrics file
def initialize_metrics_file() -> None:
    if not METRICS_FILE.exists():
        METRICS_FILE.write_text(json.dumps(default_metrics(), indent=2), encoding='utf-8')

# Check if file is allowed
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

# Sanitize filename
def sanitize_filename(filename: str) -> str:
    return secure_filename(filename)

# Load metrics
def load_metrics() -> Dict[str, Any]:
    if not METRICS_FILE.exists():
        return default_metrics()
    try:
        return json.loads(METRICS_FILE.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        logger.warning('Metrics file is corrupted. Resetting metrics file.')
        return default_metrics()

# Save metrics
def save_metrics(metrics: Dict[str, Any]) -> None:
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

# Update metrics
def update_metrics(latency_ms: float, success: bool, detection_count: int = 0, error_message: str | None = None) -> None:
    metrics = load_metrics()
    metrics['total_requests'] += 1
    metrics['last_latency_ms'] = round(latency_ms, 2)
    metrics['last_request_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    metrics['last_error'] = error_message

    if success:
        metrics['successful_requests'] += 1
        metrics['total_detections'] += detection_count
        successes = metrics['successful_requests']
        prior_total = metrics['average_latency_ms'] * (successes - 1)
        metrics['average_latency_ms'] = round((prior_total + latency_ms) / successes, 2)
    else:
        metrics['failed_requests'] += 1

    metrics['history'].append({
        'timestamp': metrics['last_request_at'],
        'latency_ms': round(latency_ms, 2),
        'success': success,
        'detection_count': detection_count,
        'error': error_message,
    })
    metrics['history'] = metrics['history'][-30:]
    save_metrics(metrics)

# Convert image to serving payload
def image_to_serving_payload(image_path: Path) -> Dict[str, Any]:
    """Convert image into TF Serving payload.

    Faster R-CNN TF2 exports commonly accept a uint8 tensor shaped [1, H, W, 3].
    With the REST API, that means one item inside the instances list.
    """
    image = Image.open(image_path).convert('RGB')
    image_array = np.asarray(image, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f'Expected RGB image tensor with shape [H, W, 3], got {image_array.shape}')
    # Convert image to serving payload
    pixels = image_array.tolist()
    # Log that the image has been converted to serving payload
    logger.info(
        'Converted image %s to payload with shape %s and size %sx%s',
        image_path.name,
        image_array.shape,
        image.width,
        image.height,
    )
    # Return the serving payload
    return {'instances': [pixels]}

# Coerce prediction root
def _coerce_prediction_root(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Handle common TF Serving response shapes for object detection models."""
    if 'predictions' in predictions and predictions['predictions']:
        first = predictions['predictions'][0]
        if isinstance(first, dict):
            return first
        if isinstance(first, list):
            return {'raw': first}
    if 'outputs' in predictions and isinstance(predictions['outputs'], dict):
        return predictions['outputs']
    return predictions


def normalize_detections(predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Faster R-CNN outputs into a template-friendly list.

    Supported shapes:
    - {'predictions': [{'detection_boxes': ..., 'detection_scores': ..., 'detection_classes': ...}]}
    - {'outputs': {...}}
    """
    root = _coerce_prediction_root(predictions)
    boxes = root.get('detection_boxes', [])
    scores = root.get('detection_scores', [])
    classes = root.get('detection_classes', [])
    num_detections = root.get('num_detections', len(scores))

    if isinstance(num_detections, list):
        try:
            num_detections = int(num_detections[0])
        except Exception:
            num_detections = len(scores)
    else:
        try:
            num_detections = int(num_detections)
        except Exception:
            num_detections = len(scores)

    detections: List[Dict[str, Any]] = []
    for index, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        if index >= num_detections:
            break
        score = float(score)
        if score < MIN_SCORE_THRESHOLD:
            continue
        class_id = int(float(cls))
        ymin, xmin, ymax, xmax = [float(v) for v in box]
        detections.append({
            'class_id': class_id,
            'label': COCO_LABELS.get(class_id, f'class_{class_id}'),
            'score': round(score, 4),
            'box': [ymin, xmin, ymax, xmax],
        })

    logger.info('Normalized %s detections above threshold %.2f', len(detections), MIN_SCORE_THRESHOLD)
    return detections


def draw_boxes_on_image(original_path: Path, output_path: Path, detections: List[Dict[str, Any]]) -> None:
    image = Image.open(original_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()

    for detection in detections:
        ymin, xmin, ymax, xmax = detection['box']
        left, top = xmin * width, ymin * height
        right, bottom = xmax * width, ymax * height
        label = f"{detection['label']} ({detection['score']:.2f})"

        draw.rectangle([(left, top), (right, bottom)], outline='red', width=3)
        text_top = max(0, top - 14)
        draw.rectangle([(left, text_top), (left + max(80, len(label) * 7), top)], fill='red')
        draw.text((left + 2, text_top + 1), label, fill='white', font=font)

    image.save(output_path)
    logger.info('Saved processed image with bounding boxes: %s', output_path.name)


import zipfile
import shutil
from config import MODEL_DIR

def extract_and_deploy_model(zip_path: Path) -> int:
    """Extract a new model version zip and deploy it to the TF model directory.
    Returns the new version number."""
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
    current_versions = []
    for d in MODEL_DIR.iterdir():
        if d.is_dir() and d.name.isdigit():
            current_versions.append(int(d.name))
            
    head_version = max(current_versions) if current_versions else 0
    new_version = head_version + 1
    target_dir = MODEL_DIR / str(new_version)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Extracting new model to %s", target_dir)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Check if saved_model.pb is at the root. Handles wrapped directory scenarios.
        pb_file = target_dir / "saved_model.pb"
        if not pb_file.exists():
            for f in target_dir.rglob("saved_model.pb"):
                base = f.parent
                for item in base.iterdir():
                    shutil.move(str(item), str(target_dir / item.name))
                break
                
        logger.info("Model version %s deployed successfully.", new_version)
        return new_version
    except Exception as e:
        logger.error("Failed to deploy model: %s", e)
        shutil.rmtree(target_dir, ignore_errors=True)
        raise e
