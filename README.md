# Faster RCNN Object Detection App

A full-stack MLOps web application that lets you upload a photo and instantly see
what objects the AI found — with coloured bounding boxes drawn around each one.
Built with **Python + Flask** on the front-end and **TensorFlow Serving** running
the actual AI model in the background, all wired together with **Docker Compose**.

---

## Table of Contents

1. [How it Works (Big Picture)](#1-how-it-works-big-picture)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Setting Up the AI Model](#4-setting-up-the-ai-model)
5. [Running the App (Docker — Recommended)](#5-running-the-app-docker--recommended)
6. [Running Just the Website (No Docker)](#6-running-just-the-website-no-docker)
7. [Environment Variables](#7-environment-variables)
8. [Page-by-Page Guide](#8-page-by-page-guide)
   - [Home / Upload](#-home--upload----localhost5000)
   - [Results](#-results----localhost5000predict)
   - [Metrics Dashboard](#-metrics-dashboard----localhost5000metrics)
   - [Inference Gallery](#-inference-gallery----localhost5000gallery)
   - [Server Logs](#-server-logs----localhost5000logs)
   - [Health Check](#-health-check----localhost5000health)
   - [Model Manager](#-model-manager--settings----localhost5000settings)
   - [Error Page](#-error-page)
9. [Uploading a New Model (Hot-Swap)](#9-uploading-a-new-model-hot-swap)
10. [Where Files Are Stored](#10-where-files-are-stored)
11. [Troubleshooting](#11-troubleshooting)
12. [Technical Notes](#12-technical-notes)

---

## 1. How it Works (Big Picture)

```
  You (Browser)
       │
       │  Upload image via web form
       ▼
  Flask Web App  (port 5000)
       │
       │  Converts image → JSON pixel array
       │  Sends HTTP POST request
       ▼
  TensorFlow Serving  (port 8501)
       │
       │  Runs Faster R-CNN inference
       │  Returns bounding boxes + confidence scores
       ▼
  Flask Web App
       │
       │  Draws coloured boxes on the image
       │  Saves metrics to JSON file
       │  Returns results page to you
       ▼
  You (Browser)  — sees annotated image + detection table
```

The two services (Flask + TF Serving) run as separate Docker containers and
talk to each other over an internal Docker network.  You only ever interact
with the Flask app on port 5000.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Python 3 · Flask 3.0 |
| AI inference server | TensorFlow Serving (Docker image) |
| AI model type | Faster R-CNN (COCO dataset — 90 object classes) |
| Image processing | Pillow (PIL) · NumPy |
| Frontend charts | Chart.js (loaded from CDN) |
| CSS | Vanilla CSS with custom design system |
| Containerisation | Docker · Docker Compose |
| HTTP client | Python `requests` library |
| Logging | Python `logging` module with rotating file handler |

---

## 3. Project Structure

```
mlops-object-detection-app/
│
├── docker-compose.yml          ← Starts both containers with one command
├── requirements.txt            ← Python package dependencies
│
├── docker/
│   └── Dockerfile.flask        ← Build instructions for the Flask container
│
├── model/
│   └── object_detection/       ← Put your AI model files here
│       └── 1/                  ← Version number folder (TF Serving requirement)
│           ├── saved_model.pb
│           └── variables/
│
└── app/
    ├── main.py                 ← Entry point — starts the Flask server
    ├── app_factory.py          ← Builds and configures the Flask app
    ├── config.py               ← All settings and folder paths in one place
    ├── routes.py               ← Every URL the app responds to
    ├── tf_serving_client.py    ← Handles all communication with TF Serving
    ├── utils.py                ← Helper functions (image processing, metrics, etc.)
    ├── logging_config.py       ← Sets up log file and console output
    │
    ├── templates/              ← HTML pages (Jinja2 templating)
    │   ├── base.html           ← Shared layout (navbar, header, footer)
    │   ├── index.html          ← Home / upload page
    │   ├── results.html        ← Detection results page
    │   ├── metrics.html        ← Analytics dashboard
    │   ├── gallery.html        ← Previously processed images
    │   ├── logs.html           ← Server log viewer
    │   ├── health.html         ← System health / diagnostics
    │   ├── settings.html       ← Model upload / management
    │   └── error.html          ← Friendly error page (404, 500, etc.)
    │
    ├── static/
    │   └── css/style.css       ← All styling for the entire app
    │
    ├── originals/              ← Auto-created: stores uploaded images as-is
    ├── processed/              ← Auto-created: stores images with boxes drawn on
    ├── metrics/                ← Auto-created: stores metrics.json
    └── logs/                   ← Auto-created: stores application.log
```

> **Note:** The `originals/`, `processed/`, `metrics/`, and `logs/` folders are
> created automatically when the app starts for the first time. You do not need
> to create them manually.

---

## 4. Setting Up the AI Model

Before starting the app you must place a trained TensorFlow SavedModel into the
correct folder.  TF Serving requires version numbers as folder names (1, 2, 3…).

Place your files exactly like this:

```
model/
└── object_detection/
    └── 1/
        ├── saved_model.pb
        └── variables/
            ├── variables.data-00000-of-00001
            └── variables.index
```

- `saved_model.pb` is the main model file exported from TensorFlow.
- The `variables/` folder holds the model's trained weights.
- The folder name `1` is the version number — TF Serving will serve the
  highest-numbered version as the active model.

> **Compatible models:** This app is built for **Faster R-CNN** models trained
> on the COCO dataset.  The model must output `detection_boxes`,
> `detection_scores`, and `detection_classes` tensors.

---

## 5. Running the App (Docker — Recommended)

This is the easiest way.  One command starts both containers (Flask + TF Serving).

**Requirements:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
must be installed and running.

```bash
# Step 1 — Open a terminal and navigate to the project folder
cd mlops-object-detection-app

# Step 2 — Build the images and start both containers
docker-compose up --build
```

Wait until you see a message like `Flask application initialized successfully.`
in the terminal output.  Then open your browser:

| Page | URL |
|---|---|
| Home (upload images) | http://localhost:5000 |
| Metrics dashboard | http://localhost:5000/metrics |
| Inference gallery | http://localhost:5000/gallery |
| Server logs | http://localhost:5000/logs |
| Health check | http://localhost:5000/health |
| Model settings | http://localhost:5000/settings |

To **stop** the app:
```bash
# Press Ctrl+C in the terminal, then run:
docker-compose down
```

> **GPU Support:** The `docker-compose.yml` is configured to use an NVIDIA GPU
> (`tensorflow/serving:latest-gpu`).  If you don't have a GPU, change the image
> to `tensorflow/serving:latest` in `docker-compose.yml`.

---

## 6. Running Just the Website (No Docker)

Use this if you only want to develop or test the Flask front-end without needing
the full TF Serving inference engine.

```bash
# Step 1 — Install Python dependencies
pip install -r requirements.txt

# Step 2 — Start the Flask development server
cd app
python main.py
```

The app will be available at http://localhost:5000.

> ⚠️ **Important:** Uploading an image will fail with a connection error because
> TF Serving is not running.  Everything else (logs, metrics, gallery, health,
> settings pages) will still work fine for development purposes.

---

## 7. Environment Variables

You can customise the app's behaviour by setting these environment variables.
In Docker, set them in the `environment:` section of `docker-compose.yml`.

| Variable | Default | What it does |
|---|---|---|
| `MIN_SCORE_THRESHOLD` | `0.50` | Minimum confidence (0–1) for a detection to be shown. Raise it to reduce false positives. |
| `MAX_CONTENT_LENGTH` | `2147483648` (2 GB) | Maximum allowed file upload size in bytes. |
| `TF_SERVING_URL` | `http://tf-serving:8501/v1/models/object_detection:predict` | The full URL of the TF Serving inference endpoint. |
| `TF_MODEL_STATUS_URL` | `http://tf-serving:8501/v1/models/object_detection` | The URL used to check TF Serving's health status. |
| `MODEL_DIR` | `../model/object_detection` | Path to the folder containing numbered model version directories. |
| `REQUEST_TIMEOUT_SECONDS` | `120` | How long (in seconds) to wait for TF Serving before giving up. |
| `APP_SECRET_KEY` | `development-secret-key` | Flask session cookie signing key. **Change this in production!** |
| `LOG_LEVEL` | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

---

## 8. Page-by-Page Guide

### Home / Upload — `localhost:5000/`

This is the main page you land on when you open the app.

**What you can do here:**
- Click the upload area (or drag and drop) to choose a photo from your computer.
- After selecting a file, a live preview of the image appears inside the upload box.
- Click **"Run Inference"** to send the image to the AI for analysis.
- A progress bar and spinner appear while the AI is processing — this can take
  a few seconds depending on your hardware.

**What appears on the page:**
- **Upload card** — the file picker and submit button.
- **Live System Pulse** — a small sparkline chart showing the response times of
  your last few requests, so you can see at a glance how fast the AI is responding.
- **Supported Objects** — a note about the 90 COCO object categories the model
  can detect, with a link to the full COCO dataset documentation.

> Accepted file types: `.jpg`, `.jpeg`, `.png`

---

###  Results — `localhost:5000/predict`

This page appears automatically after a successful image upload and detection.

**What you see:**
- **Side-by-side image comparison** — your original photo on the left, the same
  photo with red bounding boxes drawn on it on the right.
- **Inference Summary** — tells you the filename and how many objects were detected
  above the confidence threshold.
- **Detection table** — lists every detected object with:
  - Its category label (e.g. "person", "car", "dog")
  - A visual confidence bar showing how certain the model is (0–100%)
  - The raw bounding box coordinates `[ymin, xmin, ymax, xmax]` as fractions of
    the image size (values between 0.0 and 1.0)
- **Probability Distribution chart** — a bar chart (only shown if objects were
  found) showing the confidence score for each detected object side by side.
- **"Analyse Another Image" button** — takes you back to the home page.

> If the model found nothing above your confidence threshold, the table shows
> "No objects were detected with high enough confidence." — try lowering
> `MIN_SCORE_THRESHOLD` in your environment variables.

---

###  Metrics Dashboard — `localhost:5000/metrics`

A full analytics dashboard tracking the performance and reliability of the system
over time.  All data is read from `app/metrics/metrics.json`.

**What you see:**

**KPI tiles (top row):**
| Tile | Meaning |
|---|---|
| Total Requests | How many images have been submitted since the app started |
| Successful | How many returned valid detections |
| Failed | How many errored out (e.g. TF Serving was down) |
| Avg Latency | Average TF Serving response time across all successful requests |
| Last Latency | Response time of the most recent request |
| Total Detections | Grand total of all objects detected across every image |

**Charts (middle section):**
- **Inference Latency Trend** — line chart showing how response time has changed
  across your last 30 requests.  A sudden spike indicates the model struggled
  with a complex image.
- **Density: Targets per Inference** — bar chart showing how many objects were
  found in each of the last 30 requests.
- **Success vs Failure Distribution** — doughnut chart showing the ratio of
  successful to failed requests.
- **Gateway & TF Serving Core Status** — text panel showing the timestamp and
  error message (if any) from the most recent request.

**Raw Engineering Logs table (bottom section):**
A table showing every request in the history buffer (last 30), sorted newest
first, with timestamp, latency, pass/fail badge, detection count, and any
error message.

---

###  Inference Gallery — `localhost:5000/gallery`

A visual archive of every image that has been processed by the AI.

**What you see:**
- A responsive grid of thumbnail images, sorted newest first.
- Each thumbnail is the annotated version of the image (with bounding boxes drawn).
- Clicking any thumbnail opens the full-resolution image in a new browser tab.
- Below each thumbnail, the original filename is shown (the `processed_` prefix
  is stripped for cleaner display).
- A badge in the top-right of the card shows the total number of stored images.

If no images have been processed yet, an empty state message is shown with a
button linking back to the upload page.

> Images are stored in `app/processed/` and served directly by Flask.

---

###  Server Logs — `localhost:5000/logs`

A terminal-style log viewer showing the live output of the Python backend.

**What you see:**
- The last **1,000 lines** of `app/logs/application.log` displayed in a
  dark-themed scrollable panel (styled like a terminal window).
- The log auto-scrolls to the bottom so the most recent entries are always
  visible without manual scrolling.
- A **"Refresh Logs"** button reloads the page to pull the latest entries.

**Log format:**
```
2025-04-21 12:00:00 | INFO | app.routes | Prediction request received.
2025-04-21 12:00:01 | INFO | app.utils | Normalized 3 detections above threshold 0.50
```

Each line contains: timestamp | level | module name | message.

> The log file rotates automatically when it gets too large, so it won't
> fill up your disk over time.

---

###  Health Check — `localhost:5000/health`

Shows whether all parts of the system are running and connected properly.
Useful for diagnosing connection problems between Flask and TF Serving.

**What you see:**
- **System Health Overview banner** — a single large green ("All Systems
  Operational") or red ("System Error Detected") badge giving an instant
  pass/fail summary.
- **Flask API Gateway card** — confirms the web server itself is online and
  shows the protocol it uses (HTTP/WSGI).
- **TensorFlow Inference Core card** — shows the TF Serving model's:
  - Engine state (e.g. `AVAILABLE` = model is loaded and ready)
  - Active model version number (e.g. `v2`)
  - Internal error code (`OK` when everything is fine)

If TF Serving can't be reached, a red error box shows the connection failure
message so you know exactly what went wrong.

> This endpoint also returns HTTP 500 when TF Serving is down, making it
> compatible with Docker health checks and uptime monitoring tools.

---

###  Model Manager / Settings — `localhost:5000/settings`

An admin console for uploading and deploying a new version of the AI model
without restarting the containers.

**What you can do here:**
1. Click the upload area and select a `.zip` file containing your new model.
2. The ZIP must contain a `saved_model.pb` file and a `variables/` folder.
3. Click **"Inject Upgrade"** to upload and deploy it.
4. A progress bar appears while the ZIP is being transferred and extracted.
5. On success, a flash message tells you the new version number that was deployed.

**Host Volume Image Registry table:**
Below the upload form is a table listing all model versions currently deployed
on disk.  The highest version number is marked as **PROD - ACTIVE HEAD**
(the version TF Serving is currently using) and all older versions are marked
as **STANDBY REPLICA**.

> **How versioning works:** Each uploaded model is extracted into a new
> numbered folder (e.g. `model/object_detection/3/`).  TF Serving automatically
> detects and loads the highest version number.  Old versions are kept on disk
> as a rollback reference.

> ZIP files can be up to 2 GB.  The upload may take a while on a slow
> connection — do not close the browser tab while the progress bar is animating.

---

###  Error Page

A friendly error page shown whenever something unexpected happens.

- **404** — the URL you tried to visit doesn't exist.
- **500** — an internal server error occurred (check the Logs page for details).

The page shows the error code and a plain English description.
There is no link back to this page — it only appears when something goes wrong.

---

## 9. Uploading a New Model (Hot-Swap)

You can replace the AI model while the app is still running in two ways:

### Option A — Via the Web Interface (easiest)
1. Go to http://localhost:5000/settings
2. Pack your new model into a `.zip` file containing `saved_model.pb` and
   the `variables/` folder.
3. Upload the ZIP using the form.
4. The app extracts it into a new version folder automatically.
5. TF Serving detects the new version and switches to it within ~30 seconds.

### Option B — Via the Filesystem (manual)
1. Create a new numbered folder inside `model/object_detection/` (e.g. `2/`).
2. Copy `saved_model.pb` and `variables/` into that folder.
3. TF Serving will detect and load it automatically.

After deployment, go to http://localhost:5000/health to confirm the new version
number is showing under "TensorFlow Inference Core".

---

## 10. Where Files Are Stored

| Location | What is stored there |
|---|---|
| `app/originals/` | Every image uploaded by the user (unchanged) |
| `app/processed/` | The same images but with bounding boxes drawn on them |
| `app/metrics/metrics.json` | Request statistics (counts, latencies, history) |
| `app/logs/application.log` | All log messages written by the Python backend |
| `model/object_detection/` | TensorFlow model version folders (1/, 2/, 3/, …) |

> All four `app/` directories are created automatically on first startup.
> The `model/` directory must be set up manually before starting (see Section 4).

---

## 11. Troubleshooting

### "Connection refused" or "TF Serving unavailable"
- Make sure both containers are running: `docker ps`
- Check TF Serving logs: `docker logs tf-serving`
- Verify the model folder structure is correct (see Section 4)
- Go to http://localhost:5000/health to see the exact error message

### "Model not loading" after uploading a ZIP
- Make sure the ZIP contains `saved_model.pb` at its root (or inside one
  sub-folder — the app handles that case automatically)
- Check http://localhost:5000/logs for the exact extraction error

### Image upload fails with a 413 error (payload too large)
- The default limit is 2 GB. For larger files, increase `MAX_CONTENT_LENGTH`
  in `docker-compose.yml` and also update the Nginx/proxy config if applicable

### No objects detected in an image
- The confidence threshold may be too high. Lower `MIN_SCORE_THRESHOLD`
  (e.g. change `0.50` to `0.30` in `docker-compose.yml`) and restart
- Make sure your image contains objects from the 90 COCO categories

### GPU not being used
- Change `tensorflow/serving:latest-gpu` to `tensorflow/serving:latest` in
  `docker-compose.yml` if you don't have an NVIDIA GPU installed
- If you have a GPU but it's not being detected, make sure the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  is installed

### Logs file is empty
- The logs file is created when the first log message is written
- Run an inference first, then check http://localhost:5000/logs

---

## 12. Technical Notes

### Detection threshold
The `MIN_SCORE_THRESHOLD` setting (default `0.50`) controls how confident the
model must be before a detection is shown.  `0.50` means "at least 50% sure".
- **Higher value** (e.g. `0.70`) → fewer results, but all high-confidence ones
- **Lower value** (e.g. `0.30`) → more results, but may include false positives

### Model compatibility
This app is designed specifically for **Faster R-CNN** models from the
[TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
trained on the COCO dataset.  The model must return these output keys:
- `detection_boxes` — bounding box coordinates (normalised 0–1 fractions)
- `detection_scores` — confidence scores (0–1)
- `detection_classes` — COCO class IDs (integers 1–90)

If your model returns a different structure, edit the `normalize_detections()`
and `_coerce_prediction_root()` functions in `app/utils.py`.

### COCO object classes
The model can detect **90 different categories** from the COCO dataset including
people, animals, vehicles, furniture, food, electronics, and more.
Full list: https://cocodataset.org/#explore

### Metrics persistence
All request statistics are stored in `app/metrics/metrics.json` as plain JSON.
The history buffer keeps the last **30 requests** to prevent the file growing
too large.  The file is never deleted automatically — delete or reset it
manually if you want to start fresh.

### Bounding box coordinate format
Bounding box coordinates are returned as `[ymin, xmin, ymax, xmax]` where each
value is a fraction of the image dimensions (between 0.0 and 1.0).  For example,
`[0.1, 0.2, 0.8, 0.9]` on a 500×400 image means:
- Top edge: 40px from the top
- Left edge: 100px from the left
- Bottom edge: 320px from the top
- Right edge: 450px from the left

---

*Built with Python · Flask · TensorFlow Serving · Docker*
