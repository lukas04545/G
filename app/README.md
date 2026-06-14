# YOLO Object Detection

Real-time object detection using [Ultralytics YOLOv8](https://docs.ultralytics.com/).
Supports webcam streams, video files, and still images with GPU acceleration and annotated output saving.

---

## Requirements

- Python 3.12+
- CUDA-capable GPU (optional — automatic CPU fallback)

---

## Installation

```bash
# 1. Clone / enter the project
cd app/

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU users) install the CUDA-enabled PyTorch build first:
#    See https://pytorch.org/get-started/locally/
```

---

## Quick Start

```bash
# Webcam (default)
python main.py

# Video file
python main.py --source /path/to/video.mp4

# Still image
python main.py --source /path/to/photo.jpg

# Save annotated output
python main.py --source video.mp4 --output detected.mp4

# Use a larger model with higher confidence threshold
python main.py --model yolov8m --confidence 0.6

# Force CPU inference
python main.py --device cpu
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index, video path, or image path |
| `--model` | `yolov8n` | Model size: `yolov8n/s/m/l/x` |
| `--confidence` | `0.5` | Confidence threshold (0–1) |
| `--iou` | `0.45` | NMS IoU threshold (0–1) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `--max-det` | `300` | Max detections per frame |
| `--output` | — | Save annotated video/image to this path |
| `--no-display` | — | Disable live preview window |
| `--no-fps` | — | Hide FPS/latency HUD |
| `--no-labels` | — | Hide class name labels |
| `--no-confidence` | — | Hide confidence scores |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--no-warmup` | — | Skip model warmup pass |

---

## Architecture

```
app/
├── main.py       CLI entry point and main processing loop
├── detector.py   YOLODetector, Detection, InferenceMetrics
├── video.py      VideoSource (webcam/video/image), VideoWriter
├── config.py     Dataclass configuration (DetectorConfig, VideoConfig, AppConfig)
├── utils.py      FPSCounter, HUD overlay, logging setup, resize helper
└── requirements.txt

tests/
├── conftest.py   sys.path fixture
├── test_config.py
├── test_detector.py
├── test_utils.py
└── test_video.py
```

### Key design decisions

- **`SourceType` enum + `detect_source_type()`** — a single function determines whether the
  input is a webcam, video file, or still image, keeping `VideoSource` free of ad-hoc conditionals.
- **`InferenceMetrics`** — rolling-window latency stats decoupled from the detector loop,
  making it easy to query `avg / min / max` at any time.
- **`FPSCounter`** — uses `time.perf_counter()` and a fixed-size deque instead of global
  counters; thread-safe for read operations.
- **`draw_overlay()`** — alpha-composited panel written directly onto the frame; avoids
  allocating a second full-frame buffer by using `cv2.addWeighted`.

---

## Running Tests

```bash
# From the project root
pip install pytest
pytest tests/ -v
```

---

## Keyboard Shortcuts (display window)

| Key | Action |
|---|---|
| `q` or `ESC` | Quit |
| Any key | Close image viewer |
