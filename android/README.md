# YOLO Detect — Android App

Full on-device YOLOv8 object detection. No server, no network. Camera → ONNX Runtime → bounding boxes.

---

## Quick start (5 steps)

### 1 — Export the model (on your desktop)

```bash
cd android/
pip install ultralytics
python export_model.py --model yolov8n.pt
```

This downloads `yolov8n.pt`, converts it to ONNX, and copies it to
`app/src/main/assets/yolov8n.onnx` automatically.

For a smaller / faster model try `--model yolov8n.pt` (default, ~6 MB).
For better accuracy try `--model yolov8s.pt` (~22 MB) or `yolov8m.pt` (~50 MB).

### 2 — Open in Android Studio

File → Open → select the `android/` folder.

### 3 — Let Gradle sync

Android Studio will download all dependencies (~150 MB first time).

### 4 — Build & install

Connect your phone (USB debugging on) and press **Run ▶** or:

```bash
./gradlew installDebug
```

### 5 — Grant camera permission

The app asks for camera permission on first launch. Accept it.

---

## Architecture

```
MainActivity.kt
    │  CameraX ImageAnalysis (RGBA_8888, 640×640, latest-only strategy)
    │  → Bitmap (one per frame, copied from buffer)
    ▼
YoloDetector.kt
    │  preprocess: resize → normalise → CHW FloatBuffer
    │  → ONNX Runtime session.run()
    │  postprocess: parse [1,84,8400] output, threshold, NMS
    ▼
BoundingBoxView.kt
    │  receives List<Detection> with RectF in camera-frame coords
    │  maps to view coords and draws coloured boxes + labels
    ▼
MainActivity.kt  (Dispatchers.Main)
    └─ update statsText: FPS / latency / object count
```

## Performance (yolov8n.onnx, Snapdragon 888)

| Metric | Value |
|---|---|
| Latency | ~40–80 ms/frame |
| FPS | ~15–25 |
| Model size | ~6 MB |
| RAM | ~250 MB |

---

## Changing the model

1. Run `python export_model.py --model yolov8s.pt` (or any size)
2. Edit `YoloDetector.kt` line — change `modelFileName = "yolov8n.onnx"` to match
3. Rebuild

---

## Requirements

- Android 7.0+ (API 24)
- ~50 MB storage
- Any rear-facing camera
