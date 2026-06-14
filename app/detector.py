"""YOLO-based object detector with annotation and metrics tracking."""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import DetectorConfig

logger = logging.getLogger(__name__)


class Detection:
    """A single detected object."""

    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        self.bbox = bbox  # [x1, y1, x2, y2] in pixel coords
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    @property
    def area(self) -> float:
        w = float(self.bbox[2] - self.bbox[0])
        h = float(self.bbox[3] - self.bbox[1])
        return w * h

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"Detection(class={self.class_name!r}, conf={self.confidence:.2f}, "
            f"bbox=[{x1},{y1},{x2},{y2}])"
        )


class InferenceMetrics:
    """Rolling-window inference latency tracker."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self.inference_times: list[float] = []

    def update(self, elapsed: float) -> None:
        self.inference_times.append(elapsed)
        if len(self.inference_times) > self._window:
            self.inference_times.pop(0)

    def _stat(self, fn, default: float = 0.0) -> float:
        return fn(self.inference_times) * 1000 if self.inference_times else default

    @property
    def avg_latency_ms(self) -> float:
        return self._stat(lambda t: sum(t) / len(t))

    @property
    def min_latency_ms(self) -> float:
        return self._stat(min)

    @property
    def max_latency_ms(self) -> float:
        return self._stat(max)


# BGR color palette — one colour per class (cycles if > len)
_PALETTE: list[Tuple[int, int, int]] = [
    (0, 255, 0),
    (255, 80, 0),
    (0, 80, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 255, 128),
    (255, 0, 128),
]


class YOLODetector:
    """Production YOLO object detector."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.metrics = InferenceMetrics()
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from ultralytics import YOLO  # deferred so tests can mock easily

        model_name = f"{self.config.model_size.value}.pt"
        logger.info("Loading model %s on device=%s", model_name, self.config.device)
        try:
            self._model = YOLO(model_name)
            self._model.to(self.config.device)
            logger.info("Model ready — %d classes", len(self._model.names))
        except Exception:
            logger.exception("Failed to load model %s", model_name)
            raise

    def warmup(self, shape: Tuple[int, int, int] = (640, 640, 3)) -> None:
        logger.info("Running model warmup…")
        dummy = np.zeros(shape, dtype=np.uint8)
        self.detect(dummy)
        logger.info("Warmup complete (%.1f ms)", self.metrics.avg_latency_ms)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Tuple[list[Detection], float]:
        """Run inference. Returns (detections, elapsed_seconds)."""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()
        results = self._model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            device=self.config.device,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        self.metrics.update(elapsed)

        detections = self._parse(results[0])
        return detections, elapsed

    def _parse(self, result) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for i in range(len(result.boxes)):
            bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(result.boxes.conf[i].cpu().numpy())
            cls_id = int(result.boxes.cls[i].cpu().numpy())
            detections.append(Detection(bbox, conf, cls_id, self._model.names[cls_id]))
        return detections

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        *,
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Return a copy of *frame* annotated with bounding boxes and labels."""
        out = frame.copy()
        for det in detections:
            color = _PALETTE[det.class_id % len(_PALETTE)]
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            parts: list[str] = []
            if show_labels:
                parts.append(det.class_name)
            if show_confidence:
                parts.append(f"{det.confidence:.2f}")
            if not parts:
                continue

            label = " ".join(parts)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y1 - 5, th + baseline + 5)
            # filled rectangle behind text
            cv2.rectangle(out, (x1, top - th - baseline), (x1 + tw + 4, top + baseline), color, -1)
            cv2.putText(out, label, (x1 + 2, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names if self._model else {}
