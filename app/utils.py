"""Shared utilities: logging setup, FPS counter, HUD overlay, frame helpers."""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a timestamped handler on stderr."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers.clear()
    root.addHandler(handler)


class FPSCounter:
    """Rolling-window frames-per-second counter."""

    def __init__(self, window: int = 30) -> None:
        self._ts: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._ts.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0


def draw_overlay(
    frame: np.ndarray,
    *,
    fps: float,
    latency_ms: float,
    detection_count: int,
    device: str,
    show: bool = True,
) -> np.ndarray:
    """Render a semi-transparent HUD panel with runtime metrics."""
    if not show:
        return frame

    lines = [
        f"FPS:     {fps:6.1f}",
        f"Latency: {latency_ms:5.1f} ms",
        f"Objects: {detection_count:3d}",
        f"Device:  {device.upper()}",
    ]

    pad = 8
    lh = 22
    panel_w = 200
    panel_h = len(lines) * lh + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = pad + (i + 1) * lh - 4
        cv2.putText(
            frame, line, (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return frame


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """Resize while preserving aspect ratio if only one dimension is given."""
    if width is None and height is None:
        return frame
    h, w = frame.shape[:2]
    if width and not height:
        height = int(h * width / w)
    elif height and not width:
        width = int(w * height / h)
    return cv2.resize(frame, (width, height))
