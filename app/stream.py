"""MJPEG streaming server — view annotated detection in any browser/phone."""
from __future__ import annotations

import logging
import threading
import time
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame bus — lets the detection loop publish frames to HTTP clients
# ---------------------------------------------------------------------------

class FrameBus:
    """Thread-safe single-slot frame store; readers always get the latest frame."""

    def __init__(self) -> None:
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def publish(self, frame: np.ndarray, quality: int = 80) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self._lock:
            self._frame = buf.tobytes()
        self._event.set()

    def subscribe(self, timeout: float = 5.0) -> Generator[bytes, None, None]:
        """Yield JPEG bytes whenever a new frame arrives."""
        while True:
            if self._event.wait(timeout):
                self._event.clear()
                with self._lock:
                    data = self._frame
                if data is not None:
                    yield data
            else:
                yield b""  # keepalive so the connection doesn't time out


# Global bus shared between the detection loop and the web server
bus = FrameBus()


# ---------------------------------------------------------------------------
# FastAPI MJPEG server
# ---------------------------------------------------------------------------

def _mjpeg_response(quality: int):
    """Generator that produces an MJPEG multipart stream."""
    boundary = b"--frame"
    for jpeg in bus.subscribe():
        if not jpeg:
            continue
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
            b"\r\n" + jpeg + b"\r\n"
        )


def build_app(jpeg_quality: int = 80):
    """Build and return the FastAPI application."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    app = FastAPI(title="YOLO Detection Stream")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _PAGE_HTML

    @app.get("/stream")
    async def stream():
        return StreamingResponse(
            _mjpeg_response(jpeg_quality),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, jpeg_quality: int = 80) -> None:
    """Start the MJPEG server in the current thread (blocking)."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Install uvicorn: pip install uvicorn")

    app = build_app(jpeg_quality)
    logger.info("Streaming at  http://%s:%d/", host, port)
    logger.info("Open on phone: http://<your-machine-ip>:%d/", port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def serve_in_background(host: str = "0.0.0.0", port: int = 8080, jpeg_quality: int = 80) -> threading.Thread:
    """Start the server in a daemon thread; returns immediately."""
    t = threading.Thread(target=serve, args=(host, port, jpeg_quality), daemon=True)
    t.start()
    time.sleep(0.5)  # give uvicorn a moment to bind
    return t


# ---------------------------------------------------------------------------
# Viewer HTML (single-page, mobile-friendly)
# ---------------------------------------------------------------------------

_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Detection</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: sans-serif;
           display: flex; flex-direction: column; align-items: center;
           min-height: 100vh; padding: 12px; }
    h1  { font-size: 1.1rem; margin-bottom: 10px; color: #0f0; letter-spacing: .05em; }
    img { width: 100%; max-width: 960px; border: 2px solid #0f0; border-radius: 4px; }
    footer { margin-top: 10px; font-size: .75rem; color: #666; }
  </style>
</head>
<body>
  <h1>&#x1F4F9; YOLO Object Detection</h1>
  <img src="/stream" alt="Detection stream">
  <footer>MJPEG live stream &mdash; open on any device on the same network</footer>
</body>
</html>
"""
