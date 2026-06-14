"""Video/image/webcam source and output writer abstractions."""
from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from config import VideoConfig

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}


class SourceType(Enum):
    WEBCAM = auto()
    VIDEO_FILE = auto()
    IMAGE = auto()


def detect_source_type(source: str) -> SourceType:
    """Infer input source type from the *source* string."""
    # Pure integer → webcam index
    try:
        int(source)
        return SourceType.WEBCAM
    except ValueError:
        pass

    # Linux video device
    if source.startswith("/dev/video"):
        return SourceType.WEBCAM

    suffix = Path(source).suffix.lower()
    if suffix in _IMAGE_EXTS:
        return SourceType.IMAGE
    if suffix in _VIDEO_EXTS:
        return SourceType.VIDEO_FILE

    raise ValueError(
        f"Cannot determine source type for {source!r}. "
        "Provide a webcam index, image path, or video path."
    )


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter with context-manager support."""

    def __init__(self, path: str, fps: float, frame_size: Tuple[int, int]) -> None:
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {path!r}")
        logger.info(
            "Video writer opened: %s  %dx%d @ %.1f fps",
            path, frame_size[0], frame_size[1], fps,
        )

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
        logger.info("Saved annotated video: %s", self.path)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()


class VideoSource:
    """Unified reader for webcam, video file, and still image inputs."""

    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self.source_type = detect_source_type(config.source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._image: Optional[np.ndarray] = None
        self._open()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self.source_type is SourceType.IMAGE:
            self._image = cv2.imread(self.config.source)
            if self._image is None:
                raise FileNotFoundError(f"Cannot read image: {self.config.source!r}")
            h, w = self._image.shape[:2]
            logger.info("Loaded image: %s  (%dx%d)", self.config.source, w, h)
            return

        raw = self.config.source
        cap_src: int | str = int(raw) if raw.isdigit() else raw
        self._cap = cv2.VideoCapture(cap_src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.source!r}")

        if self.config.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        kind = "webcam" if self.source_type is SourceType.WEBCAM else "video"
        logger.info(
            "Opened %s: %s  %dx%d @ %.1f fps  (%d frames)",
            kind, self.config.source,
            self.frame_width, self.frame_height,
            self.fps, self.total_frames,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        if self.source_type is SourceType.IMAGE:
            return 30.0
        return (self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0) or 30.0

    @property
    def frame_width(self) -> int:
        if self.source_type is SourceType.IMAGE and self._image is not None:
            return self._image.shape[1]
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def frame_height(self) -> int:
        if self.source_type is SourceType.IMAGE and self._image is not None:
            return self._image.shape[0]
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.frame_width, self.frame_height)

    @property
    def total_frames(self) -> int:
        if self.source_type is SourceType.IMAGE:
            return 1
        if self._cap is None:
            return 0
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else -1  # -1 means unknown (live webcam)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.source_type is SourceType.IMAGE:
            if self._image is not None:
                return True, self._image.copy()
            return False, None
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until exhausted or KeyboardInterrupt."""
        if self.source_type is SourceType.IMAGE:
            if self._image is not None:
                yield self._image.copy()
            return
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_) -> None:
        self.release()
