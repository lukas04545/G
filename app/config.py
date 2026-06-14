"""Application configuration dataclasses and enums."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelSize(str, Enum):
    NANO = "yolov8n"
    SMALL = "yolov8s"
    MEDIUM = "yolov8m"
    LARGE = "yolov8l"
    XLARGE = "yolov8x"


@dataclass
class DetectorConfig:
    model_size: ModelSize = ModelSize.NANO
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    max_detections: int = 300

    def __post_init__(self) -> None:
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"


@dataclass
class VideoConfig:
    source: str = "0"
    output_path: Optional[str] = None
    display: bool = True
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    log_level: str = "INFO"
    show_fps: bool = True
    show_labels: bool = True
    show_confidence: bool = True
