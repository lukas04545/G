"""Unit tests for config.py."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from config import AppConfig, DetectorConfig, ModelSize, VideoConfig


class TestDetectorConfig:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.confidence_threshold == 0.5
        assert cfg.iou_threshold == 0.45
        assert cfg.max_detections == 300

    def test_auto_device_selects_cpu_when_no_cuda(self):
        import types
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            # Re-evaluate __post_init__ by constructing a fresh instance
            # with the patched torch in scope via importlib reload
            import importlib, config as cfg_mod
            importlib.reload(cfg_mod)
            obj = cfg_mod.DetectorConfig(device="auto")
        assert obj.device == "cpu"

    def test_auto_device_selects_cuda_when_available(self):
        import types
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            import importlib, config as cfg_mod
            importlib.reload(cfg_mod)
            obj = cfg_mod.DetectorConfig(device="auto")
        assert obj.device == "cuda"

    def test_explicit_device_not_overridden(self):
        cfg = DetectorConfig(device="cpu")
        assert cfg.device == "cpu"


class TestModelSize:
    def test_all_values_are_yolov8_variants(self):
        for m in ModelSize:
            assert m.value.startswith("yolov8")

    def test_from_string(self):
        assert ModelSize("yolov8n") is ModelSize.NANO
        assert ModelSize("yolov8x") is ModelSize.XLARGE


class TestVideoConfig:
    def test_defaults(self):
        cfg = VideoConfig()
        assert cfg.source == "0"
        assert cfg.output_path is None
        assert cfg.display is True


class TestAppConfig:
    def test_nested_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.detector, DetectorConfig)
        assert isinstance(cfg.video, VideoConfig)
        assert cfg.log_level == "INFO"
        assert cfg.show_fps is True
        assert cfg.show_labels is True
        assert cfg.show_confidence is True
