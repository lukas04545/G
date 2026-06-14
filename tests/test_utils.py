"""Unit tests for utils.py."""
from __future__ import annotations

import time

import numpy as np
import pytest

from utils import FPSCounter, draw_overlay, resize_frame


class TestFPSCounter:
    def test_empty(self):
        c = FPSCounter()
        assert c.fps == 0.0

    def test_single_tick_no_fps(self):
        c = FPSCounter()
        c.tick()
        assert c.fps == 0.0

    def test_approximate_fps(self):
        c = FPSCounter(window=10)
        interval = 0.02  # 50 fps
        for _ in range(6):
            c.tick()
            time.sleep(interval)
        # Allow ±30 % tolerance for slow CI runners
        assert 35.0 < c.fps < 65.0

    def test_window_limit(self):
        c = FPSCounter(window=5)
        for _ in range(20):
            c.tick()
        assert len(c._ts) <= 5


class TestResizeFrame:
    def _frame(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_no_resize_returns_same(self):
        f = self._frame()
        result = resize_frame(f)
        assert result.shape == f.shape

    def test_resize_by_width_preserves_ratio(self):
        result = resize_frame(self._frame(480, 640), width=320)
        assert result.shape[1] == 320
        assert result.shape[0] == 240

    def test_resize_by_height_preserves_ratio(self):
        result = resize_frame(self._frame(480, 640), height=240)
        assert result.shape[0] == 240
        assert result.shape[1] == 320

    def test_resize_both_dimensions(self):
        result = resize_frame(self._frame(480, 640), width=160, height=120)
        assert result.shape[:2] == (120, 160)


class TestDrawOverlay:
    def _frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_same_shape(self):
        f = self._frame()
        result = draw_overlay(f, fps=30.0, latency_ms=10.0, detection_count=3, device="cpu")
        assert result.shape == f.shape

    def test_show_false_returns_unmodified(self):
        f = self._frame()
        original = f.copy()
        result = draw_overlay(f, fps=30.0, latency_ms=10.0, detection_count=3, device="cpu", show=False)
        np.testing.assert_array_equal(result, original)

    def test_overlay_modifies_pixels(self):
        f = self._frame()  # all zeros
        result = draw_overlay(f, fps=60.0, latency_ms=5.0, detection_count=1, device="cuda")
        # HUD panel spans top-left ~200x100 px; at least some pixel must be non-zero
        assert result[:100, :200].sum() > 0
