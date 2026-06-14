"""Unit tests for detector.py (no real model required)."""
from __future__ import annotations

import numpy as np
import pytest

from detector import Detection, InferenceMetrics


class TestDetection:
    def test_area_normal(self):
        det = Detection(np.array([0, 0, 10, 20]), 0.9, 0, "person")
        assert det.area == 200.0

    def test_area_zero(self):
        det = Detection(np.array([5, 5, 5, 5]), 0.8, 1, "car")
        assert det.area == 0.0

    def test_repr_contains_class_name(self):
        det = Detection(np.array([0, 0, 10, 10]), 0.75, 2, "dog")
        assert "dog" in repr(det)
        assert "0.75" in repr(det)


class TestInferenceMetrics:
    def test_empty_returns_zero(self):
        m = InferenceMetrics()
        assert m.avg_latency_ms == 0.0
        assert m.min_latency_ms == 0.0
        assert m.max_latency_ms == 0.0

    def test_single_sample(self):
        m = InferenceMetrics()
        m.update(0.05)
        assert abs(m.avg_latency_ms - 50.0) < 1e-6
        assert abs(m.min_latency_ms - 50.0) < 1e-6
        assert abs(m.max_latency_ms - 50.0) < 1e-6

    def test_multiple_samples(self):
        m = InferenceMetrics()
        m.update(0.1)
        m.update(0.2)
        assert abs(m.avg_latency_ms - 150.0) < 1e-6
        assert abs(m.min_latency_ms - 100.0) < 1e-6
        assert abs(m.max_latency_ms - 200.0) < 1e-6

    def test_window_eviction(self):
        m = InferenceMetrics(window=3)
        for i in range(6):
            m.update(0.01 * (i + 1))
        assert len(m.inference_times) == 3
        # Last three samples: 40ms, 50ms, 60ms
        assert abs(m.min_latency_ms - 40.0) < 1e-6
        assert abs(m.max_latency_ms - 60.0) < 1e-6
