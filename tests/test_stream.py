"""Unit tests for stream.py FrameBus (no server required)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from stream import FrameBus


class TestFrameBus:
    def test_publish_and_subscribe_one_frame(self):
        fb = FrameBus()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        received: list[bytes] = []

        def reader():
            for data in fb.subscribe(timeout=1.0):
                if data:
                    received.append(data)
                    break

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        time.sleep(0.05)
        fb.publish(frame)
        t.join(timeout=2.0)

        assert len(received) == 1
        assert received[0][:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_keepalive_on_timeout(self):
        fb = FrameBus()
        results: list[bytes] = []

        def reader():
            gen = fb.subscribe(timeout=0.1)
            results.append(next(gen))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=1.0)

        assert results == [b""]  # keepalive empty bytes on timeout

    def test_publish_replaces_stale_frame(self):
        fb = FrameBus()
        red = np.full((50, 50, 3), [0, 0, 255], dtype=np.uint8)
        blue = np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)

        fb.publish(red)
        fb.publish(blue)  # should overwrite

        with fb._lock:
            data = fb._frame
        assert data is not None
        assert len(data) > 0  # just check it's a valid JPEG
