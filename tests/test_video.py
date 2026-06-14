"""Unit tests for video.py source-type detection and VideoWriter."""
from __future__ import annotations

import pytest

from video import SourceType, detect_source_type


class TestDetectSourceType:
    @pytest.mark.parametrize("src", ["0", "1", "2"])
    def test_integer_string_is_webcam(self, src):
        assert detect_source_type(src) is SourceType.WEBCAM

    @pytest.mark.parametrize("src", ["/dev/video0", "/dev/video1"])
    def test_linux_device_is_webcam(self, src):
        assert detect_source_type(src) is SourceType.WEBCAM

    @pytest.mark.parametrize("src", [
        "photo.jpg", "image.jpeg", "snapshot.png",
        "bitmap.bmp", "scan.tiff", "modern.webp",
    ])
    def test_image_extensions(self, src):
        assert detect_source_type(src) is SourceType.IMAGE

    @pytest.mark.parametrize("src", [
        "clip.mp4", "recording.avi", "movie.mov",
        "video.mkv", "broadcast.wmv", "stream.flv",
    ])
    def test_video_extensions(self, src):
        assert detect_source_type(src) is SourceType.VIDEO_FILE

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot determine source type"):
            detect_source_type("file.xyz")

    def test_empty_string_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            detect_source_type("")
