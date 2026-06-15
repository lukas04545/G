#!/usr/bin/env python3
"""Entry point for the YOLO object detection application."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import cv2

from config import AppConfig, DetectorConfig, ModelSize, VideoConfig
from detector import YOLODetector
from utils import FPSCounter, draw_overlay, setup_logging
from video import SourceType, VideoSource, VideoWriter
from stream import bus, serve_in_background

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yolo-detect",
        description="Real-time YOLO object detection for webcam, video, and images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inp = p.add_argument_group("Input")
    inp.add_argument(
        "--source", default="0",
        help="Webcam index (0/1/…), video file path, or image path",
    )

    mdl = p.add_argument_group("Model")
    mdl.add_argument(
        "--model", default=ModelSize.NANO.value,
        choices=[m.value for m in ModelSize],
        help="YOLO model variant",
    )
    mdl.add_argument(
        "--confidence", type=float, default=0.5, metavar="FLOAT",
        help="Detection confidence threshold (0–1)",
    )
    mdl.add_argument(
        "--iou", type=float, default=0.45, metavar="FLOAT",
        help="NMS IoU threshold (0–1)",
    )
    mdl.add_argument(
        "--device", default="auto",
        help="Inference device: auto | cpu | cuda | cuda:0 | mps",
    )
    mdl.add_argument(
        "--max-det", type=int, default=300, metavar="N",
        help="Maximum detections per frame",
    )

    out = p.add_argument_group("Output")
    out.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save annotated output to this video/image path",
    )
    out.add_argument("--no-display", action="store_true", help="Suppress preview window")
    out.add_argument("--no-fps", action="store_true", help="Hide FPS/latency HUD")
    out.add_argument("--no-labels", action="store_true", help="Hide class labels")
    out.add_argument("--no-confidence", action="store_true", help="Hide confidence scores")

    web = p.add_argument_group("Web streaming")
    web.add_argument(
        "--web", action="store_true",
        help="Serve MJPEG stream at http://<ip>:<port>/ (view on phone/browser)",
    )
    web.add_argument("--web-port", type=int, default=8080, metavar="PORT", help="HTTP port")
    web.add_argument("--web-host", default="0.0.0.0", metavar="HOST", help="Bind address")
    web.add_argument(
        "--jpeg-quality", type=int, default=80, metavar="1-100",
        help="JPEG compression quality for the web stream",
    )

    misc = p.add_argument_group("Misc")
    misc.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    misc.add_argument(
        "--warmup", default=True, action=argparse.BooleanOptionalAction,
        help="Run a warmup pass before processing",
    )

    return p


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0.0 < args.confidence <= 1.0:
        parser.error(f"--confidence must be in (0, 1], got {args.confidence}")
    if not 0.0 < args.iou <= 1.0:
        parser.error(f"--iou must be in (0, 1], got {args.iou}")


def _build_config(args: argparse.Namespace) -> AppConfig:
    # In web-only mode suppress the local OpenCV window automatically
    display = not args.no_display and not (args.web and not hasattr(args, "_force_display"))
    return AppConfig(
        detector=DetectorConfig(
            model_size=ModelSize(args.model),
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            device=args.device,
            max_detections=args.max_det,
        ),
        video=VideoConfig(
            source=args.source,
            output_path=args.output,
            display=display,
        ),
        log_level=args.log_level,
        show_fps=not args.no_fps,
        show_labels=not args.no_labels,
        show_confidence=not args.no_confidence,
    )


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

def run(config: AppConfig, web: bool = False, web_host: str = "0.0.0.0",
        web_port: int = 8080, jpeg_quality: int = 80) -> int:
    """Main processing loop. Returns POSIX exit code."""
    # --- Detector ---
    try:
        detector = YOLODetector(config.detector)
    except Exception:
        logger.exception("Detector initialisation failed")
        return 1

    # --- Source ---
    try:
        source = VideoSource(config.video)
    except Exception:
        logger.exception("Cannot open source: %s", config.video.source)
        return 1

    # --- Output writer ---
    writer: Optional[VideoWriter] = None
    if config.video.output_path:
        try:
            writer = VideoWriter(config.video.output_path, source.fps, source.frame_size)
        except Exception:
            logger.exception("Cannot open output writer")
            source.release()
            return 1

    # --- Web streaming server ---
    if web:
        serve_in_background(host=web_host, port=web_port, jpeg_quality=jpeg_quality)

    # --- Warmup ---
    # (done after source is open so we know actual frame size)
    if hasattr(config, "warmup") and config.warmup:
        detector.warmup()

    fps_counter = FPSCounter(window=30)
    frame_count = 0
    window = "YOLO Object Detection  [q / ESC to quit]"

    if config.video.display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    logger.info("Processing %s: %s", source.source_type.name, config.video.source)

    try:
        for frame in source.frames():
            frame_count += 1

            detections, latency_s = detector.detect(frame)

            annotated = detector.draw_detections(
                frame, detections,
                show_labels=config.show_labels,
                show_confidence=config.show_confidence,
            )

            fps_counter.tick()
            annotated = draw_overlay(
                annotated,
                fps=fps_counter.fps,
                latency_ms=latency_s * 1000,
                detection_count=len(detections),
                device=config.detector.device,
                show=config.show_fps,
            )

            if writer:
                writer.write(annotated)

            if web:
                bus.publish(annotated, quality=jpeg_quality)

            if config.video.display:
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Quit requested")
                    break

            # Images: wait for a keypress then exit
            if source.source_type is SourceType.IMAGE:
                if config.video.display:
                    logger.info("Press any key to close…")
                    cv2.waitKey(0)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        source.release()
        if writer:
            writer.release()
        if config.video.display:
            cv2.destroyAllWindows()

    logger.info(
        "Finished — %d frames | avg FPS %.1f | avg latency %.1f ms",
        frame_count,
        fps_counter.fps,
        detector.metrics.avg_latency_ms,
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    _validate(args, parser)
    config = _build_config(args)
    sys.exit(run(
        config,
        web=args.web,
        web_host=args.web_host,
        web_port=args.web_port,
        jpeg_quality=args.jpeg_quality,
    ))


if __name__ == "__main__":
    main()
