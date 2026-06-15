#!/usr/bin/env python3
"""Export a YOLOv8 .pt model to ONNX and place it in the Android assets folder."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export YOLOv8 to ONNX for Android deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="yolov8n.pt", help="Source .pt model")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    p.add_argument(
        "--assets-dir",
        default="app/src/main/assets",
        help="Destination directory (relative to android/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.model} …")
    model = YOLO(args.model)

    print("Exporting to ONNX …")
    onnx_path = Path(
        model.export(
            format="onnx",
            simplify=True,
            opset=args.opset,
            imgsz=args.imgsz,
            dynamic=False,
        )
    )

    assets_dir = Path(__file__).parent / args.assets_dir
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / onnx_path.name
    shutil.move(str(onnx_path), dest)

    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"\nModel saved: {dest}")
    print(f"Size:        {size_mb:.1f} MB")
    print(f"\nNext step: open android/ in Android Studio and build the APK.")


if __name__ == "__main__":
    main()
