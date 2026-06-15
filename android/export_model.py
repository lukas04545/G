#!/usr/bin/env python3
"""Export YOLOv8 to ONNX (both quality and fast variants) for Android."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export_one(model_name: str, imgsz: int, dest_name: str, assets_dir: Path, opset: int) -> None:
    print(f"\n── Exporting {dest_name}  (imgsz={imgsz}) ──")
    model = YOLO(model_name)
    onnx_path = Path(
        model.export(
            format="onnx",
            simplify=True,
            opset=opset,
            imgsz=imgsz,
            dynamic=False,
        )
    )
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / dest_name
    shutil.move(str(onnx_path), dest)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"   Saved: {dest}  ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export quality (640) and fast (320) YOLO models for Android",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="yolov8n.pt", help="Source .pt weights")
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--assets-dir", default="app/src/main/assets")
    p.add_argument("--quality-only", action="store_true", help="Skip fast (320) export")
    p.add_argument("--fast-only",    action="store_true", help="Skip quality (640) export")
    args = p.parse_args()

    assets_dir = Path(__file__).parent / args.assets_dir

    if not args.fast_only:
        export_one(args.model, 640, "yolov8n.onnx",      assets_dir, args.opset)
    if not args.quality_only:
        export_one(args.model, 320, "yolov8n_fast.onnx", assets_dir, args.opset)

    print("\nDone. Rebuild the APK in Android Studio or with: ./gradlew assembleDebug")


if __name__ == "__main__":
    main()
