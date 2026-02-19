#!/usr/bin/env python3
"""
Generate rotated copies for a single image or all images in a folder.
Outputs are saved as PNG.

Example (single image, default 10..270 step 10):
  python3 rotate_dataset.py \
    --input ../data/db_images/dataset60/dt_028.png \
    --out ../data/db_images/dataset60_rot_10_270

Example (folder with custom angles):
  python3 rotate_dataset.py \
    --input ../data/db_images/dataset60 \
    --out ../data/db_images/dataset60_rot \
    --angles 90,180,270
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_angles(raw: str) -> list[int]:
    angles = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        angles.append(int(part))
    if not angles:
        raise ValueError("No angles provided.")
    return angles


def parse_range(raw: str) -> list[int]:
    # format: start:end:step (inclusive)
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError("Range must be in start:end:step format (e.g., 10:270:10).")
    start, end, step = (int(p) for p in parts)
    if step <= 0:
        raise ValueError("step must be > 0")
    if end < start:
        raise ValueError("end must be >= start")
    return list(range(start, end + 1, step))


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
    raise RuntimeError(f"Input path not found: {input_path}")
    return angles


def rotate_one(src: Path, dst_dir: Path, angles: list[int], expand: bool) -> None:
    img = Image.open(src)
    stem = src.stem

    for a in angles:
        out_name = f"{stem}_rot{a}.png"
        out_path = dst_dir / out_name
        if out_path.exists():
            continue
        rot = img.rotate(a, expand=expand)
        rot.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input image file or directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--angles", default="", help="comma-separated angles (overrides --range)")
    ap.add_argument("--range", default="10:270:10", help="angle range start:end:step (inclusive)")
    ap.add_argument("--no-expand", action="store_true", help="do not expand canvas when rotating")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    angles = parse_angles(args.angles) if args.angles else parse_range(args.range)
    expand = not args.no_expand

    files = list_images(input_path)
    if not files:
        raise RuntimeError(f"No images found in: {input_path}")

    for p in files:
        rotate_one(p, out_dir, angles, expand)

    print(f"done. images={len(files)} angles={angles} out={out_dir}")


if __name__ == "__main__":
    main()
