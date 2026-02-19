#!/usr/bin/env python3
"""
Feature matching + RANSAC quick test.

Usage:
  python3 feature_match_ransac.py \
    --query /path/to/query.png \
    --cand /path/to/candidate.png \
    --out /path/to/output_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np


def _load_image(path: Path, max_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"image not found or unreadable: {path}")

    if max_size:
        h, w = img.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def _create_detector(method: str):
    method = method.lower()
    if method == "orb":
        return cv2.ORB_create(nfeatures=5000), "hamming"
    if method == "akaze":
        return cv2.AKAZE_create(), "hamming"
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT not available in this OpenCV build.")
        return cv2.SIFT_create(), "l2"
    raise ValueError(f"unknown method: {method}")


def _match(detector, matcher_type: str, g1, g2, ratio: float):
    k1, d1 = detector.detectAndCompute(g1, None)
    k2, d2 = detector.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
        return k1, k2, [], []

    if matcher_type == "hamming":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn = matcher.knnMatch(d1, d2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    return k1, k2, d1, d2, good


def _ransac(k1, k2, matches, reproj_th: float):
    if len(matches) < 4:
        return None, None

    src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_th)
    if mask is None:
        return None, None
    return H, mask.ravel().tolist()


def _draw_matches(img1, k1, img2, k2, matches, mask, out_path: Path, title: str):
    if mask is None:
        draw = cv2.drawMatches(img1, k1, img2, k2, matches, None)
    else:
        inliers = [m for m, ok in zip(matches, mask) if ok]
        draw = cv2.drawMatches(img1, k1, img2, k2, inliers, None,
                               matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))

    cv2.imwrite(str(out_path), draw)
    print(f"{title} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="query image path")
    ap.add_argument("--cand", required=True, help="candidate image path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--method", default="orb", choices=["orb", "akaze", "sift"])
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test")
    ap.add_argument("--max-size", type=int, default=1600, help="resize longest side (0 disables)")
    ap.add_argument("--ransac", type=float, default=3.0, help="RANSAC reprojection threshold")
    ap.add_argument("--open", action="store_true", help="open result images after run (macOS)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_size = args.max_size if args.max_size and args.max_size > 0 else None

    img1, g1 = _load_image(Path(args.query), max_size=max_size)
    img2, g2 = _load_image(Path(args.cand), max_size=max_size)

    detector, matcher_type = _create_detector(args.method)
    k1, k2, d1, d2, good = _match(detector, matcher_type, g1, g2, args.ratio)

    print(f"keypoints: query={len(k1)} cand={len(k2)}")
    print(f"good matches (ratio<{args.ratio}): {len(good)}")

    H, mask = _ransac(k1, k2, good, args.ransac)
    if mask is None:
        inliers = 0
        inlier_ratio = 0.0
    else:
        inliers = int(sum(mask))
        inlier_ratio = inliers / max(1, len(good))

    print(f"ransac inliers: {inliers} / {len(good)} (ratio={inlier_ratio:.3f})")
    print(f"homography found: {H is not None}")

    all_path = out_dir / "matches_all.png"
    inlier_path = out_dir / "matches_inliers.png"
    _draw_matches(img1, k1, img2, k2, good, None, all_path, "all matches")
    _draw_matches(img1, k1, img2, k2, good, mask, inlier_path, "inlier matches")

    if args.open:
        # Open inliers first (cleaner view), then full matches
        subprocess.run(["open", str(inlier_path)], check=False)
        subprocess.run(["open", str(all_path)], check=False)


if __name__ == "__main__":
    main()
