#!/usr/bin/env python3
"""Run basic reproducible quality and distribution checks for synthetic images."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def image_stats(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "channels": int(image.shape[2]),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "edge_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def summarize(stats):
    frame = pd.DataFrame(stats)
    return {
        "count": int(len(frame)),
        "dimensions": sorted(
            {f"{h}x{w}x{c}" for h, w, c in zip(
                frame["height"], frame["width"], frame["channels"]
            )}
        ),
        "mean_pixel": {
            "mean": float(frame["mean"].mean()),
            "std": float(frame["mean"].std(ddof=0)),
        },
        "pixel_std": {
            "mean": float(frame["std"].mean()),
            "std": float(frame["std"].std(ddof=0)),
        },
        "edge_variance": {
            "mean": float(frame["edge_variance"].mean()),
            "std": float(frame["edge_variance"].std(ddof=0)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", default="data/synthetic_train_only")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--output", default="results/synthetic_quality.json")
    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic_dir)
    real_dir = Path(args.real_dir)
    extensions = {".jpg", ".jpeg", ".png"}
    synthetic_files = sorted(p for p in synthetic_dir.iterdir() if p.suffix.lower() in extensions)
    real_files = sorted(p for p in real_dir.iterdir() if p.suffix.lower() in extensions)
    if not synthetic_files:
        raise SystemExit(f"No synthetic images found in {synthetic_dir}")
    if not real_files:
        raise SystemExit(f"No real images found in {real_dir}")

    synthetic_stats = [image_stats(path) for path in synthetic_files]
    real_stats = [image_stats(path) for path in real_files]
    report = {
        "synthetic_dir": str(synthetic_dir),
        "real_dir": str(real_dir),
        "synthetic": summarize(synthetic_stats),
        "real": summarize(real_stats),
        "checks": {
            "all_synthetic_readable": True,
            "all_synthetic_dimensions_224": all(
                item["height"] == 224 and item["width"] == 224
                for item in synthetic_stats
            ),
            "warning": (
                "These distribution checks do not establish medical realism or "
                "lesion-preserving validity. Dermatologist review or a validated "
                "medical-image quality metric is still required."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nSaved quality report to {output}")


if __name__ == "__main__":
    main()
