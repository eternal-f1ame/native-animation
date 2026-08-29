#!/usr/bin/env python3
"""Smoke gate: does the 16x-spatial Wan VAE preserve line art at 480x832?

Encodes and decodes keyframes drawn from a metadata CSV, reporting per-image
PSNR and Canny-edge IoU plus side-by-side PNGs for eyeball judgment
(spec §4 gate 2). Pass guidance: median PSNR >= 28, median edge IoU >= 0.6 —
but the side-by-sides are the real verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402
from native_animation.data.sampling import (  # noqa: E402
    extract_first_frame,
    read_metadata_rows,
    resolve_video_path,
    select_rows,
)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else 99.0


def edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 100, 200) > 0
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), 100, 200) > 0
    union = np.logical_or(ea, eb).sum()
    return float(np.logical_and(ea, eb).sum() / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                                   origin_file_pattern="Wan2.2_VAE.pth")],
    )

    rows = read_metadata_rows(args.metadata_csv)
    selected = select_rows(rows, limit=args.num_images, unique_series=True)
    results = []
    for row_index, row in selected:
        video_path = resolve_video_path(args.dataset_base_path, row)
        key_path = args.out_dir / f"{row_index:05d}_src.png"
        extract_first_frame(video_path, key_path, resize=(args.width, args.height))
        image = Image.open(key_path).convert("RGB")

        pixel = pipe.preprocess_image(image).transpose(0, 1)
        with torch.no_grad():
            z = pipe.vae.encode([pixel.to(dtype=pipe.torch_dtype, device=pipe.device)],
                                device=pipe.device)
            decoded = pipe.vae.decode(z, device=pipe.device)
        frame = pipe.vae_output_to_video(decoded)[0]
        round_trip = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        source = cv2.imread(str(key_path))
        round_trip = cv2.resize(round_trip, (source.shape[1], source.shape[0]))

        metrics = {"row": row_index, "psnr": psnr(source, round_trip),
                   "edge_iou": edge_iou(source, round_trip)}
        results.append(metrics)
        side = np.concatenate([source, round_trip], axis=1)
        cv2.imwrite(str(args.out_dir / f"{row_index:05d}_side_by_side.png"), side)
        print(metrics, flush=True)

    summary = {"median_psnr": float(np.median([r["psnr"] for r in results])),
               "median_edge_iou": float(np.median([r["edge_iou"] for r in results])),
               "results": results}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("median_psnr", "median_edge_iou")}))


if __name__ == "__main__":
    main()
