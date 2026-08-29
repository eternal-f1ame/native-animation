"""Minimal curation scoring for pre-curated booru shots (spec §2 Stage 0).

Sakugabooru posts are community-curated craft, so unlike raw-episode pipelines
we only screen for technical unusability: hold-only (static) clips, softness
(blur), and decode failure. Thresholds live in configs/stage0.yaml.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def static_score(frames: list[np.ndarray]) -> float:
    """Mean absolute inter-frame difference; ~0 for hold-only clips."""
    grays = [_gray(f).astype(np.float32) for f in frames]
    if len(grays) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(b - a)) for a, b in zip(grays, grays[1:])]))


def blur_score(frames: list[np.ndarray]) -> float:
    """Median Laplacian variance across frames; low = soft/blurry."""
    return float(np.median([cv2.Laplacian(_gray(f), cv2.CV_64F).var() for f in frames]))


def curation_verdict(static: float, blur: float, duration_s: float, cfg: dict) -> dict:
    reasons = []
    if static < cfg["min_mean_framediff"]:
        reasons.append("static")
    if blur < cfg["min_laplacian_var"]:
        reasons.append("blur")
    if duration_s <= 0:
        reasons.append("decode")
    return {"pass": not reasons, "reasons": reasons}
