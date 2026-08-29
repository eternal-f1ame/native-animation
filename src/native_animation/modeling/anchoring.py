"""Anchoring: the animator's contract — given drawings are inviolable.

Training-side: sample an anchor set per clip, clamp anchor latents clean, and
exclude anchors from supervision. Model-side: TI2V's separated-timestep DiT
runs conditioned frames at t=0; ``build_separated_timestep`` generalizes the
upstream frame-0-only construction (pipelines/wan_video.py:1376) to any
anchor set, byte-compatible with upstream for anchors=[0].
"""
from __future__ import annotations

import random

import torch

DEFAULT_MODE_PROBS = {"keyframe": 0.50, "first_last": 0.25, "storyboard": 0.15, "none": 0.10}


def sample_anchor_set(num_latent_frames: int, rng: random.Random,
                      probs: dict | None = None) -> list[int]:
    """Draw an anchor mode and materialize its frame indices (sorted)."""
    probs = probs or DEFAULT_MODE_PROBS
    modes, weights = zip(*probs.items())
    mode = rng.choices(modes, weights=weights, k=1)[0]
    last = num_latent_frames - 1
    if mode == "none":
        return []
    if mode == "keyframe":
        return [0]
    if mode == "first_last":
        return [0, last]
    k = rng.randint(1, 3)
    interior = rng.sample(range(1, last), k=min(k, max(last - 1, 0)))
    return sorted([0] + interior)


def anchor_frame_mask(num_latent_frames: int, anchors: list[int]) -> torch.Tensor:
    mask = torch.zeros(num_latent_frames, dtype=torch.bool)
    if anchors:
        mask[torch.tensor(anchors, dtype=torch.long)] = True
    return mask


def apply_anchor_clamp(noisy: torch.Tensor, clean: torch.Tensor,
                       anchors: list[int]) -> torch.Tensor:
    """Return noisy latents with anchor slots replaced by clean latents."""
    out = noisy.clone()
    if anchors:
        idx = torch.tensor(anchors, dtype=torch.long, device=noisy.device)
        out[:, :, idx] = clean[:, :, idx].to(out.dtype)
    return out


def build_separated_timestep(t: torch.Tensor, num_latent_frames: int,
                             lat_h: int, lat_w: int, anchors: list[int]) -> torch.Tensor:
    """Flattened per-token timestep: zeros at anchor rows, ``t`` elsewhere.

    Mirrors the upstream construction exactly for anchors=[0] (parity-tested).
    """
    tokens_per_frame = lat_h * lat_w // 4
    dtype = t.dtype if torch.is_floating_point(t) else torch.float32
    per_frame = torch.ones(num_latent_frames, tokens_per_frame,
                           dtype=dtype, device=t.device) * t
    if anchors:
        per_frame[torch.tensor(anchors, dtype=torch.long, device=t.device)] = 0.0
    return per_frame.flatten()
