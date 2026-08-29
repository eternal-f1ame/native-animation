"""Anchored inference: multi-keyframe generation with per-step clamping.

The stock pipeline re-clamps only frame 0 after every scheduler step
(pipelines/wan_video.py:335). ``anchored_generate`` generalizes: every anchor
slot is re-clamped each step (via a scheduler.step wrapper) and the anchored
model function runs every anchor frame at t=0. Anchors beyond frame 0 are
encoded with the same VAE path the pipeline's own unit uses
(pipelines/wan_video.py:523).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from native_animation.modeling.anchored_model_fn import make_anchored_model_fn


def clamping_step(scheduler_step, anchor_latents: dict[int, torch.Tensor]):
    """Wrap ``scheduler.step`` so anchor slots are re-clamped after each step."""
    def wrapped(model_output, timestep, sample, **kwargs):
        out = scheduler_step(model_output, timestep, sample, **kwargs)
        for idx, latent in anchor_latents.items():
            out[:, :, idx:idx + 1] = latent.to(dtype=out.dtype, device=out.device)
        return out
    return wrapped


def encode_anchor_latents(pipe, anchor_images: dict, height: int, width: int) -> dict[int, torch.Tensor]:
    """VAE-encode each anchor image exactly like the pipeline's first-frame unit."""
    latents = {}
    for idx, image in anchor_images.items():
        pixel = pipe.preprocess_image(image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([pixel.to(dtype=pipe.torch_dtype, device=pipe.device)],
                            device=pipe.device)
        latents[idx] = z.to(dtype=pipe.torch_dtype, device=pipe.device)
    return latents


@contextmanager
def _installed(pipe, anchors: list[int], anchor_latents: dict[int, torch.Tensor]):
    original_model_fn = pipe.model_fn
    original_step = pipe.scheduler.step
    anchors_ref = {"anchors": anchors}
    pipe.model_fn = make_anchored_model_fn(anchors_ref)
    pipe.scheduler.step = clamping_step(original_step, anchor_latents)
    try:
        yield
    finally:
        pipe.model_fn = original_model_fn
        pipe.scheduler.step = original_step


def anchored_generate(pipe, prompt: str, anchor_images: dict, height: int, width: int,
                      num_frames: int, **gen_kwargs):
    """Generate with an arbitrary anchor set {latent_frame_idx: PIL.Image}.

    Frame 0 must be among the anchors (it also rides the stock ``input_image``
    path so the pipeline's own conditioning unit engages).
    """
    if 0 not in anchor_images:
        raise ValueError("anchor_images must include frame 0 (the keyframe)")
    anchors = sorted(anchor_images)
    anchor_latents = encode_anchor_latents(pipe, anchor_images, height, width)
    with _installed(pipe, anchors, anchor_latents):
        return pipe(
            prompt=prompt,
            input_image=anchor_images[0],
            height=height,
            width=width,
            num_frames=num_frames,
            **gen_kwargs,
        )
