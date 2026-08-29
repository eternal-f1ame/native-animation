"""Anchored Wan model function: run every anchor frame at timestep 0.

A targeted fork of ``diffsynth.pipelines.wan_video.model_fn_wan_video``
(vendored, baseline DiT path only) whose ONE substantive change is the
per-token timestep construction: upstream zeros only frame 0's row
(``pipelines/wan_video.py:1376``); this fork zeros every row in ``anchors``
via ``build_separated_timestep``. Parity with upstream for ``anchors=[0]``
is enforced by ``tests/test_anchored_model_fn.py``.

Dropped upstream branches (use upstream if you need them): sliding-window
tiling, LongCat, S2V/audio, VACE, VAP, animate adapter, WanToDance, TeaCache,
motion controller, reference latents, unified sequence parallel.
"""
from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from diffsynth.core import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d

from native_animation.modeling.anchoring import build_separated_timestep


def model_fn_wan_video_anchored(
    dit: WanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    anchors: list[int] | None = None,
    **kwargs,
) -> torch.Tensor:
    anchors = anchors if anchors is not None else [0]

    # Timestep — the one substantive change vs upstream.
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = build_separated_timestep(
            timestep.flatten()[0],
            latents.shape[2], latents.shape[3], latents.shape[4],
            anchors=anchors,
        ).to(dtype=latents.dtype, device=latents.device)
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image embedding (I2V variants that concat VAE/CLIP conditions)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    x = dit.patchify(x)

    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x, context, t_mod, freqs,
        )

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


def make_anchored_model_fn(anchors_ref: dict):
    """Return a ``pipe.model_fn`` drop-in that reads the live anchor set.

    ``anchors_ref`` is a mutable dict (``{"anchors": [...]}``) owned by the
    training module / inference wrapper, updated per sample or per call.
    """
    def _model_fn(**call_kwargs):
        call_kwargs.pop("anchors", None)
        return model_fn_wan_video_anchored(anchors=anchors_ref.get("anchors", [0]), **call_kwargs)
    return _model_fn
