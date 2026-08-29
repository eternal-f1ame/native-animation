"""Native Animation v2 training module.

Reuses the DiffSynth Wan pipeline for encoding (text/VAE units) but replaces
the entire noising/timestep/loss path: per-sample sigma from the timestep
density, anchor-set sampling with clean clamping, the anchored model function
(anchor frames run at t=0), and the v2 objective. The v1 module remains
untouched as the ablation baseline.
"""
from __future__ import annotations

import random

import torch

from native_animation.modeling.anchored_model_fn import make_anchored_model_fn
from native_animation.modeling.anchoring import apply_anchor_clamp, sample_anchor_set
from native_animation.modeling.objectives import native_animation_v2_loss
from native_animation.modeling.timesteps import TimestepDensity
from native_animation.training.train import NativeAnimationWanTrainingModule

NUM_TRAIN_TIMESTEPS = 1000


def compute_v2_loss(model_fn, input_latents: torch.Tensor, density: TimestepDensity,
                    rng: random.Random, cfg: dict) -> dict:
    """Pure v2 loss orchestration (testable with a stub model_fn).

    ``model_fn`` contract: called with kwargs ``latents_in`` (anchor-clamped
    noisy latents), ``timestep`` (sigma * 1000, scalar tensor), ``anchors``
    (list[int]), and ``v_target`` (the velocity target, for stubs); returns
    the velocity prediction with ``input_latents``' shape.
    """
    num_frames = input_latents.shape[2]
    anchors = sample_anchor_set(num_frames, rng, cfg.get("anchor_probs"))

    sigma = density.sample(1).to(device=input_latents.device, dtype=torch.float32)
    s = sigma.reshape(1, 1, 1, 1, 1).to(input_latents.dtype)
    noise = torch.randn_like(input_latents)
    noisy = (1 - s) * input_latents + s * noise
    clamped = apply_anchor_clamp(noisy, input_latents, anchors)
    v_target = noise - input_latents
    timestep = (sigma * NUM_TRAIN_TIMESTEPS).reshape(1).to(input_latents.device)

    v_pred = model_fn(latents_in=clamped, timestep=timestep,
                      anchors=anchors, v_target=v_target)

    losses = native_animation_v2_loss(
        v_pred, v_target, input_latents, anchors,
        alpha=cfg.get("alpha", 1.0),
        lambda0=cfg.get("lambda0", 0.25),
        delta_mode=cfg.get("delta_mode", "vspace"),
        sigma=sigma,
    )
    return {**losses, "anchors": anchors, "sigma": sigma}


class NativeAnimationV2Module(NativeAnimationWanTrainingModule):
    """DiffSynth-hosted module wired to the v2 objective and anchored model."""

    def __init__(self, *args, v2_cfg: dict | None = None, density_cfg: dict | None = None,
                 text_dropout: float = 0.1, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.v2_cfg = v2_cfg or {}
        self.density = TimestepDensity(**(density_cfg or {}))
        self.text_dropout = text_dropout
        self.rng = random.Random(seed)
        self.anchors_ref = {"anchors": [0]}
        # Swap in the anchored model function; the ref is updated per sample.
        self.pipe.model_fn = make_anchored_model_fn(self.anchors_ref)

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        if self.rng.random() < self.text_dropout:
            inputs_posi = {"prompt": ""}
        return inputs_shared, inputs_posi, inputs_nega

    def compute_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        input_latents = inputs_shared["input_latents"]
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        def real_model_fn(*, latents_in, timestep, anchors, v_target=None):
            self.anchors_ref["anchors"] = anchors
            call_inputs = dict(inputs_shared)
            call_inputs.update(inputs_posi)
            call_inputs["latents"] = latents_in
            # TI2V's separated-timestep path must engage so anchors run at t=0.
            call_inputs["fuse_vae_embedding_in_latents"] = True
            return pipe.model_fn(**models, **call_inputs, timestep=timestep)

        out = compute_v2_loss(
            model_fn=real_model_fn,
            input_latents=input_latents,
            density=self.density,
            rng=self.rng,
            cfg=self.v2_cfg,
        )
        return out["total"]
