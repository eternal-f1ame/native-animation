"""v2 loss orchestration: sigma sampling, anchoring, model call, objective."""
import random

import torch

from native_animation.modeling.timesteps import TimestepDensity
from native_animation.training.module_v2 import compute_v2_loss


def test_compute_v2_loss_wires_the_pieces_together():
    torch.manual_seed(0)
    latents = torch.randn(1, 4, 6, 4, 4)
    err = 0.1 * torch.randn_like(latents)
    calls = {}

    def stub_model_fn(*, latents_in, timestep, anchors, v_target):
        calls["anchors"] = anchors
        calls["timestep"] = float(timestep)
        calls["latents_in"] = latents_in
        return v_target + err

    out = compute_v2_loss(
        model_fn=stub_model_fn,
        input_latents=latents,
        density=TimestepDensity(tail_p=0.0),
        rng=random.Random(0),
        cfg={"alpha": 1.0, "lambda0": 0.25, "anchor_probs": None},
    )
    assert torch.isfinite(out["total"])
    assert calls["anchors"] == out["anchors"]
    assert 0.0 < calls["timestep"] < 1000.0
    # Anchor slots of the model input are exactly the clean latents.
    for a in out["anchors"]:
        assert torch.allclose(calls["latents_in"][:, :, a], latents[:, :, a])


def test_compute_v2_loss_none_mode_leaves_input_noisy():
    torch.manual_seed(1)
    latents = torch.randn(1, 4, 6, 4, 4)

    def stub_model_fn(*, latents_in, timestep, anchors, v_target):
        return v_target

    # Force the unconditional mode via probs.
    out = compute_v2_loss(
        model_fn=stub_model_fn,
        input_latents=latents,
        density=TimestepDensity(tail_p=0.0),
        rng=random.Random(0),
        cfg={"alpha": 0.0, "lambda0": 0.0,
             "anchor_probs": {"keyframe": 0, "first_last": 0, "storyboard": 0, "none": 1}},
    )
    assert out["anchors"] == []
    assert float(out["total"]) < 1e-10        # perfect prediction
