"""Anchored inference: per-step re-clamping of all anchor slots."""
import torch

from native_animation.inference.anchored import clamping_step


def test_clamping_step_reclamps_all_anchor_slots():
    def fake_step(model_output, timestep, sample):
        return sample + 1.0                       # drifts every slot

    anchor_latents = {0: torch.zeros(1, 4, 1, 2, 2), 3: torch.full((1, 4, 1, 2, 2), 7.0)}
    wrapped = clamping_step(fake_step, anchor_latents)
    latents = torch.zeros(1, 4, 5, 2, 2)
    out = wrapped(None, None, latents)
    assert torch.all(out[:, :, 0] == 0.0)         # re-clamped
    assert torch.all(out[:, :, 3] == 7.0)
    assert torch.all(out[:, :, 1] == 1.0)         # free slots drift normally


def test_clamping_step_passes_through_kwargs():
    captured = {}

    def fake_step(model_output, timestep, sample, to_final=False):
        captured["to_final"] = to_final
        return sample

    wrapped = clamping_step(fake_step, {})
    wrapped(None, None, torch.zeros(1, 4, 3, 2, 2), to_final=True)
    assert captured["to_final"] is True
