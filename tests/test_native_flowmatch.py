"""Invariants of the project-owned scheduler and loss helpers (CPU-only)."""
import torch

from native_animation.modeling.native_flowmatch import (
    NativeAnimationFlowMatchScheduler,
    _motion_frame_weights,
    _weighted_mse,
)


def _wan_sigmas(num_steps: int, shift: float) -> torch.Tensor:
    """Reference Wan schedule: linspace sigmas pushed through the shift map."""
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def test_scheduler_stores_and_applies_project_shift():
    sched = NativeAnimationFlowMatchScheduler(shift=3.0)
    assert sched.shift == 3.0
    sched.set_timesteps(num_inference_steps=10, training=True)
    assert torch.allclose(sched.sigmas, _wan_sigmas(10, 3.0))


def test_scheduler_explicit_shift_overrides_default():
    sched = NativeAnimationFlowMatchScheduler(shift=3.0)
    sched.set_timesteps(num_inference_steps=10, training=True, shift=5.0)
    assert torch.allclose(sched.sigmas, _wan_sigmas(10, 5.0))
    assert not torch.allclose(sched.sigmas, _wan_sigmas(10, 3.0))


def test_motion_weights_shapes_with_and_without_anchor():
    latents = torch.randn(2, 4, 5, 3, 3)  # (B, C, T, H, W)
    with_anchor = _motion_frame_weights(latents, anchor_frames=1, motion_weighting_scale=1.0)
    assert tuple(with_anchor.shape) == (2, 1, 4, 1, 1)  # T-1 weights for supervised frames
    without_anchor = _motion_frame_weights(latents, anchor_frames=0, motion_weighting_scale=1.0)
    assert tuple(without_anchor.shape) == (2, 1, 5, 1, 1)  # padded to T
    assert torch.all(without_anchor[:, :, 0] == 1.0)  # leading pad slot is neutral


def test_motion_weights_range_and_disable():
    latents = torch.randn(2, 4, 5, 3, 3)
    weights = _motion_frame_weights(latents, anchor_frames=1, motion_weighting_scale=1.0)
    # Per-clip normalization puts the most active frame at exactly 1 + scale.
    assert torch.isclose(weights.max(), torch.tensor(2.0))
    assert torch.all(weights >= 1.0)
    assert _motion_frame_weights(latents, anchor_frames=1, motion_weighting_scale=0.0) is None
    single_frame = torch.randn(2, 4, 1, 3, 3)
    assert _motion_frame_weights(single_frame, anchor_frames=0, motion_weighting_scale=1.0) is None


def test_weighted_mse_reduces_to_plain_mse_with_unit_weights():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 4, 5, 3, 3), torch.randn(2, 4, 5, 3, 3)
    unweighted = _weighted_mse(pred, target)
    unit = _weighted_mse(pred, target, torch.ones(2, 1, 5, 1, 1))
    assert torch.isclose(unweighted, unit)
    assert torch.isclose(unweighted, (pred - target).pow(2).mean())


def test_weighted_mse_is_invariant_to_weight_rescaling():
    torch.manual_seed(1)
    pred, target = torch.randn(2, 4, 5, 3, 3), torch.randn(2, 4, 5, 3, 3)
    weights = 1.0 + torch.rand(2, 1, 5, 1, 1)
    assert torch.isclose(
        _weighted_mse(pred, target, weights),
        _weighted_mse(pred, target, 2.0 * weights),
    )
