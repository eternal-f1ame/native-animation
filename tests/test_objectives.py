"""v2 objective: motion-weighted velocity + sigma-uniform delta consistency."""
import torch

from native_animation.modeling.objectives import native_animation_v2_loss


def _tensors(b=1, c=4, t=6, h=3, w=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(b, c, t, h, w, generator=g)
    v_target = torch.randn(b, c, t, h, w, generator=g)
    return latents, v_target


def test_alpha_zero_no_delta_reduces_to_masked_mse():
    latents, v_target = _tensors()
    err = torch.randn_like(v_target) * 0.1
    out = native_animation_v2_loss(v_target + err, v_target, latents,
                                   anchors=[0], alpha=0.0, lambda0=0.0)
    manual = err[:, :, 1:].pow(2).mean()          # anchors excluded
    assert torch.allclose(out["total"], manual, atol=1e-6)


def test_anchor_frames_do_not_influence_the_loss():
    latents, v_target = _tensors()
    pred = v_target + 0.1
    out_a = native_animation_v2_loss(pred, v_target, latents, anchors=[0, 3])
    corrupted = pred.clone()
    corrupted[:, :, [0, 3]] += 100.0              # garbage at anchors only
    out_b = native_animation_v2_loss(corrupted, v_target, latents, anchors=[0, 3])
    assert torch.allclose(out_a["total"], out_b["total"], atol=1e-5)


def test_delta_term_is_exactly_error_smoothness():
    # v-space residual == Delta(e): constant per-frame error has zero delta
    # penalty; alternating error has a large one (anti-flicker semantics).
    latents, v_target = _tensors()
    constant = native_animation_v2_loss(v_target + 0.5, v_target, latents,
                                        anchors=[], alpha=0.0, lambda0=1.0)
    alternating_err = torch.zeros_like(v_target)
    alternating_err[:, :, ::2] = 0.5
    alternating = native_animation_v2_loss(v_target + alternating_err, v_target,
                                           latents, anchors=[], alpha=0.0, lambda0=1.0)
    assert float(constant["delta"]) < 1e-10
    assert float(alternating["delta"]) > 0.01


def test_delta_uses_clean_values_at_anchor_boundaries():
    # With every non-anchor prediction perfect, delta must vanish even though
    # anchors carry (excluded) garbage predictions.
    latents, v_target = _tensors()
    pred = v_target.clone()
    pred[:, :, 0] += 99.0
    out = native_animation_v2_loss(pred, v_target, latents, anchors=[0], lambda0=1.0)
    assert float(out["delta"]) < 1e-10


def test_motion_weighting_shifts_loss_toward_active_frames():
    latents, v_target = _tensors()
    latents[:, :, 4] += 10.0                      # one high-motion beat
    err = torch.zeros_like(v_target)
    err[:, :, 4] = 0.3                            # error exactly on the beat
    weighted = native_animation_v2_loss(v_target + err, v_target, latents,
                                        anchors=[0], alpha=1.0, lambda0=0.0)
    unweighted = native_animation_v2_loss(v_target + err, v_target, latents,
                                          anchors=[0], alpha=0.0, lambda0=0.0)
    assert float(weighted["velocity"]) > float(unweighted["velocity"])
