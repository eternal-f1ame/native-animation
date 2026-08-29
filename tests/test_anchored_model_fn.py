"""Parity guard: the anchored fork must equal upstream for the frame-0 case."""
import torch

from diffsynth.pipelines.wan_video import model_fn_wan_video

from native_animation.modeling.anchored_model_fn import model_fn_wan_video_anchored

from tiny_wan import build_tiny_wan_model


def _inputs(seed=0):
    torch.manual_seed(seed)
    dit = build_tiny_wan_model()
    latents = torch.randn(1, 16, 5, 4, 4)
    context = torch.randn(1, 7, 32)
    t = torch.tensor([500.0])
    return dit, latents, context, t


def test_fork_matches_upstream_for_frame0_anchor():
    dit, latents, context, t = _inputs()
    with torch.no_grad():
        up = model_fn_wan_video(dit=dit, latents=latents, timestep=t, context=context,
                                fuse_vae_embedding_in_latents=True)
        ours = model_fn_wan_video_anchored(dit=dit, latents=latents, timestep=t, context=context,
                                           fuse_vae_embedding_in_latents=True, anchors=[0])
    assert up.shape == ours.shape
    assert torch.allclose(up, ours, atol=1e-5)


def test_fork_diverges_when_extra_anchors_are_added():
    dit, latents, context, t = _inputs()
    with torch.no_grad():
        a0 = model_fn_wan_video_anchored(dit=dit, latents=latents, timestep=t, context=context,
                                         fuse_vae_embedding_in_latents=True, anchors=[0])
        a03 = model_fn_wan_video_anchored(dit=dit, latents=latents, timestep=t, context=context,
                                          fuse_vae_embedding_in_latents=True, anchors=[0, 3])
    assert not torch.allclose(a0, a03)


def test_fork_no_fuse_matches_upstream_scalar_path():
    dit, latents, context, t = _inputs()
    with torch.no_grad():
        up = model_fn_wan_video(dit=dit, latents=latents, timestep=t, context=context,
                                fuse_vae_embedding_in_latents=False)
        ours = model_fn_wan_video_anchored(dit=dit, latents=latents, timestep=t, context=context,
                                           fuse_vae_embedding_in_latents=False, anchors=[])
    assert torch.allclose(up, ours, atol=1e-5)
