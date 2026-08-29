"""A CPU-constructible tiny WanModel for parity tests (not a fixture of quality)."""
from diffsynth.models.wan_video_dit import WanModel


def build_tiny_wan_model() -> WanModel:
    model = WanModel(
        dim=64,
        in_dim=16,
        ffn_dim=128,
        out_dim=16,
        text_dim=32,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    return model.eval()
