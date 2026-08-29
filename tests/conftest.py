"""Test-session setup.

On CPU-only nodes, the vendored Wan DiT's own flash-attention selector must
fall back to torch SDPA (its module flags are consulted per call, so patching
them here is sufficient and touches no vendored code). GPU runs keep the
native fast paths.
"""
import torch

if not torch.cuda.is_available():
    from diffsynth.models import wan_video_dit

    wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
    wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
    wan_video_dit.SAGE_ATTN_AVAILABLE = False
