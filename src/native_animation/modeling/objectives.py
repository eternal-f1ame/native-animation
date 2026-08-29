"""Native Animation v2 objective.

Animation purposes: spend supervision on the motion beats (weighted velocity),
punish flicker and mid-clip collapse (sigma-uniform delta consistency), and
honor the animator's contract (anchors excluded — the clamp supervises them).
Formulated in flow matching; every term is substrate-portable (spec §1).
"""
from __future__ import annotations

import torch

from native_animation.modeling.anchoring import anchor_frame_mask
from native_animation.modeling.native_flowmatch import _motion_frame_weights, _weighted_mse


def _delta_weights(motion_weights: torch.Tensor | None, num_frames: int) -> torch.Tensor | None:
    """Motion weights aligned to the T'-1 delta positions (drop the pad slot)."""
    if motion_weights is None:
        return None
    if motion_weights.shape[2] == num_frames:      # padded to T'
        return motion_weights[:, :, 1:]
    return motion_weights


def native_animation_v2_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    input_latents: torch.Tensor,
    anchors: list[int],
    alpha: float = 1.0,
    lambda0: float = 0.25,
    delta_mode: str = "vspace",
    sigma: torch.Tensor | None = None,
) -> dict:
    """Compute the v2 loss on (B, C, T', H, W) tensors.

    ``delta_mode``: "vspace" (default, sigma-uniform), "legacy_x0_needs_sigma"
    (v1's sigma^2-scaled behavior, kept for the ablation grid), or "off".
    """
    num_frames = input_latents.shape[2]
    mask = anchor_frame_mask(num_frames, anchors).to(v_pred.device)
    free = ~mask

    # --- Motion-weighted velocity loss on non-anchor frames only. ---
    motion_weights = _motion_frame_weights(input_latents, anchor_frames=0,
                                           motion_weighting_scale=alpha)
    vel_weights = motion_weights[:, :, free] if motion_weights is not None else None
    velocity = _weighted_mse(v_pred[:, :, free], v_target[:, :, free], vel_weights)

    # --- Delta consistency: anti-flicker on the velocity-error field. ---
    if lambda0 > 0 and delta_mode != "off" and num_frames > 1:
        pred_seq, target_seq = v_pred, v_target
        if anchors:                                # anchors contribute clean values
            idx = torch.tensor(anchors, dtype=torch.long, device=v_pred.device)
            pred_seq = v_pred.clone()
            pred_seq[:, :, idx] = v_target[:, :, idx]
        if delta_mode == "vspace":
            pred_deltas = pred_seq[:, :, 1:] - pred_seq[:, :, :-1]
            target_deltas = target_seq[:, :, 1:] - target_seq[:, :, :-1]
        elif delta_mode == "legacy_x0_needs_sigma":
            if sigma is None:
                raise ValueError("legacy delta mode requires sigma")
            s = sigma.reshape(-1, 1, 1, 1, 1).to(v_pred.device)
            x0_error = -s * (pred_seq - target_seq)          # x0-error = -sigma * e
            pred_deltas = x0_error[:, :, 1:] - x0_error[:, :, :-1]
            target_deltas = torch.zeros_like(pred_deltas)
        else:
            raise ValueError(f"unknown delta_mode: {delta_mode}")
        delta = _weighted_mse(pred_deltas, target_deltas,
                              _delta_weights(motion_weights, num_frames))
    else:
        delta = torch.zeros((), device=v_pred.device)

    total = velocity + lambda0 * delta
    return {"total": total, "velocity": velocity, "delta": delta}
