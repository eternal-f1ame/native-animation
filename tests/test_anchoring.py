"""Anchoring: mode sampling, masks, clamp, generalized separated timesteps."""
import random
from collections import Counter

import torch

from native_animation.modeling.anchoring import (
    anchor_frame_mask,
    apply_anchor_clamp,
    build_separated_timestep,
    sample_anchor_set,
)


def test_anchor_mode_distribution_and_shapes():
    rng = random.Random(0)
    modes = Counter()
    for _ in range(4000):
        anchors = sample_anchor_set(13, rng, probs=None)
        if anchors == []:
            modes["none"] += 1
        elif anchors == [0]:
            modes["keyframe"] += 1
        elif anchors == [0, 12]:
            modes["first_last"] += 1
        else:
            modes["storyboard"] += 1
            assert anchors[0] == 0 and all(0 < a < 12 for a in anchors[1:])
            assert anchors == sorted(anchors) and 2 <= len(anchors) <= 4
    for mode, expect in [("keyframe", 0.5), ("first_last", 0.25), ("storyboard", 0.15), ("none", 0.10)]:
        assert abs(modes[mode] / 4000 - expect) < 0.04


def test_mask_and_clamp():
    mask = anchor_frame_mask(5, [0, 3])
    assert mask.tolist() == [True, False, False, True, False]
    noisy, clean = torch.zeros(1, 4, 5, 2, 2), torch.ones(1, 4, 5, 2, 2)
    out = apply_anchor_clamp(noisy, clean, [0, 3])
    assert torch.all(out[:, :, [0, 3]] == 1.0) and torch.all(out[:, :, [1, 2, 4]] == 0.0)
    assert torch.all(noisy == 0.0)   # input not mutated


def test_separated_timestep_matches_upstream_for_frame0():
    # Upstream (pipelines/wan_video.py:1376): concat([zeros(1, h*w//4), ones(T'-1, h*w//4)*t]).flatten()
    t = torch.tensor(431.0)
    lat_h, lat_w, frames = 6, 8, 13
    upstream = torch.concat([
        torch.zeros((1, lat_h * lat_w // 4)),
        torch.ones((frames - 1, lat_h * lat_w // 4)) * t,
    ]).flatten()
    ours = build_separated_timestep(t, frames, lat_h, lat_w, anchors=[0])
    assert torch.equal(ours, upstream)


def test_separated_timestep_zeros_every_anchor_row():
    t = torch.tensor(700.0)
    vec = build_separated_timestep(t, 5, 4, 4, anchors=[0, 3])
    per_frame = vec.reshape(5, 4)
    assert torch.all(per_frame[[0, 3]] == 0.0)
    assert torch.all(per_frame[[1, 2, 4]] == 700.0)


def test_separated_timestep_with_no_anchors_is_uniform():
    vec = build_separated_timestep(torch.tensor(5.0), 3, 2, 2, anchors=[])
    assert torch.all(vec == 5.0)
