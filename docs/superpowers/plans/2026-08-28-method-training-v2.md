# Method + Training v2 Implementation Plan (v2 Plan 2 of 3)

> **Status (2026-08-30): CODE COMPLETE** — all modules TDD'd green (~85 tests); FSDP + memory smokes done (256px fits 4×80GB; 480p needs the CT-b levers). **Pending:** CT-a launch once Stage-0 data lands.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Native Animation v2 method (anchored conditioning, v2 objectives, timestep density, curriculum) and the CT/SFT training program with a resumable multi-GPU runner, anchored inference, and smoke gates — ready to launch the moment Stage-0 data lands.

**Architecture:** Pure-math modules under `src/native_animation/modeling/` and `training/` (CPU-testable, no weights), a v2 training module that reuses the DiffSynth Wan pipeline but replaces its noising/timestep/loss path entirely, a v2 runner replacing DiffSynth's 40-line loop (cosine+warmup, EMA, grad clip, `accelerate` save/load-state resume, curriculum refresh), and a targeted fork of `model_fn_wan_video` whose only change is the per-token timestep construction (zeros at ALL anchor frames, not just frame 0) — protected by a parity test. GT-anchored DPO (spec §2 Stage 3) is implemented in Plan 3 alongside the evaluator it depends on.

**Tech Stack:** PyTorch (comfy env: torch 2.7.1), DiffSynth vendored runtime (`src/diffsynth/`), accelerate + FSDP for 4–8×80GB full-FT, SLURM.

**Spec:** `docs/superpowers/specs/2026-08-28-native-fm-v2-design.md` §1 (method), §2 Stages 1–2, §4 (implementation + smoke gates). Foundations: `docs/method-v2-foundations.md`.

## Global Constraints

- `WS=/home/aeternum/Research/Comic/Cartoon`, `COMFY_PY=/home/aeternum/anaconda3/envs/comfy/bin/python`; every python invocation prefixed `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1`. Heavy work on SLURM only.
- Framing rule: modules/docs/messages lead with the animation purpose; flow matching is the substrate.
- v1 stays intact as the ablation baseline: `modeling/native_flowmatch.py` and `training/train.py` are not modified (v2 imports v1 helpers where DRY).
- Recon facts this plan is built on (verified in `src/diffsynth` on 2026-08-28):
  - TI2V conditioning: unit outputs `first_frame_latents` + `fuse_vae_embedding_in_latents=True`; inference loop re-clamps `latents[:, :, 0:1]` after every `scheduler.step` (`pipelines/wan_video.py:335`).
  - `model_fn_wan_video` (`pipelines/wan_video.py:1276`): when `dit.seperated_timestep and fuse_vae_embedding_in_latents`, builds a flattened per-token timestep = `concat([zeros(1, h·w/4), ones(T'−1, h·w/4)·t])`, then `t_mod = time_projection(time_embedding(sinusoidal(...).unsqueeze(0))).unflatten(2, (6, dim))`.
  - `FlowMatchScheduler.add_noise/training_target/training_weight` operate on a precomputed 1000-step grid — v2 bypasses the scheduler for training entirely (direct per-sample σ).
  - `launch_training_task` (`diffusion/runner.py:8`): AdamW + ConstantLR + `DataLoader(shuffle=True, collate_fn=lambda x: x[0])` (batch 1/process) + accelerate prepare + grad-accum loop; no grad clip, no EMA, no state resume → v2 runner replaces it.
- Every new pure function gets a test written FIRST (RED verified) — the standing suite (48 tests) must stay green.

---

### Task 1: Timestep density — protect the drawing through the noise schedule

**Files:**
- Create: `src/native_animation/modeling/timesteps.py`
- Test: `tests/test_timesteps.py`

**Interfaces:**
- Produces: `class TimestepDensity(m: float = 0.0, s: float = 1.0, shift: float = 3.0, tail_p: float = 0.05, tail_lo: float = 0.95)` with `sample(n: int, generator: torch.Generator | None) -> torch.Tensor` (σ ∈ (0,1), shape `(n,)`) and `shift_map(u: torch.Tensor) -> torch.Tensor` (`shift·u/(1+(shift−1)·u)`). Task 7's module consumes `sample`; the sweep grid varies the constructor args via config.

- [ ] **Step 1: Failing tests**

```python
# tests/test_timesteps.py
import torch

from native_animation.modeling.timesteps import TimestepDensity


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_samples_live_in_open_unit_interval():
    sigma = TimestepDensity().sample(10_000, _gen())
    assert sigma.shape == (10_000,)
    assert float(sigma.min()) > 0.0 and float(sigma.max()) < 1.0


def test_shift_map_is_monotone_and_pins_endpoints():
    density = TimestepDensity(shift=3.0)
    u = torch.linspace(0.001, 0.999, 100)
    mapped = density.shift_map(u)
    assert torch.all(mapped[1:] > mapped[:-1])
    assert torch.allclose(density.shift_map(torch.tensor([0.0, 1.0])), torch.tensor([0.0, 1.0]))
    # shift>1 pushes mass toward high sigma: median above 0.5 for uniform input
    assert float(density.shift_map(torch.tensor(0.5))) > 0.5


def test_tail_fraction_is_respected():
    sigma = TimestepDensity(tail_p=0.05, tail_lo=0.95).sample(100_000, _gen())
    frac_high = float((sigma >= 0.95).float().mean())
    assert 0.04 < frac_high < 0.10   # 5% forced tail + logit-normal mass that lands there


def test_mean_parameter_moves_the_median():
    low = TimestepDensity(m=-1.0, tail_p=0.0).sample(50_000, _gen(1)).median()
    high = TimestepDensity(m=+1.0, tail_p=0.0).sample(50_000, _gen(1)).median()
    assert float(high) > float(low)
```

- [ ] **Step 2: Verify RED** (`pytest tests/test_timesteps.py -q` → collection error).

- [ ] **Step 3: Implement**

```python
# src/native_animation/modeling/timesteps.py
"""Timestep density: keep training emphasis where line art and identity are
decided, instead of the heavy-noise regime that erases the drawing.

sigma ~ shift_map(LogitNormal(m, s)) with a small uniform tail on
[tail_lo, 1) for low-SNR coverage (spec §1.4). Sampled per-sample, replacing
the scheduler-grid timesteps of v1.
"""
from __future__ import annotations

import torch


class TimestepDensity:
    def __init__(self, m: float = 0.0, s: float = 1.0, shift: float = 3.0,
                 tail_p: float = 0.05, tail_lo: float = 0.95):
        self.m, self.s, self.shift = m, s, shift
        self.tail_p, self.tail_lo = tail_p, tail_lo

    def shift_map(self, u: torch.Tensor) -> torch.Tensor:
        return self.shift * u / (1 + (self.shift - 1) * u)

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        normal = torch.randn(n, generator=generator) * self.s + self.m
        u = torch.sigmoid(normal)                      # LogitNormal(m, s)
        sigma = self.shift_map(u)
        if self.tail_p > 0:
            tail = torch.rand(n, generator=generator) < self.tail_p
            uniform_hi = self.tail_lo + torch.rand(n, generator=generator) * (1 - self.tail_lo)
            sigma = torch.where(tail, uniform_hi, sigma)
        return sigma.clamp(1e-4, 1 - 1e-4)
```

- [ ] **Step 4: Verify GREEN + suite.** **Step 5: Commit** (`"Add timestep density (logit-normal + shift + low-SNR tail)"`).

---

### Task 2: Anchoring — the animator's contract (sampling, masks, per-token timesteps)

**Files:**
- Create: `src/native_animation/modeling/anchoring.py`
- Test: `tests/test_anchoring.py`

**Interfaces:**
- Produces: `sample_anchor_set(num_latent_frames: int, rng: random.Random, probs: dict | None) -> list[int]` (sorted; modes keyframe/.5, first_last/.25, storyboard/.15 with k~U{1,2,3} interior, none/.10); `anchor_frame_mask(num_latent_frames: int, anchors: list[int]) -> torch.BoolTensor` shape `(T',)`; `apply_anchor_clamp(noisy: torch.Tensor, clean: torch.Tensor, anchors: list[int]) -> torch.Tensor` (clamps dim-2 slices); `build_separated_timestep(t: torch.Tensor, num_latent_frames: int, lat_h: int, lat_w: int, anchors: list[int]) -> torch.Tensor` — flattened per-token vector, zeros at anchor rows, `t` elsewhere, `(T'·h·w/4,)`, matching upstream's construction exactly for `anchors=[0]`. Tasks 3/4/7/8 consume all four.

- [ ] **Step 1: Failing tests**

```python
# tests/test_anchoring.py
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
```

- [ ] **Step 2: Verify RED.**

- [ ] **Step 3: Implement**

```python
# src/native_animation/modeling/anchoring.py
"""Anchoring: the animator's contract — given drawings are inviolable.

Training-side: sample an anchor set per clip, clamp anchor latents clean, and
exclude anchors from supervision. Model-side: TI2V's separated-timestep DiT
runs conditioned frames at t=0; ``build_separated_timestep`` generalizes the
upstream frame-0-only construction (pipelines/wan_video.py:1376) to any
anchor set, byte-compatible with upstream for anchors=[0].
"""
from __future__ import annotations

import random

import torch

DEFAULT_MODE_PROBS = {"keyframe": 0.50, "first_last": 0.25, "storyboard": 0.15, "none": 0.10}


def sample_anchor_set(num_latent_frames: int, rng: random.Random,
                      probs: dict | None = None) -> list[int]:
    probs = probs or DEFAULT_MODE_PROBS
    modes, weights = zip(*probs.items())
    mode = rng.choices(modes, weights=weights, k=1)[0]
    last = num_latent_frames - 1
    if mode == "none":
        return []
    if mode == "keyframe":
        return [0]
    if mode == "first_last":
        return [0, last]
    k = rng.randint(1, 3)
    interior = rng.sample(range(1, last), k=min(k, max(last - 1, 0)))
    return sorted([0] + interior)


def anchor_frame_mask(num_latent_frames: int, anchors: list[int]) -> torch.Tensor:
    mask = torch.zeros(num_latent_frames, dtype=torch.bool)
    if anchors:
        mask[torch.tensor(anchors, dtype=torch.long)] = True
    return mask


def apply_anchor_clamp(noisy: torch.Tensor, clean: torch.Tensor,
                       anchors: list[int]) -> torch.Tensor:
    out = noisy.clone()
    if anchors:
        idx = torch.tensor(anchors, dtype=torch.long, device=noisy.device)
        out[:, :, idx] = clean[:, :, idx].to(out.dtype)
    return out


def build_separated_timestep(t: torch.Tensor, num_latent_frames: int,
                             lat_h: int, lat_w: int, anchors: list[int]) -> torch.Tensor:
    tokens_per_frame = lat_h * lat_w // 4
    per_frame = torch.ones(num_latent_frames, tokens_per_frame,
                           dtype=t.dtype if t.is_floating_point() else torch.float32,
                           device=t.device) * t
    if anchors:
        per_frame[torch.tensor(anchors, dtype=torch.long, device=t.device)] = 0.0
    return per_frame.flatten()
```

- [ ] **Step 4: Verify GREEN + suite.** **Step 5: Commit** (`"Add anchoring: mode sampling, masks, generalized separated timesteps"`).

---

### Task 3: v2 objectives — motion-weighted velocity + σ-uniform delta consistency

**Files:**
- Create: `src/native_animation/modeling/objectives.py`
- Test: `tests/test_objectives.py`

**Interfaces:**
- Consumes: `_motion_frame_weights`, `_weighted_mse` from `native_animation.modeling.native_flowmatch` (v1, unchanged); `anchor_frame_mask` (Task 2).
- Produces: `native_animation_v2_loss(v_pred, v_target, input_latents, anchors: list[int], alpha: float = 1.0, lambda0: float = 0.25, delta_mode: str = "vspace") -> dict` with keys `total, velocity, delta` (delta_mode ∈ {"vspace", "legacy_x0_needs_sigma", "off"}; legacy arm additionally takes `sigma`, kept for the ablation grid). Tensors are `(B, C, T', H, W)`. Task 7 calls this inside the training module.

- [ ] **Step 1: Failing tests**

```python
# tests/test_objectives.py
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
    # v-space residual == Delta(e): constant per-frame error e_t = c has zero
    # delta penalty; alternating error has a large one (anti-flicker semantics).
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
```

- [ ] **Step 2: Verify RED.**

- [ ] **Step 3: Implement**

```python
# src/native_animation/modeling/objectives.py
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
    if motion_weights.shape[2] == num_frames:      # padded to T' (anchorless path)
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
    num_frames = input_latents.shape[2]
    mask = anchor_frame_mask(num_frames, anchors).to(v_pred.device)
    free = ~mask

    # --- Motion-weighted velocity loss on non-anchor frames only. ---
    motion_weights = _motion_frame_weights(input_latents, anchor_frames=0,
                                           motion_weighting_scale=alpha)
    if motion_weights is not None:
        vel_weights = motion_weights[:, :, free]
    else:
        vel_weights = None
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
            x0_pred = -s * (pred_seq - target_seq) + 0.0  # x0-error = -sigma * e (v1 behavior)
            pred_deltas = x0_pred[:, :, 1:] - x0_pred[:, :, :-1]
            target_deltas = torch.zeros_like(pred_deltas)
        else:
            raise ValueError(f"unknown delta_mode: {delta_mode}")
        delta = _weighted_mse(pred_deltas, target_deltas,
                              _delta_weights(motion_weights, num_frames))
    else:
        delta = torch.zeros((), device=v_pred.device)

    total = velocity + lambda0 * delta
    return {"total": total, "velocity": velocity, "delta": delta}
```

- [ ] **Step 4: Verify GREEN + suite.** **Step 5: Commit** (`"Add v2 objective: motion-weighted velocity + sigma-uniform delta consistency"`).

---

### Task 4: Anchored model function — targeted fork with parity guard

**Files:**
- Create: `src/native_animation/modeling/anchored_model_fn.py`
- Test: `tests/test_anchored_model_fn.py`

**Interfaces:**
- Consumes: `build_separated_timestep` (Task 2); `model_fn_wan_video` upstream.
- Produces: `make_anchored_model_fn(anchors_ref: dict) -> callable` — a drop-in for `pipe.model_fn` that reads the current anchor list from the mutable `anchors_ref["anchors"]` and calls upstream `model_fn_wan_video` after swapping in the generalized per-token timestep. Implementation strategy: NOT a copy of the 130-line upstream function — instead a **wrapper** that pre-builds the timestep vector and passes `fuse_vae_embedding_in_latents=False` with a pre-shaped timestep... **verified impossible** (the else-branch unflattens at dim 1, wrong shape). Therefore: a minimal fork that monkey-wraps the ONE construction: temporarily patch `torch.concat`? No — the honest minimal fork is to copy `model_fn_wan_video` and replace the two-row concat with `build_separated_timestep`. To keep the fork from rotting, the parity test below runs BOTH functions on a tiny real `WanModel` and asserts identical outputs for `anchors=[0]`.

- [ ] **Step 1: Read `models/wan_video_dit.py` `WanModel.__init__`** and record the smallest valid constructor for a tiny model (target: `dim≈64, num_heads=2, num_layers=1, ffn_dim=128, patch_size=(1,2,2), text_dim` matching, `seperated_timestep=True, fuse_vae_embedding_in_latents=True` flags as constructor args or attributes — set attributes directly after construction if not args). If a tiny CPU-constructible model proves impossible within ~30 minutes, FALL BACK to: parity test of the timestep vector only (already in Task 2) + integration deferred to the Task 9 smoke — record the fallback in the commit message.

- [ ] **Step 2: Failing test**

```python
# tests/test_anchored_model_fn.py
"""Parity guard: the fork must equal upstream for the frame-0 anchor case."""
import torch

from native_animation.modeling.anchored_model_fn import model_fn_wan_video_anchored
from diffsynth.pipelines.wan_video import model_fn_wan_video

from tests.tiny_wan import build_tiny_wan_model  # helper written in Step 1


def test_fork_matches_upstream_for_frame0_anchor():
    torch.manual_seed(0)
    dit = build_tiny_wan_model()
    latents = torch.randn(1, dit.in_dim, 5, 4, 4)
    context = torch.randn(1, 7, dit.text_dim)
    t = torch.tensor([500.0])
    up = model_fn_wan_video(dit=dit, latents=latents, timestep=t, context=context,
                            fuse_vae_embedding_in_latents=True)
    ours = model_fn_wan_video_anchored(dit=dit, latents=latents, timestep=t, context=context,
                                       fuse_vae_embedding_in_latents=True, anchors=[0])
    assert torch.allclose(up, ours, atol=1e-5)


def test_fork_diverges_when_extra_anchors_are_added():
    torch.manual_seed(0)
    dit = build_tiny_wan_model()
    latents = torch.randn(1, dit.in_dim, 5, 4, 4)
    context = torch.randn(1, 7, dit.text_dim)
    t = torch.tensor([500.0])
    a0 = model_fn_wan_video_anchored(dit=dit, latents=latents, timestep=t, context=context,
                                     fuse_vae_embedding_in_latents=True, anchors=[0])
    a03 = model_fn_wan_video_anchored(dit=dit, latents=latents, timestep=t, context=context,
                                      fuse_vae_embedding_in_latents=True, anchors=[0, 3])
    assert not torch.allclose(a0, a03)
```

- [ ] **Step 3: Implement** — copy `model_fn_wan_video`'s body into `model_fn_wan_video_anchored(..., anchors: list[int])` (keep only the code paths our calls exercise: the baseline DiT path incl. separated-timestep, USP guards left intact; delete the audio/vace/motion-controller branches with a header note pointing at upstream), replacing the timestep concat with:

```python
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = build_separated_timestep(
            timestep.flatten()[0], latents.shape[2], latents.shape[3], latents.shape[4],
            anchors=anchors if anchors is not None else [0],
        ).to(dtype=latents.dtype, device=latents.device)
```

plus `make_anchored_model_fn(anchors_ref)` returning `functools.partial`-style closure that injects `anchors=anchors_ref["anchors"]` per call. Also write `tests/tiny_wan.py` with `build_tiny_wan_model()` from Step 1's findings.

- [ ] **Step 4: GREEN + suite.** **Step 5: Commit** (`"Add anchored model_fn fork with upstream parity guard"`).

---

### Task 5: Curriculum — from near-physical to full sakuga without collapse

**Files:**
- Create: `src/native_animation/training/curriculum.py`, `src/native_animation/training/__init__.py` update if needed
- Test: `tests/test_curriculum.py`

**Interfaces:**
- Consumes: v2 metadata rows (dicts with `q_motion`, `q_deform`, plus rebalance axis columns e.g. `series`).
- Produces: `difficulty(row, bucket_cols=("q_motion", "q_deform"), q=4) -> float` (mean normalized ∈[0,1]); `curriculum_weight(d: float, tau: float, gamma: float = 8.0, beta: float = 0.25) -> float` (σ(γ(τ−d+β)));
`rebalance_weights(rows, axis_cols=("series", "q_motion"), exponent=0.7) -> list[float]`; `class CurriculumSampler(rows, seed, ...)` with `refresh(tau)` recomputing combined probabilities and `sample_index() -> int`; `class CurriculumDataset(torch.utils.data.Dataset)` wrapping a base dataset + sampler (`__getitem__(i)` ignores `i`, draws by weight — DiffSynth's hardwired shuffle then costs nothing). Task 7 wires it around `UnifiedDataset`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_curriculum.py
import random

from native_animation.training.curriculum import (
    CurriculumSampler,
    curriculum_weight,
    difficulty,
    rebalance_weights,
)


def _rows():
    rows = []
    for i in range(400):
        rows.append({"q_motion": (i % 4) + 1, "q_deform": (i // 100) + 1,
                     "series": f"s{i % 8}"})
    return rows


def test_difficulty_normalizes_to_unit_interval():
    assert difficulty({"q_motion": 1, "q_deform": 1}) == 0.0
    assert difficulty({"q_motion": 4, "q_deform": 4}) == 1.0
    assert abs(difficulty({"q_motion": 1, "q_deform": 4}) - 0.5) < 1e-9


def test_curriculum_gate_opens_with_tau():
    hard = 1.0
    assert curriculum_weight(hard, tau=0.0) < 0.01
    assert curriculum_weight(hard, tau=1.0) > 0.85
    easy = 0.0
    assert curriculum_weight(easy, tau=0.0) > 0.85


def test_rebalance_upweights_rare_series():
    rows = [{"series": "big", "q_motion": 1}] * 90 + [{"series": "small", "q_motion": 1}] * 10
    weights = rebalance_weights(rows, axis_cols=("series",), exponent=0.7)
    assert weights[-1] > weights[0]


def test_sampler_hard_fraction_grows_with_tau():
    rows = _rows()
    sampler = CurriculumSampler(rows, seed=0)

    def hard_fraction(tau, draws=4000):
        sampler.refresh(tau)
        hard = sum(difficulty(rows[sampler.sample_index()]) > 0.66 for _ in range(draws))
        return hard / draws

    assert hard_fraction(0.05) < hard_fraction(0.95) - 0.1
```

- [ ] **Step 2: RED.** **Step 3: Implement**

```python
# src/native_animation/training/curriculum.py
"""Curriculum: a controlled migration from near-physical motion to full sakuga.

Difficulty = mean normalized (motion, deformation) quantile bucket; sampling
probability = sigmoid gate x long-tail rebalance weight (spec §2 Stage 2,
AniMatrix Eq. 16 + Eq. 2 adapted to our metadata columns).
"""
from __future__ import annotations

import math
import random
from collections import Counter

import torch


def difficulty(row: dict, bucket_cols=("q_motion", "q_deform"), q: int = 4) -> float:
    values = [(int(row[c]) - 1) / (q - 1) for c in bucket_cols]
    return sum(values) / len(values)


def curriculum_weight(d: float, tau: float, gamma: float = 8.0, beta: float = 0.25) -> float:
    return 1.0 / (1.0 + math.exp(-gamma * (tau - d + beta)))


def rebalance_weights(rows, axis_cols=("series", "q_motion"), exponent: float = 0.7):
    counts = {col: Counter(str(row[col]) for row in rows) for col in axis_cols}
    weights = []
    for row in rows:
        product = 1.0
        for col in axis_cols:
            product *= 1.0 / counts[col][str(row[col])]
        weights.append(product ** exponent)
    return weights


class CurriculumSampler:
    def __init__(self, rows, seed: int = 0, bucket_cols=("q_motion", "q_deform"),
                 axis_cols=("series", "q_motion"), gamma: float = 8.0, beta: float = 0.25):
        self.rows = rows
        self.rng = random.Random(seed)
        self.difficulties = [difficulty(r, bucket_cols) for r in rows]
        self.rebalance = rebalance_weights(rows, axis_cols)
        self.gamma, self.beta = gamma, beta
        self.refresh(tau=0.0)

    def refresh(self, tau: float) -> None:
        raw = [curriculum_weight(d, tau, self.gamma, self.beta) * w
               for d, w in zip(self.difficulties, self.rebalance)]
        total = sum(raw)
        self.cumulative = []
        acc = 0.0
        for value in raw:
            acc += value / total
            self.cumulative.append(acc)

    def sample_index(self) -> int:
        u = self.rng.random()
        lo, hi = 0, len(self.cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cumulative[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo


class CurriculumDataset(torch.utils.data.Dataset):
    """Wrap a base dataset: every __getitem__ draws by curriculum probability."""

    def __init__(self, base, sampler: CurriculumSampler):
        self.base, self.sampler = base, sampler

    def __len__(self):
        return len(self.base)

    def __getitem__(self, _index):
        return self.base[self.sampler.sample_index()]
```

- [ ] **Step 4: GREEN + suite.** **Step 5: Commit** (`"Add curriculum: difficulty gate + rebalancing + resampling dataset"`).

---

### Task 6: Runner v2 — cosine+warmup, EMA, grad clip, resumable state

**Files:**
- Create: `src/native_animation/training/runner_v2.py`
- Test: `tests/test_runner_v2.py`

**Interfaces:**
- Produces: `cosine_warmup_lambda(warmup_steps: int, total_steps: int) -> callable` (0→1 linear over warmup, cosine to 0.1 floor after); `class EMA(params, decay=0.995)` with `update(params)`, `copy_to(params)`, `state_dict/load_state_dict`; `train_loop(accelerator, model, dataset, config: dict, curriculum: CurriculumSampler | None)` — builds AdamW + LambdaLR, DataLoader (collate `x[0]`, workers from config), `accelerator.prepare`, loop with grad accumulation, `accelerator.clip_grad_norm_`, EMA update, `curriculum.refresh(step/total)` every `config["curriculum_refresh_steps"]`, `accelerator.save_state(dir/step_N)` every `config["save_state_steps"]` keeping last 2, model checkpoint via the DiffSynth `ModelLogger` pattern every `config["save_steps"]`, JSONL loss log, and `--resume`: `accelerator.load_state(latest)` + skip to step. Tasks 7's entrypoints call `train_loop`.

- [ ] **Step 1: Failing tests** (pure parts + a CPU round-trip)

```python
# tests/test_runner_v2.py
import torch

from native_animation.training.runner_v2 import EMA, cosine_warmup_lambda


def test_cosine_warmup_shape():
    lam = cosine_warmup_lambda(warmup_steps=10, total_steps=110)
    assert lam(0) == 0.0
    assert abs(lam(5) - 0.5) < 1e-9
    assert abs(lam(10) - 1.0) < 1e-9
    assert lam(60) < 1.0 and lam(60) > lam(110)
    assert abs(lam(110) - 0.1) < 0.02        # cosine floor


def test_ema_tracks_parameters():
    p = [torch.nn.Parameter(torch.zeros(3))]
    ema = EMA(p, decay=0.5)
    with torch.no_grad():
        p[0].add_(1.0)
    ema.update(p)
    assert torch.allclose(ema.shadow[0], torch.full((3,), 0.5))
    ema.update(p)
    assert torch.allclose(ema.shadow[0], torch.full((3,), 0.75))
    ema.copy_to(p)
    assert torch.allclose(p[0].data, torch.full((3,), 0.75))
    state = ema.state_dict()
    ema2 = EMA([torch.nn.Parameter(torch.zeros(3))], decay=0.5)
    ema2.load_state_dict(state)
    assert torch.allclose(ema2.shadow[0], ema.shadow[0])
```

- [ ] **Step 2: RED.** **Step 3: Implement** `cosine_warmup_lambda` (`step<warmup: step/warmup; else floor 0.1 + 0.9·0.5·(1+cos(π·progress))`), `EMA` (shadow on CPU float32; detached), and `train_loop` per the interface (structure mirrors `diffusion/runner.py:8` — read it side-by-side; add the six additions listed; keep `collate_fn=lambda x: x[0]`). `train_loop` itself is exercised by Task 9's tiny e2e, not unit tests.

- [ ] **Step 4: GREEN + suite.** **Step 5: Commit** (`"Add v2 runner: cosine+warmup, EMA, grad clip, resumable state"`).

---

### Task 7: v2 training module + CT/SFT entrypoints + configs + FSDP + sbatch

**Files:**
- Create: `src/native_animation/training/module_v2.py`, `src/native_animation/training/stages/__init__.py`, `src/native_animation/training/stages/train_stage.py` (one entrypoint, stage picked by config), `configs/ct_a.yaml`, `configs/ct_b.yaml`, `configs/sft.yaml`, `configs/accelerate_fsdp.yaml`, `scripts/slurm/train_v2.sbatch`
- Test: `tests/test_module_v2.py`

**Interfaces:**
- Consumes: everything above + DiffSynth (`WanVideoPipeline`, `UnifiedDataset`, `ModelLogger`).
- Produces: `NativeAnimationV2Module(DiffusionTrainingModule)` — builds the Wan pipe exactly like v1's module (same `parse_model_configs` / tokenizer path), sets `pipe.model_fn = make_anchored_model_fn(self.anchors_ref)`, and overrides the loss path: after the pipeline units produce `inputs_shared` (with `input_latents`), it (1) samples `anchors = sample_anchor_set(T', rng, cfg)` and stores into `anchors_ref`, (2) samples `sigma = density.sample(1)`, (3) noises `z_σ = (1−σ)z + σε`, clamps via `apply_anchor_clamp`, (4) runs `pipe.model_fn` with `timestep = sigma·1000` (scalar; the anchored fork builds the per-token vector), (5) computes `native_animation_v2_loss(...)`. Text dropout p=0.1 implemented by swapping the prompt for `""` before the text-encode unit. Also produces the **pure orchestration function** `compute_v2_loss(model_fn, input_latents, context_kwargs, density, rng, cfg) -> dict` that the module delegates to — this is what the unit test drives with a stub model.
- The single stage entrypoint `train_stage.py` reads a YAML config: dataset CSV path, curriculum on/off + params, density params, objective params, lr/warmup/steps/save cadence, resolution/frames, trainable=dit(full) — CT-a/CT-b/SFT differ only by config file.

- [ ] **Step 1: Failing test** — drive `compute_v2_loss` with a stub `model_fn` returning `v_target + e` for a fixed `e`; assert: returns finite dict; anchors list recorded in `anchors_ref`; loss equals `native_animation_v2_loss` called directly with the same pieces (wiring correctness); text-dropout branch replaces prompt (probe via a spy dict).

```python
# tests/test_module_v2.py
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
```

(`compute_v2_loss`'s stub contract: it must call `model_fn` with kwargs `latents_in` (the clamped noisy latents), `timestep` (σ·1000 scalar tensor), `anchors`, and `v_target` (ε−z, so the stub can echo it) — the REAL adapter inside `module_v2` maps these onto the DiffSynth `model_fn` signature and in-pipeline models; the `v_target` kwarg is consumed only by tests via a `_expose_target=True` flag, default off.)

- [ ] **Step 2: RED.** **Step 3: Implement** `compute_v2_loss` + `NativeAnimationV2Module` (subclassing v1's `NativeAnimationWanTrainingModule` where the pipeline-building boilerplate can be inherited; override `compute_sft_loss` path and drop the v1 scheduler swap — v2 does not use the scheduler at training time). **Step 4: GREEN + suite.**

- [ ] **Step 5: Write the four configs + sbatch.** `configs/ct_a.yaml` (256×448, 17f, B-tier CSV, curriculum off, lr 5e-5, warmup 500, EMA 0.995, clip 1.0, save_state 500, density defaults), `ct_b.yaml` (480×832, 49f, rest same), `sft.yaml` (A/S-tier CSV filter, curriculum on γ=8 β=0.25 refresh 200, lr 2e-5), `accelerate_fsdp.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: 4            # raise to 8 when an 8-GPU node is held
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: DiTBlock   # confirm class name in models/wan_video_dit.py at execution
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_offload_params: false
```

`scripts/slurm/train_v2.sbatch`: partition gpu, `--gpus-per-node=4`, `--constraint=gmem80`, `--time=2-00:00:00`, sources `paths.env`, `accelerate launch --config_file configs/accelerate_fsdp.yaml --num_processes $SLURM_GPUS_ON_NODE src/native_animation/training/stages/train_stage.py --config $STAGE_CONFIG --resume auto`.

- [ ] **Step 6: Commit** (`"Add v2 training module, stage entrypoint, configs, FSDP + SLURM"`).

---

### Task 8: Anchored inference

**Files:**
- Create: `src/native_animation/inference/anchored.py`
- Test: `tests/test_anchored_inference.py`

**Interfaces:**
- Produces: `encode_anchor_latents(pipe, anchor_images: dict[int, PIL.Image], height, width, num_frames) -> dict[int, torch.Tensor]` (per-anchor VAE latents via the same preprocess path as the pipeline's `first_frame_latents` unit at `wan_video.py:523`); `clamping_step(scheduler_step, anchor_latents: dict[int, torch.Tensor]) -> callable` — wraps `scheduler.step` so every denoise step re-clamps all anchor slots (generalizing the frame-0 clamp at `wan_video.py:335`); `anchored_generate(pipe, prompt, anchor_images, **gen_kwargs) -> frames` — installs the anchored model_fn (`anchors_ref`), wraps the scheduler step, passes `input_image=anchor_images[0]` so the stock unit path runs, restores everything after.

- [ ] **Step 1: Failing test** (the wrapper, with a fake scheduler)

```python
# tests/test_anchored_inference.py
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
```

- [ ] **Step 2: RED.** **Step 3: Implement** (`clamping_step` trivial; `encode_anchor_latents` mirrors the unit at `wan_video.py:523-530`; `anchored_generate` = context-managed install/restore of `pipe.model_fn` and `pipe.scheduler.step`). **Step 4: GREEN + suite.** **Step 5: Commit** (`"Add anchored inference: multi-keyframe clamped sampling"`).

---

### Task 9: Smoke gates (spec §4) — memory, VAE fidelity, tiny end-to-end

**Files:**
- Create: `tools/vae_fidelity_check.py`, `scripts/slurm/smoke_memory.sbatch`, `scripts/slurm/smoke_tiny_e2e.sbatch`

- [ ] **Step 1: `tools/vae_fidelity_check.py`** — load pipe VAE only; for N test keyframes: encode→decode at 480×832, report per-image PSNR + edge-preservation (Canny-edge IoU between source and round-trip) + save side-by-side PNGs to `experiments/smoke/vae_fidelity/`. Pass gate: median PSNR ≥ 28 and median edge-IoU ≥ 0.6 (calibration numbers — judge the side-by-sides by eye and record the verdict).
- [ ] **Step 2: `smoke_memory.sbatch`** — 4×gmem80, runs `train_stage.py --config configs/ct_b.yaml --smoke-steps 3 --dataset-limit 8`: full-FT 3 optimizer steps at 480×832×49; prints `torch.cuda.max_memory_allocated()` per rank + seconds/step. Pass: no OOM; record s/step → compute CT step budgets and write them into `configs/ct_*.yaml` (`total_steps`).
- [ ] **Step 3: `smoke_tiny_e2e.sbatch`** — 1×gmem48, `ct_a.yaml` on a 100-row CSV slice, 30 steps + one `anchored_generate` sample at the end; verifies the whole loop (data→loss→backward→save_state→resume→inference) on one GPU.
- [ ] **Step 4:** Run all three (they gate Stage-1 launch, not this plan's completion if the queue is busy — submit and record job IDs). **Step 5: Commit** tools+sbatch (`"Add v2 smoke gates: memory, VAE fidelity, tiny e2e"`).

---

### Task 10: Verification + docs

- [ ] **Step 1:** Full suite green (expect ~65 tests).
- [ ] **Step 2:** `docs/method.md`: add a short "v2" section (anchoring generalization, corrected delta term with the σ² derivation pointer, timestep density, curriculum) — animation-purpose-first wording.
- [ ] **Step 3:** Repo `CLAUDE.md`: add the v2 commands (`sbatch scripts/slurm/train_v2.sbatch` with `STAGE_CONFIG`, smoke gates, anchored inference example).
- [ ] **Step 4:** `git status` clean, push, report: what is launch-ready, which smoke gates passed, measured s/step and the derived CT/SFT step budgets.

---

## Self-review record

- **Spec coverage:** §1.1→T2/T4/T7 (anchor sampling + per-token timesteps + clamp-and-exclude), §1.2–1.3→T3 (incl. legacy ablation arm), §1.4→T1, §1.5 ablation-tier C4/C8 deliberately not built until their ablation slots (grid runs from Stage-2; promoted only on wins — noted, not a gap), §2 Stage 1–2→T5/T6/T7, §2 Stage 3 (DPO)→**Plan 3** (depends on the per-sample GT scorer; stated in header), §3 multi-anchor eval→T8 provides the mechanism (runners in Plan 3), §4 smoke gates→T9, configs debt→T7.
- **Placeholders:** none — every step has code or an exact recipe; the two genuine unknowns (tiny-WanModel constructibility, FSDP wrap class name) carry explicit verify-then-decide instructions with fallbacks.
- **Type consistency:** `sample_anchor_set/anchor_frame_mask/apply_anchor_clamp/build_separated_timestep` names match across T2→T3/T4/T7/T8; `TimestepDensity.sample` matches T7's usage; `CurriculumSampler.refresh/sample_index` match T6's loop hook; `native_animation_v2_loss` signature matches T7's delegation; v1 helper imports (`_motion_frame_weights`, `_weighted_mse`) exist (verified in v1 source).
