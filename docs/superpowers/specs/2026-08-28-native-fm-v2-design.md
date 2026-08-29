# Native FM v2 — Method, Training Program, and Campaign Design

- **Date:** 2026-08-28 · **Status:** Approved in design review (all sections, all flagged calls)
- **Objective (unchanged):** keyframe → short native-animation clip on Sakugabooru-derived data; CVPR 2027 target (~mid-Nov 2026)
- **Companions:** `docs/method-v2-foundations.md` (intake: AniMatrix math + literature map, component menu C1–C10), `docs/related/animatrix.md` (strategic read + data plan). This spec selects and freezes the design; the implementation plan derives from it.

## 0. Settled context

- **Compute:** occasional 4–8×80GB multi-GPU nodes + gmem48 singles; **unlimited GPU-hour quota** — node availability is the only constraint. Full fine-tuning of a 5B dense model is in envelope (FSDP + activation checkpointing).
- **Backbone/init (decision A):** primary arm = **Wan2.2-TI2V-5B from vanilla weights**, our own anime CT-lite; **AniSora-V3.2 is a baseline, not an init**; optional late second arm on AniSora init if variant-compatible and time permits.
- **Spine (decision A):** *objective spine* — core = C1 + C2 + C3 + C6 + C7 (below). VLM annotations are **rich text prompts only**; the AniMatrix-style tag channel (C5) is explicitly out of scope (cited as future work). C4 (VideoJAM-lite) and C8 (style-preserving REPA) are ablation-tier: promoted into the paper only if they win their ablations.
- **Data:** the ~180k-clip corpus acquisition is in flight (snapshot mirror + delta scrape); Stage 0 below consumes it.

## 1. Method specification

### 1.1 Anchored conditional flow matching (C1)

Clean video latents `z ∈ R^{C×T'×h×w}` (Wan2.2 ST-VAE: 16× spatial, 4× temporal, T' = (T−1)/4+1). Anchor set `A ⊆ {0,…,T'−1}`. Forward noising with `σ` from the density in §1.4:

```
z_σ = (1−σ)·z + σ·ε,   ε ~ N(0,I),   then   z_σ[:, :, A] ← z[:, :, A]
```

The network predicts velocity `v̂ = v_θ(z_σ, σ, c_text)`; target `v* = ε − z`. The velocity loss is computed **only on Ā** (anchors are supervised by the clamp itself — v1's principle, generalized).

**Anchor-mode sampling per training example:**

| mode | A | p |
|---|---|---|
| keyframe (I2V) | {0} | 0.50 |
| first+last | {0, T'−1} | 0.25 |
| storyboard | {0} ∪ k random interior, k~U{1,2,3} | 0.15 |
| unconditional | ∅ | 0.10 |

Text dropout 0.10 (independent) for standard CFG. Inference accepts any anchor pattern → one model serves I2V, first-last interpolation, and sparse-storyboard conditioning. Probabilities are initial design values; the anchor-mode mix is a swept hyperparameter, not a constant of the method.

### 1.2 Motion-aware frame weighting (kept from v1, formalized)

Per-delta motion `m_t = mean|z_t − z_{t−1}|`, per-clip normalized `m̄_t = m_t / max_t m_t`; weights `w_t = 1 + α·m̄_t`, `α = 1.0`, applied with the mean-preserving normalizer (α=0 recovers unweighted MSE exactly — pinned by existing tests). Frame-level weights follow v1's convention (frame t takes the (t−1,t) delta weight; leading unanchored frame takes 1).

### 1.3 σ-corrected temporal-delta consistency (C2)

**The v1 flaw, derived:** with `x̂₀ = z_σ − σ·v̂` and `v̂ = v* + e`, the x₀-estimate error is `−σe`, so v1's delta residual `(Δx̂₀ − Δx₀) = −σ·Δe` puts a **σ²** factor on the loss — high-noise timesteps dominate exactly where `x̂₀` is least informative.

**v2 default — v-space form (mathematically ≡ the 1/σ² correction, numerically cleaner):**

```
L_Δ = λ₀ · Σ_t w̃_t · ‖(v̂_t − v̂_{t−1}) − (v*_t − v*_{t−1})‖²
```

whose residual is exactly `Δe` (σ-uniform). Anchor boundaries: the sequence used for deltas takes clean values at anchors (v̂ ≡ v* there), so anchor-adjacent deltas supervise continuity off the anchor — the v1 concat semantics, generalized to arbitrary A. `λ₀ = 0.25` initial (v1's value), swept.

**Ablation arms:** (a) v1-legacy x₀-space unweighted (the σ² behavior, as the "before"); (b) `λ(σ) = λ₀(1−σ)^κ`, κ∈{1,2} modulation on the v-space form (emphasizes near-data fidelity). The derivation goes in the paper appendix.

### 1.4 Explicit timestep density (C3)

Per-**sample** timesteps (fixes v1's one-timestep-per-batch variance): `u ~ LogitNormal(m, s)` mapped through the shift transform `σ = shift·u / (1 + (shift−1)·u)`, plus a 5% uniform floor on σ∈[0.95, 1] (low-SNR coverage). Defaults `m=0, s=1, shift=3` — v1's shift=3 becomes one point in a principled family. Sweep grid (small budget): `m∈{−0.5, 0, +0.5} × s∈{0.75, 1.0} × shift∈{2, 3, 5}`, selected on the Stage-2 validation metric, not FVD.

### 1.5 Total objective

```
L = L_vel^w (motion-weighted, Ā only) + L_Δ (v-space, σ-uniform)
```

Ablation-tier additions (each its own arm, promoted only on wins): **C4** VideoJAM-lite — a small head predicting latent frame-deltas as an auxiliary target (+ optional Inner-Guidance at inference); **C8** style-preserving representation alignment — TRD-style alignment restricted to *spatial* token relations against a DINOv2-class encoder (explicitly not temporal relations, to avoid importing the physics prior; the tension is argued in foundations §2.3).

## 2. Training program (one objective throughout)

### Stage 0 — Data (pipeline, mostly in flight)

Snapshot extraction ✓ (in flight) → delta scrape → **shot splitting** (PySceneDetect + TransNetV2, 2–10 s single-shot clips) → curation mini-cascade (static/dup/OCR/blur) → **motion profiling** (flow energy `m(x)`, non-rigid residual `d(x)` = flow minus global affine fit, style cluster `k(x)` from series/VLM style tag; quantile buckets Q=4) → **Qwen3-VL annotation** (AniMatrix's structured JSON schema + three-section `<tag>/<summary>/<description>` rewrite, local batch inference; `<summary>+<description>` feed the text encoder) → metadata build with series-split (fixed builder) + **quality tiers from community signals**: S = top score quantile (~5%), A = top ~30%, B = remainder; favorites as tiebreak; exact cutoffs set from corpus statistics when the merged corpus lands.

### Stage 1 — CT-lite (anime domain adaptation)

Full-FT of the 5B DiT (text encoder + VAE frozen) with the **full v2 objective already on** (single-objective story; anchor sampling per §1.1). Two sub-stages: (a) ≈256-class resolution (e.g., 256×448, /16-aligned), 17 frames, B-tier corpus, ~1–2 passes; (b) 480×832, 49 frames, B-tier. FSDP (accelerate) on 4–8×80GB; bf16; lr 5e-5 cosine w/ warmup; grad-clip 1.0; EMA 0.995; checkpoint-resume mandatory. Throughput measured in the smoke phase sets exact step budgets.

### Stage 2 — SFT-quality (curriculum)

A/S-tier data at 480×832/49f (81f arm optional later). Sampling probability = curriculum weight × rebalance weight:

```
P_τ(x) ∝ σ(γ_cur·(τ − D(b(x)) + β_cur)) · (Π_axes 1/n_axis)^0.7,   D(b) = mean of normalized quantile indices
```

`τ` = Stage-2 progress ∈[0,1]; `γ_cur, β_cur` tuned so the hardest bucket reaches full inclusion by τ≈0.7 (implementation: weighted sampler refreshed every N steps). lr 2e-5. This stage's checkpoint is the ablation-grid base.

### Stage 3 — GT-anchored preference (C7)

From ~10–20k held-out-series keyframes: N=4 candidates each (Stage-2 model, varied seeds), scored **against the true continuation** by the per-sample GT suite (CFS/TCS/WorstSeg/DFS → FinalScore; + AnimeReward as second scorer if its weights are public — availability check in the plan). All ordered pairs with score gap ≥ δ and loser above a floor (reject degenerate winners); GAPO-style gap weighting on the pair loss. **Diffusion-DPO on a LoRA (rank 64) over the frozen Stage-2 model** (ref model = Stage-2 itself; memory-cheap, revertible); β_DPO swept {500, 2000, 5000}; Wallace et al. per-timestep loss-difference estimator. Anti-hacking: metric-diverse scoring, rejection floor, and a 50-sample human spot-check before accepting the stage.

*Note:* JEDi is distributional (MMD over sets) — it belongs to evaluation (§3), **not** to per-pair scoring.

## 3. Evaluation program (C10)

- **Baselines:** vanilla Wan2.2-TI2V-5B, AniSora-V3.2 (open weights), Native FM v1 (internal, LoRA), optional CogVideoX-I2V.
- **Benchmarks:** (a) frozen held-out GT benchmark under `data/benchmarks/` (series-disjoint; regenerated once on the new corpus and never touched); (b) **AniSora 948-clip benchmark**; (c) distributional JEDi (V-JEPA features, MMD) on generated-vs-real sets.
- **Human study (metric validation = contribution):** AniMatrix's 5-dimension protocol (Style Fidelity, Prompt Understanding, Artistic Motion, Structural Stability, Anime Aesthetic), 1–5 scale, 100–200 prompts × 3 raters, Krippendorff's α reported; correlate our automated suite against human scores; reproduce the FVD anti-correlation quantitatively. **Open logistics (user-owned): rater recruitment.**
- **Ablation grid** (from the Stage-2 checkpoint, one axis at a time): anchor-mode mix; L_Δ form {off, v1-legacy, v-space, (1−σ)^κ}; timestep density grid; curriculum on/off; motion weighting on/off; DPO on/off; C4; C8.
- **Multi-anchor eval:** first-last and storyboard-conditioned generation reported on the GT benchmark (anchors drawn from the GT clip), demonstrating C1's task generality.

## 4. Implementation architecture

New modules (pure-math cores TDD-able; entrypoints follow existing repo conventions):

```
src/native_animation/
  modeling/anchoring.py      # anchor-mode sampling, clamp/mask application, Ā-loss masks
  modeling/objectives.py     # v2 losses (motion-weighted vel, v-space delta); v1 kept as ablation path
  modeling/timesteps.py      # LogitNormal+shift density, per-sample sampling, 5% high-σ floor
  training/curriculum.py     # difficulty bucketing, sigmoid schedule, weighted sampler
  training/stages/{ct,sft,dpo}.py   # stage entrypoints (thin; config-driven)
  data/profile_motion.py     # flow energy + non-rigid residual + quantiles (GPU batch job)
  data/annotate.py           # Qwen3-VL structured captioning + three-section rewriting (local)
  evaluation/jedi.py         # JEDi wrapper; benchmark runners for GT-suite and AniSora-948
configs/*.yaml               # per-stage configs (pays the config-system debt); accelerate/FSDP configs
```

**Smoke gates before any real run:** (1) 5B full-FT memory/throughput on one 8×80GB allocation; (2) VAE keyframe-fidelity check at 480×832 (encode-decode line-art degradation measurement under the 16× spatial VAE); (3) end-to-end tiny-run of each stage entrypoint on 100 clips. Tests: every pure-math component (anchor masks, delta-loss σ-behavior on synthetic errors, density histograms, curriculum weights, pair construction) plus the existing suite stays green.

## 5. Campaign timeline (Sep → mid-Nov)

- **Sep W1–2:** Stage-0 pipeline complete on merged corpus; v2 modules + tests; smoke gates.
- **Sep W3–4:** CT-lite (both sub-stages); Stage-2 SFT first pass.
- **Oct W1–2:** ablation grid (parallel single-GPU + short multi-GPU jobs); Stage-3 DPO.
- **Oct W3–4:** human study, AniSora-948 + JEDi sweeps, optional AniSora-init second arm.
- **Nov W1–2:** paper (CVPR author kit in `paper/`), figures from `experiments/`.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Full-FT instability on stylized data | EMA + warmup + grad clip; fallback high-rank LoRA; curriculum limits early exposure to extreme deformation (AniMatrix's collapse finding) |
| DPO reward hacking of our own metric | metric-diverse ensemble, rejection floor, gap weighting, 50-sample human spot-check gate |
| Node availability crunch | everything checkpoint-resumable; ablations sized to single GPUs; multi-GPU reserved for CT/SFT |
| VAE 16× spatial hurts line art | smoke gate (2); if failed, evaluate 720p training arm or report as backbone limitation |
| AniSora variant incompatibility (second arm) | second arm is optional by design; verify variant before committing |
| Their promised AniMatrix release lands pre-deadline | monitor monthly; benchmark on their release if it appears |

## 7. Out of scope (deliberate)

Tag-channel architecture (C5) and dual CFG; few-step distillation (DMD/CausVid lane); long-video/autoregressive extension; audio; AniMatrix-style human-expert data tiering beyond community-score tiers.

## 8. Open items

1. Rater recruitment for the human study (user).
2. AniSora-V3.2 backbone variant + license-compat check (plan task).
3. AnimeReward weights availability check (plan task).
4. Exact S/A/B score cutoffs (set from merged-corpus statistics).
