# Native FM v2 — Technical Foundations (Intake for the Method Overhaul)

**Date:** 2026-08-28 · **Status:** intake complete; design in progress
**Objective (unchanged):** keyframe → short native-animation clip, on Sakugabooru-derived data.
**Purpose of this doc:** everything absorbed from the AniMatrix report (model/pretraining/math) and the recent A* literature, organized as a menu of grounded v2 components. The design spec selects from this menu; nothing here is yet a commitment. Companion: `related/animatrix.md` (strategic read + data plan).

---

## 1. AniMatrix method internals we may adapt (math-level)

Extracted from arXiv:2605.03652 (full derivations in the paper's §4–§5; numbers verified against the report).

**1.1 Tag encoder (TabTransformer-style).** Tag `(f_i, v_i)` → `e_i = W^field_{f_i} + W^value_{v_i}` (additive field/value decomposition preserves the Cartesian axis structure); `[CLS] + tags` through a 3-layer pre-LN transformer → per-tag sequence `h^tag_seq` and global `h^tag_CLS`.

**1.2 Dual-path injection.** Path 1: concat `[W^proj h^text_seq ; h^tag_seq]` into existing cross-attention, with learnable type embeddings `τ_text, τ_tag` added per source. Path 2: per-block AdaLN — `c_ℓ = SiLU(W^t_ℓ t_emb + W^g_ℓ h^tag_CLS)`; per-sub-layer `γ = W^γ c + 1, β = W^β c` (identity-init). **Scale problem for us:** full-rank per-sub-layer `W^γ, W^β ∈ R^{d×d}` over all blocks is ~10⁹ params on a 5B DiT — they train the full backbone; we cannot. PEFT adaptation (a real contribution surface): low-rank factorized modulation `γ = 1 + U_ℓ(V_ℓ c)` with `U ∈ R^{d×r}, V ∈ R^{r×d}`, or diagonal/vector modulation `γ = 1 + u_ℓ ⊙ (Wc)` — FiLM-grounded, trainable beside LoRA.

**1.3 Robustness + dual CFG.** Conditioning-mode dropout Categorical(hybrid .7, tag .1, text .1, ∅ .1); partial tag drop p=.15; synonym substitution p=.10; controlled tag–text conflict p=.05 with tag-authoritative targets. Inference (Composable-Diffusion-style marginal decomposition):
`ε̂ = ε_∅ + ω_text(ε_text − ε_∅) + ω_tag(ε_tag+text − ε_text)`, independent scales (they deploy 5.0 / 2.0).

**1.4 Curriculum.** Per-clip difficulty buckets `b(x) = (q_style, q_motion, q_deform)` (quantiles of style-cluster diversity, optical-flow energy, non-rigid flow residual); sampling weight `w_τ(b) = σ(γ_cur · (τ − D(b) + β_cur))` with `D(b)` = mean normalized bucket index; multiplied by the long-tail rebalancing weight `w_i = (Π_axes 1/n)^{0.7}`. Their claim: bridging physics→extreme-deformation *in one step collapses training*; the curriculum is the fix.

**1.5 Preference stage.** Diffusion-DPO (Wallace et al.) with the per-timestep approximation
`log π_θ(y|p)/π_ref(y|p) ≈ −½ E_{t,ε}[‖ε − ε_θ(z_t,t,p)‖² − ‖ε − ε_ref(z_t,t,p)‖²]`
inside `L_DPO = −E log σ(β_DPO(·_w − ·_l))`. Pairs: N=4 candidates/prompt, judge-scored all-pairs with min-head rejection (<2.0), expert annotation only for small-gap (~30%). Judge: video encoder + 4 linear heads (face/limb/line/motion), trained on ~20K expert-labeled clips.

**1.6 Stage recipe.** CT (6M clips, 256px/16f → 720px/65f, T2I:T2V:I2V 0.5:0.3:0.2 → 0.2:0.4:0.4) → SFT (A/S-tier + curriculum + conditioning dropout) → QT (S-tier only, lr 5e-5) → DPO. Data volume ↓, quality + specificity ↑ per stage.

**1.7 I2V + few-step (context).** I2V by temporal concat / first-latent fixing (same family as ours). DMD distillation (2 noise-experts × 4 steps, flow shift s=10, normalized real−fake gradient, CFG distilled) — out of scope for our core; noted for future efficiency work.

---

## 2. Literature map (A* + strong 2025–26 arXiv)

### 2.1 Anime vertical — the direct neighborhood
- **Index-AniSora** (Bilibili, IJCAI'25; V1 CogVideoX-5B → V2 Wan2.1-14B → V3/V3.1/V3.2 on **Wan2.2**, AnyMask spatiotemporal-mask I2V; **Apache-2.0 open weights**; ships a **948-clip anime benchmark** with action labels + corrected prompts). Consequences: (a) mandatory baseline; (b) candidate *initialization* — an anime-continue-trained Wan2.2 for free; (c) their benchmark complements ours for eval.
- **AnimeReward + GAPO** (arXiv:2504.10044, same group): 30k-sample multi-dimensional anime reward dataset; Gap-Aware Preference Optimization (preference gap magnitude enters the objective). If the reward model is public, it slots into our preference stage as a second scorer.
- **AniMatrix** (above). **ToonCrafter** (interpolation), **ToonComposer** (generative post-keyframing: sparse sketch injection + reference attention), **AniDoc** (trajectory-conditioned lineart colorization), **LVCD** (lineart video colorization) — production-pipeline lane (sketch/lineart-conditioned), related-work but distinct task from free keyframe continuation.
- **Sakuga-42M** (withdrawn) — data-provenance precedent, already handled in our data plan.

### 2.2 Motion-centric objectives (closest to our contribution lane)
- **VideoJAM** (Meta, ICML 2025): extend the prediction target to appearance *and* motion from one representation (aux flow prediction head), + **Inner-Guidance** at inference (model's own motion prediction steers sampling). "Any video model, minimal adaptation" — the strongest published cousin of our motion-aware objective; must cite, can adapt, and our delta-consistency remains distinct (GT-latent-difference regression, no motion encoder needed).
- **Track4Gen** (CVPR'25): point-tracking supervision inside video gen; **MoAlign** (2510.19022): motion-centric representation alignment. Heavier supervision routes; ablation candidates, not core.
- **Go-with-the-Flow**-style noise warping (optical-flow-correlated noise): training-cheap motion control; optional.

### 2.3 Representation alignment
- **REPA** (ICLR'25): align DiT hidden states to a frozen SSL encoder → large convergence gains. **VideoREPA** (NeurIPS'25): Token Relation Distillation (pairwise spatial-temporal token relations) to transfer *physics* into finetuned video DiTs; **CREPA**: cross-frame alignment.
- **The anime tension (design decision, not free lunch):** VideoREPA transfers a *physics* prior — the exact prior we're trying to soften. Options: (i) skip; (ii) align only spatial/appearance relations (line-art stability) not temporal ones; (iii) align to an anime-finetuned or style-focused encoder. If used, it must be argued and ablated as "style-preserving alignment," not imported blindly.

### 2.4 Objective/timestep grounding
- **SD3 / rectified-flow practice**: logit-normal timestep density (concentrates mid-σ) + shift as the standard knobs; video systems sweep the training shift (reported sweeps land ~3, e.g. α=2.95 by FVD/FID) — *independent support for our empirical shift=3*, now expressible as a density choice rather than a hack. Low-SNR augmentation trick: ~5% of timesteps sampled uniformly on σ∈[0.95,1].
- **Curriculum Sampling for FM** (2603.12517), **Gradual Fine-Tuning for FM** (2601.22495): recent grounding for staged/curriculum FM fine-tuning (read before the spec if timestep-curriculum becomes a claim).
- **Our v1 delta-loss σ-issue (derived, must fix in v2):** with `x̂₀ = z_σ − σ·v̂` and `v̂ = v* + e`, the x₀-error is `−σe`, so the delta-consistency residual scales as `σ·(e_t − e_{t−1})` and the loss term as **σ²** — the regularizer is dominated by high-noise timesteps exactly where x̂₀ is least informative. Fix candidates: divide by `max(σ, σ_min)²` (equalize), weight by `(1−σ)` (emphasize low-noise fidelity), or regress deltas in v-space. Choose via a small sweep; derivation goes in the paper appendix.
- Other v1 audit items: motion weighting is a valid importance-reweighting (normalizer keeps E[w]·mean scale ≈ unweighted; α=0 exact recovery — already tested); one shared timestep per batch (DiffSynth default) adds gradient variance — per-sample timesteps are a free improvement to validate.

### 2.5 Conditioning / task shape
- **Wan2.2-TI2V-5B mechanism confirmed:** high-compression ST-VAE (4×16×16), I2V = *first temporal latent fixed*; "fixing multiple frames turns I2V into V2V." Our anchor-clamp training matches the base mechanism — and generalizes: **anchor set A ⊆ {0..T'}, clamp clean + exclude from loss** = conditional FM on p(x | x_A). Single keyframe, first+last, and sparse storyboard keyframes are all the same math. AniSora-AnyMask and the keyframe-interpolation lane (Generative Inbetweening SIGGRAPH-Asia'24, Framer, MotionBridge, ICLR'25 keyframe-interp) demonstrate demand; AniMatrix's admitted gap #1 (text-only conditioning, no storyboards) says the giant hasn't covered it.

### 2.6 Preference / RL
- **Diffusion-DPO** (math in §1.5) — offline, stable, cheap: primary candidate. **VideoDPO** (omni-preference), **Flow-DPO** (multi-dim reward), **DanceGRPO** / **Flow-GRPO** (GRPO on rectified flows via ODE→SDE; on-policy, costlier) — alternatives if we want online RL. **GAPO** (anime-specific, gap-aware) — refinement of pair weighting.
- **Our distinctive angle stands:** keyframe-continuation has a *ground-truth continuation*; scoring candidates against GT with our evaluator builds preference pairs with no human annotators and no learned judge. Guard against reward hacking: metric-diverse scoring + min-threshold rejection (their trick) + human spot-check.

### 2.7 Evaluation
- **FVD content bias** (CVPR'24) — I3D features underweight temporal quality; **JEDi** (ICLR'25 "Beyond FVD"): V-JEPA features + MMD (polynomial kernel), 34% better human alignment, far fewer samples needed — adopt as our distributional metric alongside the GT-anchored suite. **VBench/VBench-2.0** per-dimension probes; **AniSora's 948-clip benchmark**; **AniMatrix's 5-dim human protocol** (replicate small-scale for metric validation).

### 2.8 Efficiency / long video — explicitly out of core scope
CausVid, Self-Forcing(++/Rolling/Causal/Context-Forcing, ICML'26 line), FramePack, DMD2/SwiftVideo: crowded distillation/AR lane; future work only. Pyramid-Flow/Flowception (now arXiv:2512.11438) — temporal-expansion references, archived clones on hand.

---

## 3. Candidate v2 component menu (select in the design spec)

| ID | Component | Core math | Grounding | Cost | Risk |
|----|-----------|-----------|-----------|------|------|
| C1 | Generalized temporal anchoring (multi-keyframe/storyboard) | clamp+exclude on anchor set A; conditional FM p(x\|x_A); anchor-count curriculum | Wan TI2V mechanism; AnyMask; inbetweening lane; AniMatrix gap #1 | low (loss-level) | eval design for multi-anchor |
| C2 | σ-corrected delta consistency | fix σ² scaling (§2.4); sweep {1/σ², (1−σ), v-space} | our derivation; SNR-weighting practice | low | none — strictly better-posed |
| C3 | Explicit timestep density | logit-normal(m,s)+shift replaces bare shift; +5% high-σ uniform | SD3; video shift sweeps; FM curriculum papers | low | sweep compute |
| C4 | Motion-prediction auxiliary (VideoJAM-lite) | aux head predicts latent flow/Δ; optional Inner-Guidance | VideoJAM ICML'25 | med | head capacity at LoRA scale |
| C5 | Tag-channel conditioning (PEFT) | §1.1–1.3 with low-rank AdaLN; dual CFG | AniMatrix; TabTransformer; FiLM | med-high | needs VLM tags first; param budget |
| C6 | Macro curriculum + rebalancing | §1.4 verbatim at our scale | AniMatrix; curriculum-FM | low-med | schedule tuning |
| C7 | GT-anchored preference stage | §1.5 objective; pairs from GT-scored candidates; GAPO-style gap weighting | Diffusion-DPO; VideoDPO; GAPO | med | reward hacking (mitigations §2.6) |
| C8 | Style-preserving representation alignment | TRD on spatial relations only / anime encoder | REPA/VideoREPA/CREPA | med | physics-prior contamination (§2.3) |
| C9 | AniSora-V3 initialization | swap init checkpoint; keep our objective stack | open Apache-2.0 weights on Wan2.2 | low | narrative: "built on anime CT"; verify variant/size compat with TI2V-5B |
| C10 | Eval program v2 | JEDi + GT-suite + AniSora-948 + 5-dim human study | §2.7 | med (human time) | none |

**Coherence note:** C1–C3 upgrade the *objective* (our lane, cheap, mathematically concrete). C6–C7 are the *training-program* adoptions. C4/C5/C8 are the expansion candidates where scope discipline is needed. C9–C10 are infrastructure choices.

## 4. Open design questions (to settle before the spec)

1. **Compute envelope** — concurrent GPUs (how many gmem48 vs gmem80 nodes realistically holdable), total GPU-days budget to November. Bounds: full-FT vs LoRA, 5B vs larger, curriculum length, DPO sampling budget.
2. **Initialization** — vanilla Wan2.2-TI2V-5B vs AniSora-V3 (and whether V3's backbone variant is compatible or forces a backbone change).
3. **Scope of conditioning overhaul** — objective-only v2 (C1–C3+C6–C7) vs including the tag channel (C5), which gates on VLM annotation of the new 180k corpus.
4. **Multi-keyframe (C1) in or out of the core claim** — it widens the task story (storyboard→animation) but adds eval surface.
5. **Backbone freeze policy** — LoRA rank / which modules / possibly partial-unfreeze of temporal layers under 80GB nodes.
