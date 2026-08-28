# AniMatrix (Tencent, arXiv:2605.03652) — Full Analysis and Extraction Plan

**Paper:** "AniMatrix: An Anime Video Generation Model that Thinks in Art, Not Physics" — Tencent (Workrally / Tencent Video), arXiv 2605.03652, May 2026 (v2).
**Status of their release:** model, data, and benchmark are **not public** ("preparing accompanying resources for public release"). Deployed commercially (Workrally, 60+ studios, 57 s/clip on 8×H20).
**Read:** 2026-08-28, full text + all appendices. This document is the durable extraction; cite the arXiv version for anything quoted.

## 1. What they built (numbers that matter)

- **Thesis = ours, formalized.** Physics-trained video models "flatten the artistry" of anime; they name the same failure we call the *realism trap* and make "artistic correctness" the optimization target. Validated by professional deployment — the strongest possible evidence our problem framing is real and industrially valued.
- **Production Knowledge System (PKS):** a 4-axis Industrial Production Taxonomy 𝒯 = Style × Motion × Camera × VFX (~80+ tags, full vocabularies printed in their Appendix A.1, Tables 5–11, incl. per-tag VFX metadata: meaning / appearance / placement+dynamics / scenes).
- **AniCaption:** Qwen3-VL fine-tuned (expert sub-models → CT on 16M clips → SFT on 500K human-corrected → DPO on motion+VFX) to infer taxonomy coordinates from pixels and emit (a) a six-group structured JSON caption and (b) a three-section directive `<tag>/<summary>/<description>`. Beats Gemini 2.5 Pro on their held-out set (motion failure rate 15.4% vs 61.6%).
- **Data:** 150M raw clips → 16M technically sound (general operators) → 6M B-tier (5 anime-specific operators) → 1M A-tier (expert review, >90% agreement) → 500K S-tier. Long-tail rebalancing weight `w_i = (1/(n_s·n_m·n_c·n_v))^α`, α=0.7; Motion-axis Gini 0.71→0.38.
- **Architecture:** Wan 2.2 14B MoE DiT + Causal 3D VAE, frozen umT5-XXL text channel, plus a **trainable tag encoder** (field-embedding + value-embedding additive decomposition, 3-layer transformer, [CLS]) injected via **two paths**: tag+text token concat into cross-attention (with learnable type embeddings) and the tag [CLS] via **AdaLN modulation** at every sub-layer (γ init 1, β init 0). Rationale for not just prepending tags to text: subword fragmentation, loss of field–value structure, frozen-encoder conflict.
- **Robustness:** stochastic conditioning dropout (hybrid/tag/text/∅ = 0.7/0.1/0.1/0.1) enabling **dual CFG** `ε̂ = ε∅ + ω_text(ε_text−ε∅) + ω_tag(ε_tag+text−ε_text)` (deploy ω_text=5.0, ω_tag=2.0); partial tag drop p=0.15; synonym augmentation p=0.10; **controlled tag–text conflict training** p=0.05 with tag-authoritative ground truth (tags = hard constraints, text = soft guidance).
- **Training: 4 stages.** CT (6M B-tier, 256px/16f → 720px/65f, T2I:T2V:I2V mix 0.5:0.3:0.2 → 0.2:0.4:0.4); SFT with a **style–motion–deformation curriculum** — per-clip difficulty buckets over style cluster k(x), motion amplitude m(x) (optical-flow energy), deformation intensity d(x) (non-rigid flow residuals), sampled by sigmoid schedule `w_τ(b) = σ(γ·(τ − D(b) + β))`, multiplied by the rebalancing weight; QT on S-tier only at lr 5e-5; **deformation-aware DPO** — a 4-head Judge (face topology, limb structure, line continuity, motion coherence; Wan-initialized encoder, trained on ~20K expert-annotated clips), N=4 candidates/prompt, auto-pairs by composite score with min-head threshold 2.0, expert annotation for small-gap ~30%, ~50K pairs, Diffusion-DPO (Wallace et al.) with the per-timestep denoising-loss-difference likelihood approximation.
- **Evaluation:** rejects FVD/CLIP ("**anti-correlate** with human quality judgments on anime — exaggeration is penalized, static physics-plausible output rewarded"). Human protocol: 500 taxonomy-covering I2V prompts, 15 professional evaluators, 3 raters/prompt, 5 dimensions (Style Fidelity / Prompt Understanding / Artistic Motion / Structural Stability / Anime Aesthetic), 1–5 scale, Krippendorff's α > 0.72. Results: first on 4/5 vs Seedance-Pro 1.0 and Wan2.2 — biggest wins Prompt Understanding +22.4%, Artistic Motion +16.9%; Structural Stability at parity.
- **Deployment:** DMD distillation 40→8 steps (two Wan MoE noise experts × 4-step students; flow shift s=10 biasing high noise; dual-CFG distilled into weights) → 10× wall-clock; the 8-step student *beats* the teacher on Structural Stability (+0.13) by regularizing rare deformity outputs.
- **Admitted gaps (their Conclusion, verbatim themes):** (1) conditioning is text-only (no character sheets/storyboards/audio); (2) "**artistic motion timing and effect rendering are not first-class conditioning axes; the model still inherits a uniform-motion prior** from its physics-pretrained backbone, which dampens the non-uniform rhythms"; (3) no test-time directorial planning. AniMatrix-Uni announced to address these.

## 2. Strategic read for us

**What it changes: nothing in our framing.** (Decided 2026-08-28.) AniMatrix is a non-peer-reviewed technical report with nothing public except the PDF — no model, data, or benchmark. Two groups independently converging on the same thesis is ordinary science and *strengthens* our motivation: the realism trap / artistic-correctness framing remains ours to state in our own terms, with AniMatrix cited as concurrent industrial work that corroborates it at deployment scale. We do not write defensively around it.

**Where the work is complementary — our lanes:**

1. **The objective level is untouched.** AniMatrix attacks artistic correctness through *data + conditioning + curriculum + preference alignment* around a **standard FM velocity loss**. Native FM attacks it through the **objective itself** (motion-aware frame weighting, latent temporal-delta consistency, keyframe anchoring). Nothing in their recipe conflicts with ours; nothing in theirs subsumes ours. Better: their gap (2) — the surviving *uniform-motion prior* damping non-uniform anime rhythm — is precisely what per-frame motion weighting and delta consistency target, without new conditioning axes, annotation systems, or 64×H800. Our pitch: *artistic correctness for the GPU-poor, at the objective level* — quote their own limitation as the problem statement.
2. **Automated evaluation is an explicitly open problem.** They rely on 15 professional annotators and state: "Developing automated metrics that align with professional anime quality judgments remains an important direction for future work." Our CFS/TCS/WorstSegment/DFS suite — *validated against a small human study using their 5-dimension protocol* — directly answers that call. Their observed FVD anti-correlation is a cheap, high-value experiment to reproduce and quantify.
3. **Nothing of theirs is public.** No weights, no data, no benchmark. A released, reproducible benchmark + tooling on Sakugabooru-derived data (post IDs + metadata + scripts, not clips) has no competitor from them today. Watch item: they promise a release — monitor arXiv/GitHub monthly; if their benchmark lands before November, evaluate on it.
4. **Task validation.** They distill I2V *because* "anime production starts almost always from a reference frame." Our keyframe-to-video task is the industrially correct setting — quotable.

**What NOT to chase (scale traps):** training a captioner (AniCaption = 16M clips + 500K human corrections), full continue-training of a 14B MoE, a human-annotated Judge reward model, 15-annotator evaluation, DMD deployment engineering. None are winnable or necessary at our scale.

## 3. Adoption plan, ranked

### Tier 1 — cheap, do before the next training run

| # | Item | What we do | Their source | Effort |
|---|---|---|---|---|
| 1 | **Shot splitting** | Split our 11.9k clips (5–60 s, multi-shot) into single-shot 2–10 s segments with PySceneDetect + TransNetV2. Our 49-frame training windows currently cross cuts — contaminated motion supervision. Bonus: turns 11.9k clips into an estimated 50–100k single-shot clips from data we already have. | A.3.2 | days |
| 2 | **VLM structured annotation** | Run an off-the-shelf VLM (Qwen3-VL / Claude) over our clips with **their exact JSON schema** (six groups: subjects, motion, AnimeVisualEffects w/ 3-level type hierarchy, style, camera, environment) + **their three-section rewriting prompt** (`<tag>/<summary>/<description>`, ordering: subject → camera → motion → VFX → environment). Replaces our weak prompts (comma-joined community tags incl. artist names and "presumed"). 11.9k–100k clips is directly affordable with a strong VLM; no captioner training needed. | §3.4, A.2.1–A.2.3 | ~1 wk + API cost |
| 3 | **Taxonomy mapping** | Map Sakugabooru tags onto 𝒯: smears/morphing/impact_frames → Animation Techniques + Action&Motion Effects; liquid/smoke/fire/explosions/lightning/debris/sparks → VFX categories; fighting/running/character_acting → Motion type; series → Style proxy. VLM pass (item 2) fills the axes we lack entirely: **camera, style, emotion, amplitude/speed**. Their full vocabularies are printed in Tables 5–11 — adopt them wholesale as our label space. | A.1 | days (with #2) |
| 4 | **Long-tail rebalancing** | Implement `w_i = (Π 1/n_axis)^0.7` over our metadata (our head is one_piece=1123 rows, `_other`=4810) as a sampler weight; break style–content shortcuts with minimum-representation thresholds. | §3.6 Eq. 2 | ~1 day |
| 5 | **Curation mini-cascade** | Static-clip removal, OCR subtitle filtering, near-dup removal (embedding cosine >0.95, keep highest-res), Laplacian blur w/ anime-calibrated threshold. Run after shot splitting. | A.3.2 | days |

### Tier 2 — method-level, drives experiments

| # | Item | What we do | Effort |
|---|---|---|---|
| 6 | **Style–motion–deformation curriculum** | Their sigmoid schedule (Eq. 16) at our scale: m(x) = optical-flow energy (we already compute Farneback in the evaluator), d(x) = non-rigid residual (flow minus global affine fit), k(x) = style/series cluster; Q=4 quantile buckets; multiply by rebalancing weight. Complements (not replaces) our per-frame motion weighting: theirs is a per-*sample* macro-curriculum, ours a per-*frame* micro-curriculum → clean 2×2 ablation, and their collapse analysis predicts the curriculum matters for our high-deformation tail. | ~1 wk |
| 7 | **Human-validated automated metrics** | Replicate their 5-dimension protocol at small scale (100–200 prompts, 2–3 raters, report Krippendorff's α), correlate CFS/TCS/WorstSeg/DFS against the human scores, and reproduce the FVD anti-correlation finding quantitatively. This converts our evaluator from "our metrics" into a *validated benchmark contribution* answering their stated open problem. | ~2 wks incl. rating time |
| 8 | **GT-anchored preference optimization** (our twist on their Stage 4) | They need a trained Judge + expert pairs because T2V has no ground truth. Our keyframe-continuation setting **has the true continuation**: sample N=4 per test keyframe, score each against GT with our evaluator, auto-build DPO pairs (their recipe: all-pairs by score, min-score rejection threshold), train with Diffusion-DPO (Wallace et al. likelihood approximation — formula in §5.4). No annotators, no reward model. Risk: reward-hacking our own metric — mitigate with metric-diverse scoring + human spot-checks, and it becomes a *validation* of contribution #7. | ~2–3 wks |
| 9 | **Tag-conditioning lite** (stretch) | Their trainable tag encoder + AdaLN path is small (embedding tables + 3-layer transformer + per-layer projections) and could train alongside our LoRA on frozen Wan — "structured production conditioning under low-resource adaptation." Adopt their conditioning-dropout scheme (incl. tag-only CFG fallback) if we do. Only after Tier 1 data exists. | ~1 mo |

### Tier 3 — quotes and framing for the paper

- Realism trap ↔ their physics-prior argument: cite as convergent industrial validation of the problem.
- Their gap (2) (uniform-motion prior survives their pipeline) = our objective-level motivation.
- Their FVD/CLIP anti-correlation claim = motivation for our metric suite; our study quantifies it.
- I2V-first ("production starts from a reference frame") = task justification.
- Their Judge's 4 structural heads (face/limb/line/motion) ↔ our DFS: DFS is the automated, GT-anchored cousin; a learned lightweight judge distilled from VLM labels is a plausible future work note.
- Related work we must now cite and differentiate: AniSora (arXiv:2412.10255, spatiotemporal mask control), AnimeReward (2025, human-feedback alignment for anime), ToonCrafter (2024, keyframe interpolation), VideoDPO (2025), Diffusion-DPO (Wallace et al. 2024), DMD (Yin et al. 2024), TabTransformer (2020), PointOdyssey (2023, long-term tracking for motion profiling), TransNetV2 (shot detection), Unimatch (flow).

## 4. Contribution reframe for CVPR 2027 (proposal for the roadmap session)

1. **C1 — Native FM objective** (exists): motion-aware weighting + latent temporal-delta consistency + keyframe anchoring as a *backbone-agnostic, annotation-free, LoRA-scale* route to artistic motion — positioned directly against AniMatrix's data/conditioning-heavy route and their admitted uniform-motion-prior gap. Ablations: 2×2 with the sample-level curriculum (#6).
2. **C2 — Validated automated anime evaluation** (#7): the metric suite + human-correlation study + FVD anti-correlation reproduction + released benchmark tooling — the open problem they named, answered reproducibly.
3. **C3 — Low-resource data recipe** (#1–#5): shot-split + taxonomy-mapped VLM directives + rebalancing on a fully reproducible public-source corpus; quantify each step's contribution.
4. **Stretch — C4**: GT-anchored DPO (#8) and/or tag-conditioning lite (#9).

The one-line positioning: *Two groups independently arrived at artistic correctness as the target; AniMatrix demonstrates it with industrial data and compute behind closed doors — we establish it in the open, through the objective, a validated evaluation, and a fully reproducible recipe.*

## 5. Watch items

- Their promised resource release (benchmark would change C2's landscape; weights would give us a stronger baseline to compare against). Check arXiv/HF/GitHub monthly.
- AniMatrix-Uni (announced): multimodal conditioning, motion-rhythm axis, test-time planning. If it lands before November with a rhythm-conditioning mechanism, sharpen C1's distinction (objective vs conditioning).
- Seedance 2.0 / Wan 2.5 anime capability (mentioned as unreleased/unreported at their time of writing).

## 6. Data scaling: matching their acquisition posture at our scale (2026-08-28)

**Posture (user-decided):** identical to theirs — train on publicly available anime media, redistribute nothing. Sakugabooru is a public clip-sharing forum; acquisition for research carries no meaningful scrutiny. The one cautionary precedent, [Sakuga-42M](https://arxiv.org/abs/2405.07425v1) (U. Alberta, 2024, withdrawn), failed on *redistribution of media*, not scraping — our release policy (post IDs + metadata + scripts, never clips) stays on the surviving side of that line.

**Ground truth about our holdings (verified on disk):** 11,781 clips / 156 GB — and nothing else anywhere on the filesystem (no pair-dataset output, no second cache). The April scrape was *interrupted mid-sweep* (`scrape.log` ends at `[3436/5000]` of the smears sweep); we hold ~4% of the booru. The "3.6M pairs" figure from the course era was a windowing *projection over* these clips, not additional data.

**Acquisition plan (in flight):**

1. **Bulk via mirror:** [`trojblue/sakugabooru2025`](https://huggingface.co/datasets/trojblue/sakugabooru2025) on HF is a full media mirror — 155,238 video clips + 240,242 metadata JSONs, 1.27 TB, WebDataset tars, complete coverage post ID 0 → 2024-12-28 (163,918 of 273,264 post IDs have media; rest are DMCA'd/dupes). Downloading from the HF CDN is ~50× faster than polite scraping and puts zero load on the community site. Download launched 2026-08-28 into `data/sakugabooru/snapshot-2025/` (nohup, resumable via `snapshot_download`).
2. **Delta via our scraper, zero code changes:** the booru API is currently open without Cloudflare cookies; moebooru `id:A..B` range tags work; the scraper passes `--anime` values through as raw query tags and sorts output by the *post's* series tags. Validated dry-run: `tools/scrape_sakugabooru.py --anime "id:<cutoff>..<latest>" --limit 999999 --min-score 0`. Latest post ID at check time: 314,459; snapshot cutoff ≈ 273k → delta ≈ 41k posts ≈ ~25k clips with media.
3. **Merge:** extract snapshot tars into the `clips/<series>/{post_id}_s{score}.mp4 + {post_id}.json` layout (series derived from tags, as the scraper does), dedupe by post ID against existing holdings, then run the Tier-1 pipeline (shot split → curation cascade → VLM annotation → rebalancing) over the merged corpus.

**Resulting corpus estimate:** ~180k clips ≈ 1.6–1.8 TB — a ~15× expansion that lands at the *useful ceiling*: LoRA-scale training on this cluster can consume roughly 100–250k curated clips before the CVPR deadline, so their remaining 149M-clip scale would be wasted on us. Signal-density note: Sakugabooru is community-curated craft, i.e., we start near where their curation cascade *ends* (their 150M → 1M A-tier is a 0.7% survival rate).

**Storage:** 47 TB free on the share, no quota barrier observed.
