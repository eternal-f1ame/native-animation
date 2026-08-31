# Native Animation

Keyframe-conditioned native-animation video generation — teaching video models the motion language of anime. Flow matching is the substrate, not the subject.

Given a single anime keyframe, the model generates a short continuation that preserves the frame's artistic style while producing the stylized, physics-violating motion — smears, impact frames, morphing — that defines high-quality 2D animation. Off-the-shelf video models inherit a photorealism prior from web-scale training data and reliably erase this motion (the *realism trap*); this project attacks that failure directly. The method write-up lives in `docs/method.md`; `paper/` will hold the CVPR 2027 manuscript.

## Method

The method (v2) trains a Wan2.2-TI2V-5B backbone — vanilla weights, our own continued training — with a project-owned objective spine; the architecture is untouched:

1. **Anchored conditional flow matching** — a sampled anchor set (keyframe / first+last / storyboard / none) is clamped clean and run at t=0 through TI2V's separated-timestep path; the flow-matching loss covers only non-anchor frames. The clamp *is* the anchor's supervision.
2. **σ-corrected temporal delta in v-space** — frame-to-frame consistency regressed where the x̂₀-space version provably degenerates (its error scales as −σe), with anchor substitution inside the delta.
3. **Motion-aware frame weighting** — mean-preserving weights from normalized latent deltas concentrate capacity on motion beats over the static stretches that dominate sakuga clips.
4. **Timestep density + curriculum** — a shifted logit-normal σ-sampler with a reserved 5% high-noise tail, and a difficulty/rebalance curriculum over the corpus tiers.
5. **GT-anchored DPO** (post-SFT) — preference pairs anchored on ground-truth frames.

Modules live in `src/native_animation/modeling/` and `src/native_animation/training/`; anchored inference in `src/native_animation/inference/anchored.py`. The course-era v1 objective (`modeling/native_flowmatch.py`, shift-3 + motion weighting + x̂₀ delta) is retained as a baseline/ablation arm. Long-form: `docs/method.md`, `docs/method-v2-foundations.md`; positioning vs. concurrent work: `docs/related/animatrix.md`.

## Evaluation

`src/native_animation/evaluation/evaluate.py` scores generations against held-out clips on four axes — CFS (continuation fidelity), TCS (temporal consistency), WorstSegment, and DFS (diffusion-failure score) — aggregated as `FinalScore = 0.4·CFS + 0.25·TCS + 0.2·WorstSeg − 0.5·DFS`.

## Dataset

A Sakugabooru corpus at two scales: the curated v1 set (~11.9k clips) and the full 2025 snapshot + delta scrape (~150k posts / ~2.2T, on track for ~500k curated single-shot clips with Qwen3-VL captions). The Stage-0 pipeline (`tools/`: streaming tar processing, shot splitting, motion profiling, annotation, squash consolidation) is object-quota-lean by design — shots live in per-batch pack tars consolidated into a single `shots.sqsh`. Layout, stats, and the storage tiers: `docs/dataset.md`. Raw data is never versioned or redistributed (release = post IDs + metadata + scripts); it lives at the datasets-area path wired in `configs/paths.env`.

## Repository layout

| Path | Contents |
|---|---|
| `src/native_animation/` | Project-owned code: `data/`, `modeling/`, `training/`, `inference/`, `evaluation/` |
| `src/diffsynth/` | Vendored DiffSynth runtime subset (not the contribution — see `THIRD_PARTY.md`) |
| `tools/` | Sakugabooru scraper and keyframe-pair dataset builder |
| `configs/paths.env` | Single source of truth for machine paths |
| `scripts/`, `scripts/slurm/` | Cluster entrypoints |
| `tests/` | CPU-only unit tests (`pytest`) |
| `paper/` | CVPR 2027 manuscript (in progress) |
| `docs/` | Method notes, dataset notes, research roadmap |
| `experiments/` | Run outputs (gitignored) |

## Workflow

```bash
# 0. one-time: paths (defaults assume the standard workspace layout; override via env)
source configs/paths.env

# 1. Stage-0 data pipeline (chain with --dependency=afterany:<id>; details in CLAUDE.md)
KEEP_TAR=1 sbatch scripts/slurm/stream_process.sbatch
sbatch scripts/slurm/split_shots.sbatch
sbatch scripts/slurm/profile_motion.sbatch
sbatch scripts/slurm/annotate.sbatch
sbatch scripts/slurm/consolidate_squash.sbatch

# 2. smoke gates on a GPU node
sbatch scripts/slurm/smoke_memory.sbatch
sbatch scripts/slurm/smoke_vae_fidelity.sbatch

# 3. v2 training stages (config-driven: ct_a -> ct_b -> sft)
STAGE_CONFIG=configs/ct_a.yaml sbatch scripts/slurm/train_v2.sbatch

# 4. v1 baseline arm (legacy, kept for ablations)
sbatch scripts/slurm/train_native_animation.sbatch
```

## Testing

```bash
python -m pytest
```

## License

Apache-2.0. `src/diffsynth/` provenance in `THIRD_PARTY.md`.
