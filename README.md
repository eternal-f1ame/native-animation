# Native Animation

Keyframe-conditioned native-animation video generation with Flow Matching.

Given a single anime keyframe, the model generates a short continuation that preserves the frame's artistic style while producing the stylized, physics-violating motion — smears, impact frames, morphing — that defines high-quality 2D animation. Off-the-shelf video models inherit a photorealism prior from web-scale training data and reliably erase this motion (the *realism trap*); this project attacks that failure directly. See `paper/` for the current write-up.

## Method

Native Animation Flow Matching (Native FM) extends a Wan2.2-TI2V Flow Matching backbone with three coordinated changes, all in `src/native_animation/modeling/native_flowmatch.py`. The architecture is untouched; the contribution is the objective and the noise schedule:

1. **Keyframe-preserving scheduler shift** — `NativeAnimationFlowMatchScheduler` defaults to `shift=3.0` (vs. Wan's ~5), so early timesteps stay closer to the clean signal and the conditioning keyframe survives noising.
2. **Motion-aware frame weighting** — per-frame loss weights derived from latent frame-to-frame deltas concentrate capacity on motion beats rather than the long static stretches that dominate sakuga clips.
3. **Latent temporal-difference consistency** — the predicted clean sequence's frame deltas are regressed onto the ground-truth deltas, penalizing flicker and mid-clip collapse that velocity-only losses ignore.

The keyframe latents are clamped clean during noising and excluded from the loss so the anchor stays pinned. Details: `docs/method.md`.

## Evaluation

`src/native_animation/evaluation/evaluate.py` scores generations against held-out clips on four axes — CFS (continuation fidelity), TCS (temporal consistency), WorstSegment, and DFS (diffusion-failure score) — aggregated as `FinalScore = 0.4·CFS + 0.25·TCS + 0.2·WorstSeg − 0.5·DFS`.

## Dataset

A curated Sakugabooru corpus (~11.9k clips, 240+ series, 25 technique tags). The scraping and windowing pipeline lives in `tools/`; layout and stats in `docs/dataset.md`. Raw data is not versioned — it lives at `<workspace>/data/sakugabooru/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/native_animation/` | Project-owned code: `data/`, `modeling/`, `training/`, `inference/`, `evaluation/` |
| `src/diffsynth/` | Vendored DiffSynth runtime subset (not the contribution — see `THIRD_PARTY.md`) |
| `tools/` | Sakugabooru scraper and keyframe-pair dataset builder |
| `configs/paths.env` | Single source of truth for machine paths |
| `scripts/`, `scripts/slurm/` | Cluster entrypoints |
| `tests/` | CPU-only unit tests (`pytest`) |
| `paper/` | The paper (LaTeX) |
| `docs/` | Method notes, dataset notes, research roadmap |
| `experiments/` | Run outputs (gitignored) |

## Workflow

```bash
# 0. one-time: paths (defaults assume the standard workspace layout; override via env)
source configs/paths.env

# 1. build metadata CSVs from the raw clips
sbatch scripts/slurm/build_metadata.sbatch

# 2. GPU-node environment check
sbatch scripts/slurm/env_smoke_test.sbatch

# 3. untuned-baseline generations on held-out clips
sbatch scripts/slurm/base_inference_demo.sbatch

# 4. fine-tune Native FM (hyperparameters override via env, e.g. NUM_FRAMES=81)
sbatch scripts/slurm/train_native_animation.sbatch

# 5. generate from one keyframe with a trained LoRA
python -m native_animation.inference.generate \
  --input-image keyframe.png --prompt "native animation, anime, ..." \
  --lora-path experiments/checkpoints/native_animation_flowmatch_lora/... --output out.mp4

# 6. evaluate
python -m native_animation.evaluation.evaluate \
  --summary-json experiments/demo/base/summary.json --dataset-base-path "$DATA_ROOT"
```

## Testing

```bash
python -m pytest
```

## License

Apache-2.0. `src/diffsynth/` provenance in `THIRD_PARTY.md`.
