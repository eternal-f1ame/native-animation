# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`native-animation` — keyframe-conditioned native-animation video generation targeting CVPR 2027 — the method is about native animation (flow matching is just the substrate). A Wan2.2-TI2V-5B backbone is fine-tuned with a project-owned objective; the entire method contribution is `src/native_animation/modeling/native_flowmatch.py` (~170 lines): keyframe-preserving scheduler shift (3.0), anchor-frame clamping (frame 0 clamped clean and excluded from the loss — the clamp IS its supervision), motion-aware frame weighting (`w = 1 + α·normalized latent delta`), and a latent temporal-difference consistency term (`λ=0.25`). `docs/method.md` is the long-form write-up; `paper/` will hold the CVPR 2027 manuscript — all experiments are being redone fresh, no prior numbers carry over.

This repo expects the standard workspace around it: `../data/sakugabooru/{clips,metadata}`, `../models` (weight cache, reached via the repo's `models` symlink), `../third_party` (frozen DiffSynth fork + reference clones). `configs/paths.env` is the single source of truth for these paths — source it; never hardcode.

## Commands

```bash
# tests (CPU-only, no weights) — the standing verification gate
python -m pytest

# metadata build / GPU smoke test / baseline / training — all via SLURM
sbatch scripts/slurm/build_metadata.sbatch
sbatch scripts/slurm/env_smoke_test.sbatch
sbatch scripts/slurm/base_inference_demo.sbatch
sbatch scripts/slurm/train_native_animation.sbatch
# every hyperparameter is an env override, e.g.:
NUM_FRAMES=81 DELTA_LOSS_WEIGHT=0.5 sbatch scripts/slurm/train_native_animation.sbatch

# monitor
squeue --me
sacct -j <id> --format=JobID,JobName,State,ExitCode,Elapsed -n
tail -n 120 experiments/logs/<jobname>-<id>.{out,err}
```

## Cluster conventions (hard-won — do not relearn these)

- Conda env `comfy` (Python 3.11, torch 2.7.1+cu126). Activate it and export `PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` in the login shell BEFORE `sbatch` — jobs inherit via `#SBATCH --export=ALL`. Without the thread caps, heavy imports on the login node throw `MemoryError`.
- GPU jobs need `#SBATCH --constraint=gmem48|gmem80`; the cluster has 11–12 GB nodes that cannot hold the 5B pipeline.
- Never set GPU wall time under `1-00:00:00`: first-run Wan asset downloads take hours and resume from partials on retry.
- The login node cannot judge heavy imports — only `env_smoke_test.sbatch` on a GPU node counts as validation.
- ModelScope silently redirects `Wan-AI/Wan2.2-TI2V-5B` `.pth` files to `DiffSynth-Studio/Wan-Series-Converted-Safetensors` in the cache — expected, not an error.

## Gotchas

- Training needs `--data_file_keys video` and `--extra_inputs input_image` (the keyframe is the first decoded video frame; the CSV has no image column). Both are wired into the scripts — keep them.
- Gradient checkpointing is force-enabled in the training module; video FM activations OOM without it.
- Every entrypoint carries a `sys.path` bootstrap so it runs both as `python -m native_animation.x.y` and as a bare path script — preserve it in new entrypoints.
- The evaluator needs `clip` or `open_clip` (`open-clip-torch` — not yet in `comfy`).
- Keep pip installs into `comfy` conservative (`--no-deps` where sensible); the torch stack is delicate.
- Weights, datasets, and run outputs never enter git. `experiments/` and the `models` symlink are ignored by design.

## Method invariants the tests pin

`α=0` recovers unweighted MSE exactly (`_weighted_mse` normalizer); anchor frames are sliced out of every loss tensor (`[:, :, anchor_frames:]`); shift-3 Wan sigmas follow `3σ/(1+2σ)`. If you change `native_flowmatch.py` semantics, update `docs/method.md` — it quotes these numbers.
