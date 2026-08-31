# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`native-animation` — keyframe-conditioned native-animation video generation targeting CVPR 2027 (deadline ~mid-Nov 2026). The method is about native animation; flow matching is just the substrate. A Wan2.2-TI2V-5B backbone (vanilla weights + our own continued training; AniSora-V3.2 is the external baseline) is trained with the project-owned **v2 objective spine**: anchored conditional flow matching (anchor frames clamped clean and run at t=0 through TI2V's separated-timestep path; loss on non-anchors only; anchor modes keyframe / first+last / storyboard / ∅), a σ-corrected v-space temporal delta, motion-aware frame weighting, a logit-normal timestep density with a 5% high-σ tail, a data curriculum, and GT-anchored DPO (Plan 3). Modules: `src/native_animation/modeling/{timesteps,anchoring,objectives,anchored_model_fn}.py`, `src/native_animation/training/{curriculum,runner_v2,module_v2,stages/train_stage}.py`, `src/native_animation/inference/anchored.py`. The course-era v1 objective (`modeling/native_flowmatch.py`) is retained only as a baseline/ablation arm.

`docs/method.md` and `docs/method-v2-foundations.md` are the write-ups; `docs/related/animatrix.md` fixes positioning vs. Tencent's AniMatrix (arXiv:2605.03652 — concurrent, non-peer-reviewed; cite as corroboration, never write defensively). `docs/dataset.md` covers the corpus and its object-lean storage tiers. `paper/` will hold the manuscript; all experiments run fresh — no prior numbers carry over.

All paths flow from `configs/paths.env` — source it, never hardcode. `SAKUGA_ROOT=/home/c3-0/datasets/native-animation-data/sakugabooru` holds the bulk corpus (~2.2T: 281 snapshot tars, clips, shots, captions, metadata, scraper state); the workspace `../data/sakugabooru` is a symlink to it. `../models` is the weight cache (repo `models` symlink); `../third_party` is frozen (read/copy with provenance in `THIRD_PARTY.md`, never edit).

## Commands

```bash
# tests — the standing verification gate (run on a compute node; login node lies)
TEST_TARGETS=tests sbatch scripts/slurm/run_tests.sbatch
python -m pytest            # CPU-only, fine for quick local iteration

# Stage-0 data pipeline (object-lean pack mode; tiers in docs/dataset.md)
KEEP_TAR=1 sbatch scripts/slurm/stream_process.sbatch   # snapshot tars -> clips + shot packs
sbatch scripts/slurm/split_shots.sbatch                 # loose clips -> shot packs
sbatch scripts/slurm/profile_motion.sbatch
sbatch scripts/slurm/annotate.sbatch                    # GPU array; comfy env + anno-overlay
sbatch scripts/slurm/consolidate_squash.sbatch          # packs -> shots.sqsh (single-writer, never an array)
sbatch scripts/slurm/corpus_accounting.sbatch
# Orchestrate with --dependency=afterany:<id> chains — background watchers get killed; dependencies don't.
# delta_scrape.sbatch is ON HOLD until its output is packed or the object quota is raised.

# v2 smoke gates + training stages (config-driven)
sbatch scripts/slurm/smoke_memory.sbatch          # FSDP full-FT fits? s/step?
sbatch scripts/slurm/smoke_vae_fidelity.sbatch    # 16x VAE vs line art
STAGE_CONFIG=configs/ct_a.yaml sbatch scripts/slurm/train_v2.sbatch

# v1 baseline (legacy; kept runnable for ablations)
sbatch scripts/slurm/train_native_animation.sbatch

# monitor
squeue --me
sacct -j <id> --format=JobID,JobName,State,ExitCode,Elapsed -n
tail -n 120 experiments/logs/<jobname>-<id>.{out,err}
```

## Cluster conventions (hard-won — do not relearn these)

- Conda env `comfy` (Python 3.11, torch 2.7.1+cu126). Export `PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before `sbatch` (jobs inherit via `--export=ALL`); without the caps, heavy login-node imports throw `MemoryError`.
- Use absolute interpreters in sbatch (`$COMFY_PY`, `$COMFY_ACCEL` from `paths.env`) — a torch-less `~/.local` python3.12 shadows `python3`.
- SLURM **spools scripts at submit**: derive `REPO_ROOT=${SLURM_SUBMIT_DIR:-...}` (never `BASH_SOURCE`), and remember queued jobs run the spooled sbatch — resubmit after editing an `.sbatch`, while `tools/*.py` are read at execution time.
- `short` QoS preempts at 3h: long jobs need `--requeue` and idempotent per-unit state. Preemption SIGTERMs bash and **EXIT traps do not run on an untrapped signal** — staging cleanups must trap `INT TERM` too (already in the staging sbatches, with a >4h orphan sweep).
- Node-local disks are untracked by SLURM (`TmpDisk=0`); a full node fails tasks with `ENOSPC` (c1-7, 2026-08-30). Route around: `--exclude=<node>` at submit, `scontrol update JobId=<id> ExcNodeList=<node>` for queued jobs.
- **Object quota is the binding storage constraint**: 2M files/user across the whole pool (home and the datasets area are the same filesystem). Never explode shots as loose files — pack/squash tiers per `docs/dataset.md`. Check: `ssh c3-0 'zfs get -H userobjused@aeternum pool0/export5'`.
- Annotation = `comfy` env + `PYTHONPATH=$MODELS_ROOT/anno-overlay` + `--backend transformers` (Qwen3-VL-8B, ~8s/shot). There is **no** `anno` env. Never upgrade torch/transformers inside `comfy`; overlays with `--no-deps` only.
- GPU jobs need `#SBATCH --constraint=gmem48|gmem80`; the cluster has 11–12 GB nodes that cannot hold the 5B pipeline.
- Model init is fully local: configs point `model_paths` (JSON list of local safetensors) + `tokenizer_path` at `models/…` — ModelScope 500s and partial downloads burned us; zero network at train init.

## Gotchas

- Readers never assume loose shots. Go through `native_animation.data.shot_access.materialize_shot` (loose → `NA_SHOTS_EXTRA_ROOTS`, i.e. mounted `shots.sqsh` → pack-tar extraction to tmp). GPU/reader jobs source `scripts/slurm/lib/mount_shots.sh` right after `paths.env`.
- Post tags/metadata come from loose per-post JSONs **and** `clips/sidecars/sidecars_*.jsonl` — both `build_metadata.py` and `annotate_clips.py` read both.
- Gradient checkpointing is force-enabled in training modules; video FM activations OOM without it. 480×832×49 full-FT needs 8×80GB and/or `offload_models` + FSDP CPU offload; 256×448×17 fits 4×80GB.
- Every entrypoint carries a `sys.path` bootstrap so it runs as module or bare script — preserve it in new entrypoints.
- When patching scripts programmatically, use line-index insertion and verify the result by eye — silent `str.replace` no-ops have shipped broken sbatches twice.
- v1 baseline training still needs `--data_file_keys video` and `--extra_inputs input_image`; the v1 evaluator needs `open-clip-torch` (not in `comfy`).
- Keep pip installs into `comfy` conservative (`--no-deps` where sensible); the torch stack is delicate.
- Weights, datasets, and run outputs never enter git; `experiments/` and the `models` symlink are ignored by design. `cf_cookies.json` never enters git.

## Method invariants the tests pin

v2: the anchored separated-timestep build is byte-parity with upstream's frame-0 path (parity suite over `tests/tiny_wan.py`); α=0 recovers unweighted MSE exactly; the v-space delta equals the 1/σ²-corrected x̂₀ delta (`delta_mode=vspace` vs `legacy_x0_needs_sigma`); `TimestepDensity` reserves 5% mass for σ∈[0.95,1]; anchor frames are excluded from every loss reduction. v1 (`native_flowmatch.py`): shift-3 Wan sigmas follow `3σ/(1+2σ)`; anchor slice `[:, :, anchor_frames:]`. If you change objective semantics, update `docs/method.md` — it quotes these numbers.
