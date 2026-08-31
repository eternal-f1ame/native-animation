# Workspace Restructure Implementation Plan

> **Status (2026-08-30): EXECUTED** 2026-08-28. Historical record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the `~/Research/Comic/Cartoon/` workspace from its course-project layout into the CVPR 2027 `native-animation` research codebase: one canonical repo, clean workspace geography, tests, and no course identity anywhere.

**Architecture:** Pure same-filesystem migration (every move is an instant `mv`) plus an in-repo transform executed as ordered git commits. The standalone repo `native-animation-flowmatching` becomes `native-animation`; the DiffSynth fork is frozen into `third_party/` after its runtime state (weights, CSVs, outputs) is extracted; heavy non-versioned material lives beside the repo in `data/`, `models/`, `third_party/`, `archive/`.

**Tech Stack:** bash + git + `gh` (authed as `eternal-f1ame`), Python 3.11 in the `comfy` conda env (torch 2.7.1+cu126), pytest (to be installed), SLURM.

**Spec:** `docs/superpowers/specs/2026-08-28-workspace-restructure-design.md` (read it first; this plan implements it phase-for-phase).

## Global Constraints

- `WS=/home/aeternum/Research/Comic/Cartoon` — the workspace root. `COMFY_PY=/home/aeternum/anaconda3/envs/comfy/bin/python` — the only Python to use.
- Every Python invocation MUST be prefixed `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1` — heavy imports on this login node throw `MemoryError` without the thread caps.
- Move, never copy: the clips tree is 156 GB and `models/` is 11 GB. `mv` only.
- The repo lives at `$WS/FlowMatching/native-animation-flowmatching` until Task 6 moves it to `$WS/native-animation`. Tasks ≤5 use the old path only where explicitly written; Tasks ≥6 use the new path.
- Git commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- End state must contain zero "course"/"submission-bundle" language inside the repo (`archive/` and `third_party/` are exempt).
- Known-good verified facts you may rely on: `gh` is authed with `repo` scope; pytest is NOT yet installed in `comfy`; `imageio_ffmpeg`, `cv2`, `torch`, `accelerate` ARE in `comfy`; the stale root `diffsynth/` in the repo shadows `src/diffsynth` for cwd imports and is missing `core/data` (this is why Task 6 purges it).

---

### Task 1: Preflight gate, snapshot, workspace skeleton

**Files:**
- Create: `$WS/data/sakugabooru/{metadata,scrape-logs}/`, `$WS/models/`, `$WS/third_party/`, `$WS/archive/{course-report,planning,runs-2026-04,prototypes}/`
- Create: `$WS/archive/migration-snapshot.txt`

**Interfaces:**
- Produces: the destination directories every later task moves into. Task 14 asserts the final root listing against this skeleton.

- [x] **Step 1: Gate on running SLURM jobs**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
squeue --me
for j in $(squeue --me -h -o %i); do
  scontrol show job "$j" | grep -E 'WorkDir=|StdOut=|StdErr=|Command=' | grep -F "$WS" \
    && { echo "BLOCKED: job $j touches $WS — stop and ask the owner"; exit 1; }
done
echo "preflight clear"
```

Expected: `preflight clear`. (At plan time one job existed — `358562 ucf-vsco`, WorkDir `/home/aeternum`, outside the tree. If the loop prints BLOCKED, STOP and ask the user.)

- [x] **Step 2: Record the pre-migration snapshot**

```bash
mkdir -p "$WS/archive"
{ date; echo; df -h "$WS"; echo; find "$WS" -maxdepth 2 | sort; } > "$WS/archive/migration-snapshot.txt"
```

- [x] **Step 3: Create the skeleton**

```bash
mkdir -p "$WS/data/sakugabooru/metadata" "$WS/data/sakugabooru/scrape-logs" \
         "$WS/models" "$WS/third_party" \
         "$WS/archive/course-report" "$WS/archive/planning" \
         "$WS/archive/runs-2026-04" "$WS/archive/prototypes"
```

- [x] **Step 4: Verify**

```bash
ls "$WS" && ls "$WS/archive" && test -s "$WS/archive/migration-snapshot.txt" && echo OK
```

Expected: new dirs listed alongside the old ones; `OK`.

---

### Task 2: Move ComfyUI out of the workspace

**Files:**
- Move: `$WS/ComfyUI/` → `/home/aeternum/Research/Comic/ComfyUI/`
- Move: `$WS/README_comfyui_setup.md` → `/home/aeternum/Research/Comic/README_comfyui_setup.md`

- [x] **Step 1: Move**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
mv "$WS/ComfyUI" /home/aeternum/Research/Comic/ComfyUI
mv "$WS/README_comfyui_setup.md" /home/aeternum/Research/Comic/README_comfyui_setup.md
```

- [x] **Step 2: Verify**

```bash
test -d /home/aeternum/Research/Comic/ComfyUI/models && ! test -e "$WS/ComfyUI" && echo OK
```

---

### Task 3: Relocate the dataset

**Files:**
- Move: `$WS/Anime/sakugabooru_clips/` → `$WS/data/sakugabooru/clips/` (the scraper's `_state.json` lives inside and travels with it)
- Move: `$WS/Anime/{scrape.log,scrape_335809.log}` → `$WS/data/sakugabooru/scrape-logs/`
- Move: `$WS/Anime/cf_cookies.json` → `$WS/data/sakugabooru/cf_cookies.json`
- Move: `$WS/Anime/slurm_scrape.sh` → `$WS/archive/planning/`
- Move: `$WS/Anime/README.md` → `$WS/archive/planning/anime-dataset-README.md` (Task 10 mines it for `docs/dataset.md`)
- Hold: `$WS/Anime/{scrape_sakugabooru.py,extract_cf_cookies.py,build_dataset.py}` → `$WS/archive/planning/tools-staging/` (Task 8 moves them into the repo — the repo dir doesn't exist at its final path yet)
- Delete: `$WS/Anime/__pycache__/`, then the emptied `$WS/Anime/`

**Interfaces:**
- Produces: `$WS/data/sakugabooru/clips/<series>/{post_id}_s{score}.mp4 + {post_id}.json` — the `DATA_ROOT` every script in Task 7 points at. 241 series directories.

- [x] **Step 1: Record the expected clip count, then move everything**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
SERIES_BEFORE=$(ls "$WS/Anime/sakugabooru_clips" | wc -l)   # expect 241 (+ _state.json entry = 242 lines is fine)
mv "$WS/Anime/sakugabooru_clips" "$WS/data/sakugabooru/clips"
mv "$WS/Anime/scrape.log" "$WS/Anime/scrape_335809.log" "$WS/data/sakugabooru/scrape-logs/"
mv "$WS/Anime/cf_cookies.json" "$WS/data/sakugabooru/cf_cookies.json"
mv "$WS/Anime/slurm_scrape.sh" "$WS/archive/planning/slurm_scrape.sh"
mv "$WS/Anime/README.md" "$WS/archive/planning/anime-dataset-README.md"
mkdir -p "$WS/archive/planning/tools-staging"
mv "$WS/Anime/scrape_sakugabooru.py" "$WS/Anime/extract_cf_cookies.py" "$WS/Anime/build_dataset.py" \
   "$WS/archive/planning/tools-staging/"
rm -rf "$WS/Anime/__pycache__"
rmdir "$WS/Anime"
```

`rmdir` (not `rm -rf`) is deliberate: it fails loudly if anything unexpected remains. If it fails, list the leftovers, move them to `$WS/archive/planning/`, and rerun.

- [x] **Step 2: Verify**

```bash
test $(ls "$WS/data/sakugabooru/clips" | grep -vc '^_state.json$') -eq 241 \
  && test -f "$WS/data/sakugabooru/clips/_state.json" \
  && ! test -e "$WS/Anime" && echo OK
```

---

### Task 4: Extract the fork's runtime state

**Files (source root `F=$WS/FlowMatching/DiffSynth`):**
- Move: `$F/models/*` → `$WS/models/`
- Move: `$F/data/course_flowmatch_i2v/*` → `$WS/data/sakugabooru/metadata/`
- Move: `$F/outputs/` → `$WS/archive/runs-2026-04/outputs/`
- Move: `$F/course_project/flowmatch_i2v/logs/` → `$WS/archive/runs-2026-04/slurm-logs/`
- Move: `$F/dist/` → `$WS/archive/runs-2026-04/dist/`

**Interfaces:**
- Produces: `$WS/models/DiffSynth-Studio/…` and `$WS/models/Wan-AI/…` (the shared weight cache Task 7's symlink targets); `$WS/data/sakugabooru/metadata/metadata_{all,train,val,test}.csv` (11,786 total rows; video paths are RELATIVE to the clips root, so they stay valid); `$WS/archive/runs-2026-04/dist/native_animation_submission/` (Task 6's diff target for the stale dirs).

- [x] **Step 1: Move runtime state**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
F=$WS/FlowMatching/DiffSynth
mv "$F"/models/* "$WS/models/"
mv "$F"/data/course_flowmatch_i2v/* "$WS/data/sakugabooru/metadata/"
mv "$F/outputs" "$WS/archive/runs-2026-04/outputs"
mv "$F/course_project/flowmatch_i2v/logs" "$WS/archive/runs-2026-04/slurm-logs"
mv "$F/dist" "$WS/archive/runs-2026-04/dist"
rmdir "$F/models" "$F/data/course_flowmatch_i2v" "$F/data" 2>/dev/null || true
```

- [x] **Step 2: Verify**

```bash
ls "$WS/models"                                    # expect: DiffSynth-Studio  Wan-AI
wc -l "$WS/data/sakugabooru/metadata/metadata_all.csv"     # expect 11787 (header + 11786 rows)
test -d "$WS/archive/runs-2026-04/dist/native_animation_submission" && echo OK
```

Note: the fork stays where it is, dirtier but intact — it is frozen as-is in Task 5. Do NOT commit anything inside it.

---

### Task 5: Freeze the fork and the reference clones into third_party/

**Files:**
- Move: `$WS/FlowMatching/DiffSynth/` → `$WS/third_party/DiffSynth-fork/`
- Move: `$WS/FlowMatching/{DiffSynth-Studio,goku,Sana,Pyramid-Flow,flowception,Janus,CausVid}/` → `$WS/third_party/`
- Create: `$WS/third_party/README.md`

- [x] **Step 1: Move**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
mv "$WS/FlowMatching/DiffSynth" "$WS/third_party/DiffSynth-fork"
for d in DiffSynth-Studio goku Sana Pyramid-Flow flowception Janus CausVid; do
  mv "$WS/FlowMatching/$d" "$WS/third_party/$d"
done
```

- [x] **Step 2: Write `$WS/third_party/README.md`**

```markdown
# Third-Party Trees

Read-only reference material. Nothing here is developed; the live codebase is `../native-animation/`.

| Tree | Role |
|---|---|
| `DiffSynth-fork/` | Frozen former implementation host (branch `course-flowmatch-i2v-minimal`). The repo's `src/diffsynth/` subset was carved from here; its runtime state (weights, CSVs, outputs) was extracted to `../models/`, `../data/`, and `../archive/` in Aug 2026. Copy modules from here (or upstream) when the vendored runtime needs to grow. |
| `DiffSynth-Studio/` | Pristine upstream (modelscope/DiffSynth-Studio), for diffing against the fork and pulling fresh modules. |
| `goku/` | Rectified-flow / video-DiT conceptual reference. |
| `Sana/` | Secondary I2V design reference. |
| `Pyramid-Flow/` | Multi-scale / long-horizon Flow Matching design reference. |
| `flowception/` | Frame-insertion, variable-length generation — future clip-extension direction. |
| `Janus/` | Unused (multimodal understanding; possible captioning utility). |
| `CausVid/` | Unused (autoregressive video distillation reference). |
```

- [x] **Step 3: Verify**

```bash
test $(ls "$WS/third_party" | wc -l) -eq 9 && ls "$WS/FlowMatching"
```

Expected: 9 entries (8 trees + README); `FlowMatching/` now holds only `native-animation-flowmatching`, `anime_keyframe_fm`, `Project.md`, `progress.md`, `Mid-Project Update.pdf`, `evaluation.py`, `.claude`, `.codex`.

---

### Task 6: Relocate the repo; purge stale state; rename Final_Project_Report → paper/

**Files (repo `R=$WS/native-animation` after the first step):**
- Move: `$WS/FlowMatching/native-animation-flowmatching/` → `$WS/native-animation/`
- Delete (after diff-verify): untracked `$R/course_project/`, `$R/diffsynth/`
- Move: `$R/presentation/` → `$WS/archive/course-report/presentation/`; `$R/Final_Project_Report/Final_Project_Report_Aaditya.pdf` and `$R/Final_Project_Report/frog.jpg` → `$WS/archive/course-report/`
- Git-modify: commit `main.pdf` deletion; `git mv Final_Project_Report paper`

**Interfaces:**
- Produces: repo at `$WS/native-animation` on branch `main` with `paper/` (containing `main.tex`, `sample.bib`, `images/`, result PNGs, `scripts/`) and no untracked noise. All later tasks operate here.

- [x] **Step 1: Relocate the repo**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
mv "$WS/FlowMatching/native-animation-flowmatching" "$WS/native-animation"
cd "$WS/native-animation" && git status --short   # sanity: git works at the new path
```

- [x] **Step 2: Diff-verify the stale flat-export dirs, then delete (or archive if divergent)**

The untracked root `course_project/` and `diffsynth/` are believed to be copies of the old flat export. Prove it, then delete; if either diverges, archive it instead of deleting:

```bash
cd "$WS/native-animation"
DIST=$WS/archive/runs-2026-04/dist/native_animation_submission
FORK=$WS/third_party/DiffSynth-fork
# course_project: try the live fork first, then the dist bundle
if diff -rq -x '__pycache__' course_project "$FORK/course_project" > /dev/null 2>&1 \
   || diff -rq -x '__pycache__' course_project "$DIST/course_project" > /dev/null 2>&1; then
  rm -rf course_project
else
  mkdir -p "$WS/archive/planning/stale-flat-export" && mv course_project "$WS/archive/planning/stale-flat-export/"
fi
# diffsynth: only the dist bundle is a valid twin (the fork's diffsynth is the full runtime)
if diff -rq -x '__pycache__' diffsynth "$DIST/diffsynth" > /dev/null 2>&1; then
  rm -rf diffsynth
else
  mkdir -p "$WS/archive/planning/stale-flat-export" && mv diffsynth "$WS/archive/planning/stale-flat-export/"
fi
ls course_project diffsynth 2>&1   # both must report "No such file or directory"
```

Deleting these is also a bugfix: the stale `diffsynth/` shadows `src/diffsynth` for cwd-based imports and is missing `core/data`.

- [x] **Step 3: Archive presentation and course-report artifacts**

```bash
mv presentation "$WS/archive/course-report/presentation"
mv Final_Project_Report/Final_Project_Report_Aaditya.pdf "$WS/archive/course-report/"
mv Final_Project_Report/frog.jpg "$WS/archive/course-report/"
```

- [x] **Step 4: Commit the purge (records `main.pdf` deletion; tree has no untracked noise left)**

```bash
git add -A
git status --short    # expect ONLY: "D Final_Project_Report/main.pdf" staged
git commit -m "Remove stale flat-export copies and course-report artifacts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

If `git status` shows anything besides the `main.pdf` deletion, stop and inspect before committing.

- [x] **Step 5: Rename the report directory and commit**

```bash
git mv Final_Project_Report paper
git commit -m "Rename Final_Project_Report/ to paper/

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
ls paper    # expect: main.tex sample.bib images/ scripts/ *.png
```

(`main.tex` uses relative `images/...` includes, so the rename needs no tex edits. The `.gitignore` LaTeX paths are fixed in Task 7.)

---

### Task 7: Repo identity — pyproject, paths.env, .gitignore, scripts, symlink

**Files (all under `$WS/native-animation`):**
- Modify: `pyproject.toml` (full replacement below)
- Create: `configs/paths.env`, `experiments/logs/.gitkeep`, symlink `models -> ../models`
- Modify: `.gitignore` (full replacement below)
- Modify: `scripts/train_native_animation.sh`, `scripts/slurm/{build_metadata,env_smoke_test,base_inference_demo,train_native_animation}.sbatch` (full replacements below)

**Interfaces:**
- Consumes: `$WS/data/sakugabooru/{clips,metadata}` (Task 3/4), `$WS/models` (Task 4).
- Produces: `configs/paths.env` exporting `REPO_ROOT, WORKSPACE_ROOT, DATA_ROOT, METADATA_DIR, MODELS_ROOT, EXPERIMENTS_ROOT` — every script and doc from here on assumes these names. pyproject test config: `pythonpath=["src"]`, `testpaths=["tests"]` (Task 9 relies on it).

- [x] **Step 1: Replace `pyproject.toml` with:**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "native-animation"
version = "0.2.0"
description = "Keyframe-conditioned native-animation video generation with Flow Matching."
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.10"
dependencies = [
    "accelerate",
    "datasets",
    "einops",
    "ftfy",
    "imageio[ffmpeg]",
    "matplotlib",
    "modelscope",
    "numpy",
    "open-clip-torch",
    "opencv-python-headless",
    "pandas",
    "peft",
    "Pillow",
    "protobuf",
    "safetensors",
    "sentencepiece",
    "torch>=2.0.0",
    "torchvision",
    "transformers",
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
native-animation-build-metadata = "native_animation.data.build_metadata:main"
native-animation-extract-keyframes = "native_animation.data.extract_keyframes:main"
native-animation-run-baseline = "native_animation.inference.run_baseline:main"
native-animation-generate = "native_animation.inference.generate:main"
native-animation-train = "native_animation.training.train:main"
native-animation-evaluate = "native_animation.evaluation.evaluate:main"

[tool.setuptools.package-dir]
"" = "src"

[tool.setuptools.packages.find]
where = ["src"]
include = ["native_animation", "native_animation.*", "diffsynth", "diffsynth.*"]

[tool.setuptools]
include-package-data = true

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [x] **Step 2: Create `configs/paths.env` with:**

```bash
# Native Animation — single source of truth for machine paths.
# Every entrypoint script sources this file. Every variable honors a
# pre-existing environment value, so `DATA_ROOT=... sbatch ...` overrides work.
_NA_CONFIG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$_NA_CONFIG_DIR/.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE_ROOT/data/sakugabooru/clips}
METADATA_DIR=${METADATA_DIR:-$WORKSPACE_ROOT/data/sakugabooru/metadata}
MODELS_ROOT=${MODELS_ROOT:-$WORKSPACE_ROOT/models}
EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT:-$REPO_ROOT/experiments}
export REPO_ROOT WORKSPACE_ROOT DATA_ROOT METADATA_DIR MODELS_ROOT EXPERIMENTS_ROOT
```

- [x] **Step 3: Replace `.gitignore` with:**

```
__pycache__/
*.py[cod]
*.so

.venv/
venv/
.env
.pytest_cache/
.mypy_cache/
.ipynb_checkpoints/

/data/
/dist/
/cache/
.cache/
wandb/

/models
/experiments/*
!/experiments/logs/
/experiments/logs/*
!/experiments/logs/.gitkeep

*.ckpt
*.pt
*.pth
*.safetensors
*.bin

# LaTeX build artifacts
paper/main.aux
paper/main.bbl
paper/main.blg
paper/main.log
paper/main.out
```

- [x] **Step 4: Create the experiments scaffold and the model-cache symlink**

```bash
cd "$WS/native-animation"
mkdir -p experiments/logs && touch experiments/logs/.gitkeep
ln -s ../models models
readlink -f models    # expect /home/aeternum/Research/Comic/Cartoon/models
```

The symlink makes DiffSynth's cwd-relative `./models/...` downloads land in the shared workspace cache. The committed `.gitkeep` guarantees `experiments/logs/` exists at sbatch submission time — SLURM does not create missing `--output` directories.

- [x] **Step 5: Replace `scripts/train_native_animation.sh` with:**

```bash
#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

METADATA_PATH=${METADATA_PATH:-$METADATA_DIR/metadata_train.csv}
OUTPUT_PATH=${OUTPUT_PATH:-$EXPERIMENTS_ROOT/checkpoints/native_animation_flowmatch_lora}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-49}
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DATASET_REPEAT=${DATASET_REPEAT:-20}
LORA_RANK=${LORA_RANK:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1000}
NATIVE_SCHEDULER_SHIFT=${NATIVE_SCHEDULER_SHIFT:-3.0}
MOTION_WEIGHTING_SCALE=${MOTION_WEIGHTING_SCALE:-1.0}
DELTA_LOSS_WEIGHT=${DELTA_LOSS_WEIGHT:-0.25}
MODEL_ID_WITH_ORIGIN_PATHS=${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth}
ACCELERATE_EXTRA_ARGS=${ACCELERATE_EXTRA_ARGS:-}

accelerate launch ${ACCELERATE_EXTRA_ARGS} src/native_animation/training/train.py \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --data_file_keys "video" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${LORA_RANK}" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --native_scheduler_shift "${NATIVE_SCHEDULER_SHIFT}" \
  --motion_weighting_scale "${MOTION_WEIGHTING_SCALE}" \
  --delta_loss_weight "${DELTA_LOSS_WEIGHT}"
```

- [x] **Step 6: Replace `scripts/slurm/build_metadata.sbatch` with:**

```bash
#!/bin/bash
#SBATCH --job-name=na-build-meta
#SBATCH --partition=short
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

mkdir -p "$EXPERIMENTS_ROOT/logs" "$METADATA_DIR"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V

python -m native_animation.data.build_metadata \
  --input-root "$DATA_ROOT" \
  --output-dir "$METADATA_DIR" \
  --seed 42 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

- [x] **Step 7: Replace `scripts/slurm/env_smoke_test.sbatch` with:**

```bash
#!/bin/bash
#SBATCH --job-name=na-env-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

mkdir -p "$EXPERIMENTS_ROOT/logs"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "HOSTNAME=$(hostname)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V
nvidia-smi

timeout 120s python -u -c "import sys; print('exe=', sys.executable, flush=True); import torch; print('torch=', torch.__version__, flush=True); print('cuda=', torch.cuda.is_available(), flush=True); import accelerate; print('accelerate=', accelerate.__version__, flush=True); import native_animation; print('native_animation=', native_animation.__version__, flush=True); import diffsynth; print('diffsynth=ok', flush=True)"
```

- [x] **Step 8: Replace `scripts/slurm/base_inference_demo.sbatch` with:**

```bash
#!/bin/bash
#SBATCH --job-name=na-base-demo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --constraint=gmem48|gmem80
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

mkdir -p "$EXPERIMENTS_ROOT/logs" "$EXPERIMENTS_ROOT/demo/base"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "HOSTNAME=$(hostname)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V
nvidia-smi

python -m native_animation.inference.run_baseline \
  --input-csv "$METADATA_DIR/metadata_test.csv" \
  --dataset-base-path "$DATA_ROOT" \
  --output-dir "$EXPERIMENTS_ROOT/demo/base" \
  --limit 2 \
  --unique-series \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --num-inference-steps 30 \
  --cfg-scale 5.0 \
  --seed-base 1234 \
  --fps 15 \
  --quality 5 \
  --tiled
```

- [x] **Step 9: Replace `scripts/slurm/train_native_animation.sbatch` with:**

```bash
#!/bin/bash
#SBATCH --job-name=na-train-fm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --constraint=gmem48|gmem80
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

mkdir -p "$EXPERIMENTS_ROOT/logs" "$EXPERIMENTS_ROOT/checkpoints"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V

export METADATA_PATH=${METADATA_PATH:-$METADATA_DIR/metadata_train.csv}
export OUTPUT_PATH=${OUTPUT_PATH:-$EXPERIMENTS_ROOT/checkpoints/native_animation_flowmatch_lora}
export HEIGHT=${HEIGHT:-480}
export WIDTH=${WIDTH:-832}
export NUM_FRAMES=${NUM_FRAMES:-49}
export NUM_EPOCHS=${NUM_EPOCHS:-2}
export LEARNING_RATE=${LEARNING_RATE:-1e-4}
export DATASET_REPEAT=${DATASET_REPEAT:-20}
export LORA_RANK=${LORA_RANK:-32}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export NATIVE_SCHEDULER_SHIFT=${NATIVE_SCHEDULER_SHIFT:-3.0}
export MOTION_WEIGHTING_SCALE=${MOTION_WEIGHTING_SCALE:-1.0}
export DELTA_LOSS_WEIGHT=${DELTA_LOSS_WEIGHT:-0.25}
export MODEL_ID_WITH_ORIGIN_PATHS=${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth}

bash scripts/train_native_animation.sh
```

- [x] **Step 10: Verify scripts parse and paths resolve**

```bash
cd "$WS/native-animation"
for f in scripts/train_native_animation.sh scripts/slurm/*.sbatch configs/paths.env; do bash -n "$f" || echo "SYNTAX FAIL: $f"; done
bash -c 'source configs/paths.env && echo "$DATA_ROOT" && echo "$METADATA_DIR" && test -d "$DATA_ROOT" && test -f "$METADATA_DIR/metadata_train.csv" && echo PATHS-OK'
```

Expected: no syntax failures; the two paths print under `$WS/data/sakugabooru/`; `PATHS-OK`.

- [x] **Step 11: Commit**

```bash
git add pyproject.toml configs/paths.env .gitignore scripts/ experiments/logs/.gitkeep models
git commit -m "Adopt native-animation identity: centralized paths.env, experiments/ layout, model-cache symlink

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Move the dataset tooling into the repo

**Files:**
- Move: `$WS/archive/planning/tools-staging/scrape_sakugabooru.py` → `tools/scrape_sakugabooru.py`
- Move: `$WS/archive/planning/tools-staging/extract_cf_cookies.py` → `tools/extract_cf_cookies.py`
- Move: `$WS/archive/planning/tools-staging/build_dataset.py` → `tools/build_pair_dataset.py`
- Modify: path constants in all three (exact edits below)

**Interfaces:**
- Consumes: staged scripts from Task 3.
- Produces: `tools/` whose default paths resolve to the workspace data layout: from `tools/<script>.py`, `Path(__file__).resolve().parents[2]` == `$WS`.

- [x] **Step 1: Move**

```bash
cd "$WS/native-animation" && mkdir -p tools
mv "$WS/archive/planning/tools-staging/scrape_sakugabooru.py" tools/scrape_sakugabooru.py
mv "$WS/archive/planning/tools-staging/extract_cf_cookies.py" tools/extract_cf_cookies.py
mv "$WS/archive/planning/tools-staging/build_dataset.py"      tools/build_pair_dataset.py
rmdir "$WS/archive/planning/tools-staging"
```

- [x] **Step 2: Update path constants**

In `tools/scrape_sakugabooru.py` (constants at ~lines 38 and 229–230):

```python
# old
COOKIE_FILE = Path(__file__).parent / "cf_cookies.json"
# new
COOKIE_FILE = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "cf_cookies.json"
```

```python
# old
OUTPUT_DIR = Path(__file__).parent / "sakugabooru_clips"
# new
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "clips"
```

(`STATE_FILE = OUTPUT_DIR / "_state.json"` stays as-is — it follows `OUTPUT_DIR`.)

In `tools/build_pair_dataset.py` (constants at ~lines 36–37):

```python
# old
SOURCE_DIR = Path(__file__).parent / "sakugabooru_clips"
OUTPUT_DIR = Path(__file__).parent / "dataset"
# new
SOURCE_DIR = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "clips"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "pairs"
```

In `tools/extract_cf_cookies.py`: find every `Path(__file__).parent / "cf_cookies.json"` (or equivalent local `cf_cookies.json` write) and point it at `Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "cf_cookies.json"`. Verify nothing was missed:

```bash
grep -n 'cf_cookies\|sakugabooru_clips\|"dataset"' tools/*.py
```

Every hit must be the new `parents[2]`-based form (or a docstring mention).

- [x] **Step 3: Verify compile + default resolution**

```bash
COMFY_PY=/home/aeternum/anaconda3/envs/comfy/bin/python
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 \
  $COMFY_PY -m py_compile tools/*.py && echo COMPILE-OK
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 \
  $COMFY_PY - <<'EOF'
from pathlib import Path
p = Path("tools/scrape_sakugabooru.py").resolve().parents[2]
assert (p / "data/sakugabooru/clips").is_dir(), p
print("RESOLVE-OK", p)
EOF
```

- [x] **Step 4: Commit**

```bash
git add tools/
git commit -m "Bring dataset scraping and pair-building tools into the repo

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Test suite

**Files:**
- Create: `tests/test_imports.py`, `tests/test_native_flowmatch.py`, `tests/test_build_metadata.py`, `tests/test_evaluate_metrics.py`

**Interfaces:**
- Consumes: `[tool.pytest.ini_options]` from Task 7 (`pythonpath=["src"]`).
- Produces: a green `pytest` run — the standing verification gate for all future work.

All expected values below were verified against the live code on this machine before this plan was written: the shift-3 Wan sigmas match `3σ/(1+2σ)`, `_motion_frame_weights` returns `(2,1,4,1,1)` with `anchor_frames=1` and max exactly `1+scale`, DFS is `0.0` on a flat curve and `≈0.777` on the collapse curve. A test failure therefore means a real regression (or environment problem), not a wrong expectation.

- [x] **Step 1: Install pytest into `comfy` (verified absent at plan time)**

```bash
COMFY_PY=/home/aeternum/anaconda3/envs/comfy/bin/python
$COMFY_PY -m pip install pytest
$COMFY_PY -m pytest --version
```

- [x] **Step 2: Write `tests/test_imports.py`**

```python
"""Import sweep over the light modules.

The heavy pipeline entrypoints (training/train.py, inference/*) pull the full
Wan stack and are covered by py_compile in the migration verification instead.
"""
import importlib

import pytest

MODULES = [
    "native_animation",
    "native_animation.data.build_metadata",
    "native_animation.data.sampling",
    "native_animation.data.extract_keyframes",
    "native_animation.modeling.native_flowmatch",
    "native_animation.evaluation.evaluate",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
```

- [x] **Step 3: Write `tests/test_native_flowmatch.py`**

```python
"""Invariants of the project-owned scheduler and loss helpers (CPU-only)."""
import torch

from native_animation.modeling.native_flowmatch import (
    NativeAnimationFlowMatchScheduler,
    _motion_frame_weights,
    _weighted_mse,
)


def _wan_sigmas(num_steps: int, shift: float) -> torch.Tensor:
    """Reference Wan schedule: linspace sigmas pushed through the shift map."""
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def test_scheduler_stores_and_applies_project_shift():
    sched = NativeAnimationFlowMatchScheduler(shift=3.0)
    assert sched.shift == 3.0
    sched.set_timesteps(num_inference_steps=10, training=True)
    assert torch.allclose(sched.sigmas, _wan_sigmas(10, 3.0))


def test_scheduler_explicit_shift_overrides_default():
    sched = NativeAnimationFlowMatchScheduler(shift=3.0)
    sched.set_timesteps(num_inference_steps=10, training=True, shift=5.0)
    assert torch.allclose(sched.sigmas, _wan_sigmas(10, 5.0))
    assert not torch.allclose(sched.sigmas, _wan_sigmas(10, 3.0))


def test_motion_weights_shapes_with_and_without_anchor():
    latents = torch.randn(2, 4, 5, 3, 3)  # (B, C, T, H, W)
    with_anchor = _motion_frame_weights(latents, anchor_frames=1, motion_weighting_scale=1.0)
    assert tuple(with_anchor.shape) == (2, 1, 4, 1, 1)  # T-1 weights for supervised frames
    without_anchor = _motion_frame_weights(latents, anchor_frames=0, motion_weighting_scale=1.0)
    assert tuple(without_anchor.shape) == (2, 1, 5, 1, 1)  # padded to T
    assert torch.all(without_anchor[:, :, 0] == 1.0)  # leading pad slot is neutral


def test_motion_weights_range_and_disable():
    latents = torch.randn(2, 4, 5, 3, 3)
    weights = _motion_frame_weights(latents, anchor_frames=1, motion_weighting_scale=1.0)
    # Per-clip normalization puts the most active frame at exactly 1 + scale.
    assert torch.isclose(weights.max(), torch.tensor(2.0))
    assert torch.all(weights >= 1.0)
    assert _motion_frame_weights(latents, anchor_frames=1, motion_weighting_scale=0.0) is None
    single_frame = torch.randn(2, 4, 1, 3, 3)
    assert _motion_frame_weights(single_frame, anchor_frames=0, motion_weighting_scale=1.0) is None


def test_weighted_mse_reduces_to_plain_mse_with_unit_weights():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 4, 5, 3, 3), torch.randn(2, 4, 5, 3, 3)
    unweighted = _weighted_mse(pred, target)
    unit = _weighted_mse(pred, target, torch.ones(2, 1, 5, 1, 1))
    assert torch.isclose(unweighted, unit)
    assert torch.isclose(unweighted, (pred - target).pow(2).mean())


def test_weighted_mse_is_invariant_to_weight_rescaling():
    torch.manual_seed(1)
    pred, target = torch.randn(2, 4, 5, 3, 3), torch.randn(2, 4, 5, 3, 3)
    weights = 1.0 + torch.rand(2, 1, 5, 1, 1)
    assert torch.isclose(
        _weighted_mse(pred, target, weights),
        _weighted_mse(pred, target, 2.0 * weights),
    )
```

- [x] **Step 4: Write `tests/test_build_metadata.py`**

```python
"""Metadata builder: split determinism, series-level leakage guarantee, CSV shape."""
import csv
import json
import sys
from pathlib import Path

from native_animation.data.build_metadata import (
    build_prompt,
    build_series_split,
    find_video_for_json,
    main,
)


def _make_clip(root: Path, series: str, post_id: int, score: int = 100) -> None:
    series_dir = root / series
    series_dir.mkdir(parents=True, exist_ok=True)
    (series_dir / f"{post_id}.json").write_text(
        json.dumps(
            {
                "id": post_id,
                "score": score,
                "tags": "animated smears fighting",
                "width": 852,
                "height": 480,
                "source": "test",
            }
        )
    )
    # The builder only checks file existence; an empty file is enough.
    (series_dir / f"{post_id}_s{score}.mp4").write_bytes(b"")


def test_series_split_is_deterministic_and_covers_all():
    names = [f"series_{i}" for i in range(10)]
    split_a = build_series_split(names, val_ratio=0.1, test_ratio=0.1, seed=42)
    split_b = build_series_split(names, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert split_a == split_b
    assert set(split_a) == set(names)
    assert sorted(set(split_a.values())) == ["test", "train", "val"]


def test_find_video_matches_score_suffixed_files(tmp_path):
    _make_clip(tmp_path, "series_a", 111, score=250)
    json_path = tmp_path / "series_a" / "111.json"
    assert find_video_for_json(json_path).name == "111_s250.mp4"
    (tmp_path / "series_a" / "999.json").write_text("{}")
    assert find_video_for_json(tmp_path / "series_a" / "999.json") is None


def test_prompt_format():
    prompt = build_prompt("one_piece", ["smears", "fighting"], max_tags=20, prompt_prefix="native animation, anime")
    assert prompt == "native animation, anime, one_piece, smears, fighting"


def test_end_to_end_build_has_no_series_leakage(tmp_path, monkeypatch):
    clips_root = tmp_path / "clips"
    for i in range(10):
        _make_clip(clips_root, f"series_{i}", post_id=1000 + 2 * i)
        _make_clip(clips_root, f"series_{i}", post_id=1001 + 2 * i)
    out_dir = tmp_path / "meta"

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_metadata", "--input-root", str(clips_root), "--output-dir", str(out_dir), "--seed", "42"],
    )
    main()

    with (out_dir / "metadata_all.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    # Leakage guarantee: every series lives in exactly one split.
    series_to_splits = {}
    for row in rows:
        series_to_splits.setdefault(row["series"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in series_to_splits.values())
    # The per-split CSVs partition the full set.
    sizes = {}
    for name in ("train", "val", "test"):
        with (out_dir / f"metadata_{name}.csv").open() as handle:
            sizes[name] = len(list(csv.DictReader(handle)))
    assert sum(sizes.values()) == 20
    assert sizes["val"] >= 2 and sizes["test"] >= 2  # at least one series each (2 clips/series)
```

- [x] **Step 5: Write `tests/test_evaluate_metrics.py`**

```python
"""Aggregate metric behavior on synthetic score curves (no CLIP, no video IO)."""
import numpy as np

from native_animation.evaluation.evaluate import (
    classify_result,
    continuation_fidelity,
    diffusion_failure_score,
    final_score,
    temporal_consistency,
    worst_segment,
)


def _flat(value: float = 0.9, n: int = 50) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def test_flat_curve_is_healthy():
    scores = _flat()
    dfs, *_ = diffusion_failure_score(scores)
    assert dfs == 0.0
    assert np.isclose(temporal_consistency(scores), 0.9, atol=1e-6)
    assert np.isclose(worst_segment(scores), 0.9, atol=1e-6)
    assert np.isclose(continuation_fidelity(scores), 0.9, atol=1e-6)


def test_mid_clip_collapse_trips_dfs():
    collapse = np.concatenate(
        [np.full(17, 0.9), np.full(16, 0.4), np.full(17, 0.9)]
    ).astype(np.float32)
    dfs, mid_drop, _, start, middle, end = diffusion_failure_score(collapse)
    assert dfs >= 0.7  # verified 0.777 on the live implementation
    assert mid_drop > 0.3
    assert middle < start and middle < end


def test_jitter_hurts_tcs_more_than_dfs():
    jitter = np.tile([0.9, 0.7], 25).astype(np.float32)
    dfs, *_ = diffusion_failure_score(jitter)
    assert 0.25 <= dfs <= 0.5  # smoothness term only; no mid-drop
    assert temporal_consistency(jitter) < temporal_consistency(_flat(0.8))


def test_worst_segment_finds_the_dip():
    curve = np.concatenate([np.full(10, 0.9), np.full(5, 0.2), np.full(10, 0.9)]).astype(np.float32)
    assert np.isclose(worst_segment(curve, window=5), 0.2, atol=1e-6)


def test_final_score_formula_and_classification():
    assert np.isclose(final_score(1.0, 1.0, 1.0, 0.0), 0.85)
    assert np.isclose(final_score(0.0, 0.0, 0.0, 1.0), -0.5)
    failing = {"CFS": 0.8, "TCS": 0.8, "WorstSegment": 0.8, "DFS": 0.75}
    assert classify_result(failing) == "[FAIL] Diffusion failure"
```

- [x] **Step 6: Run the suite**

```bash
cd "$WS/native-animation"
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 \
  /home/aeternum/anaconda3/envs/comfy/bin/python -m pytest -v
```

Expected: all tests PASS (≈17 tests). A failure is a real finding — investigate the code, do not adjust the expectation to match; if it exposes a genuine pre-existing bug, report it to the user before "fixing" project semantics.

- [x] **Step 7: Commit**

```bash
git add tests/
git commit -m "Add CPU-only test suite for scheduler, loss helpers, metadata build, and metrics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Docs — README, THIRD_PARTY, dataset.md, roadmap.md, repo CLAUDE.md, string audit

**Files:**
- Modify: `README.md` (full replacement below), `THIRD_PARTY.md` (full replacement), `docs/method.md` (one link fix)
- Delete: `export_manifest.json`
- Create: `docs/dataset.md`, `docs/roadmap.md`, `CLAUDE.md` (repo-level)

- [x] **Step 1: Replace `README.md` with:**

```markdown
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
```

- [x] **Step 2: Replace `THIRD_PARTY.md` with:**

```markdown
# Third-Party Components

`src/diffsynth/` is a vendored subset of DiffSynth-Studio (Apache-2.0), carried in-repo so the project runs from a single clone.

Provenance: the subset was carved from the project's former DiffSynth working fork — frozen at `<workspace>/third_party/DiffSynth-fork` — and migrated into this repository in August 2026. Upstream: https://github.com/modelscope/DiffSynth-Studio.

When the method needs deeper runtime access, copy the required modules from the frozen fork (or current upstream) into `src/diffsynth/` and note the addition here. Keep the root `LICENSE` file with the vendored code when publishing.
```

- [x] **Step 3: Delete the exporter manifest and fix the method.md link**

```bash
git rm export_manifest.json
```

In `docs/method.md`, change the intro line's link target `../Final_Project_Report/main.tex` → `../paper/main.tex` (both occurrences if repeated; check with `grep -n Final_Project_Report docs/method.md`).

- [x] **Step 4: Create `docs/dataset.md` with:**

```markdown
# Sakugabooru Dataset

Curated animation clips (sakuga) scraped from [Sakugabooru](https://www.sakugabooru.com) — community-selected examples of notable animation craft: smears, morphing, impact frames, effects animation, fluid motion, and more.

**Stats:** ~11.9k clips · ~156 GB · ~114 hours · 241 series · 25 technique tags. Higher Sakugabooru score = more notable animation; clips were downloaded best-first.

## Location (not versioned)

```
<workspace>/data/sakugabooru/
├── clips/<series>/            # {post_id}_s{score}.mp4 + {post_id}.json (tags, source, dimensions)
│   └── _state.json            # scraper resume state — do not delete
├── metadata/                  # metadata_{all,train,val,test}.csv + summary.json (built, series-split)
├── pairs/                     # (reserved) keyframe→clip windowed pairs from tools/build_pair_dataset.py
├── cf_cookies.json            # Cloudflare cookies for the scraper (never commit)
└── scrape-logs/
```

`<workspace>` is the directory containing this repo. All paths flow from `configs/paths.env`.

## Metadata CSVs

Built by `python -m native_animation.data.build_metadata` (or `scripts/slurm/build_metadata.sbatch`). Columns: `video,prompt,series,tags,score,clip_id,width,height,source,split`. `video` is relative to `clips/`. The split is **by series** (seed 42) to prevent leakage: 10,632 train / 799 val / 355 test rows at last build. Prompts are templated as `"native animation, anime, <series>, <tag1>, …"`.

## Growing the dataset

- **More clips:** `tools/scrape_sakugabooru.py` (resumable via `_state.json`, dedupes by post id; needs `cf_cookies.json` from `tools/extract_cf_cookies.py`). Rebuild metadata afterward.
- **Windowed pairs:** `tools/build_pair_dataset.py` slides a window over each clip and emits keyframe JPG + short MP4 + `manifest.jsonl` into `pairs/`. Stride controls volume: 0.1 s ≈ 4.3M pairs / 1.7 TB; 0.5 s ≈ 860k / 340 GB; 1.0 s ≈ 430k / 170 GB (non-overlapping).
- **Benchmarks:** frozen evaluation sets belong in `<workspace>/data/benchmarks/` (reserved).
```

- [x] **Step 5: Create `docs/roadmap.md` with:**

```markdown
# Research Roadmap — CVPR 2027

Status: skeleton. Content to be filled in a dedicated planning session.
Target: CVPR 2027 submission (~mid-November 2026).

## Workstreams

1. **Training scale-up** — beyond the single-epoch 5B LoRA run: multi-epoch schedules, LoRA-rank sweep, full fine-tune feasibility, 81-frame training.
2. **Ablations** — leave-one-out over the three method components (scheduler shift, motion weighting, delta consistency) plus sensitivity on `shift`, `alpha`, `lambda`.
3. **Baselines** — untuned Wan2.2-TI2V-5B, Wan2.1-I2V-14B preset, and at least one non-Wan I2V model.
4. **Evaluation** — freeze a benchmark split under `data/benchmarks/`; add FVD and/or a small user study; calibrate DFS thresholds against labeled clips.
5. **Paper** — port `paper/` to the CVPR template; figure pipeline from `experiments/`; related-work refresh.

## Known environment debts

- `open-clip-torch` is not installed in the `comfy` env (the evaluator needs a CLIP backend).
- `experiments/` has no per-run directory convention yet; introduce one alongside a config system.
```

- [x] **Step 6: Create repo-level `CLAUDE.md` with:**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`native-animation` — keyframe-conditioned native-animation video generation with Flow Matching (Native FM), targeting CVPR 2027. A Wan2.2-TI2V-5B backbone is fine-tuned with a project-owned objective; the entire method contribution is `src/native_animation/modeling/native_flowmatch.py` (~170 lines): keyframe-preserving scheduler shift (3.0), anchor-frame clamping (frame 0 clamped clean and excluded from the loss — the clamp IS its supervision), motion-aware frame weighting (`w = 1 + α·normalized latent delta`), and a latent temporal-difference consistency term (`λ=0.25`). `docs/method.md` is the long-form write-up; `paper/` is the paper.

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

`α=0` recovers unweighted MSE exactly (`_weighted_mse` normalizer); anchor frames are sliced out of every loss tensor (`[:, :, anchor_frames:]`); shift-3 Wan sigmas follow `3σ/(1+2σ)`. If you change `native_flowmatch.py` semantics, update `docs/method.md` and `paper/` — they quote these numbers.
```

- [x] **Step 7: String audit**

```bash
cd "$WS/native-animation"
grep -rniE 'course|submission bundle|team contribution' \
  --include='*.py' --include='*.md' --include='*.toml' --include='*.sh' --include='*.sbatch' \
  . | grep -v 'docs/superpowers/'
```

Expected: zero hits (the spec/plan under `docs/superpowers/` are historical records and exempt). Rewrite any straggler in research-neutral language, EXCEPT inside `paper/*.tex` — if the audit hits the paper, flag it to the user instead of editing the paper's prose.

- [x] **Step 8: Commit**

```bash
git add README.md THIRD_PARTY.md docs/ CLAUDE.md
git commit -m "Rewrite docs for the research identity; add dataset notes, roadmap, and agent guide

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Rename the GitHub remote and push

**Files:** none (remote + git config only)

- [x] **Step 1: Rename on GitHub (gh is authed as `eternal-f1ame` with repo scope)**

```bash
cd "$WS/native-animation"
gh repo rename native-animation -R eternal-f1ame/Dynamic-Panel-Animation --yes
```

If this fails, tell the user to rename `Dynamic-Panel-Animation` → `native-animation` in the GitHub UI (Settings → General), then continue.

- [x] **Step 2: Update the local remote and push**

```bash
git remote set-url origin git@github.com:eternal-f1ame/native-animation.git
git remote -v
git push origin main
```

Expected: push succeeds (GitHub redirects old clones, but the explicit set-url keeps things clean).

---

### Task 12: Dissolve FlowMatching/ and index the archive

**Files:**
- Move: `$WS/FlowMatching/anime_keyframe_fm/`, `Project.md`, `progress.md`, `Mid-Project Update.pdf` → `$WS/archive/planning/`; `$WS/FlowMatching/evaluation.py` → `$WS/archive/prototypes/`
- Delete: `$WS/FlowMatching/.claude/`, `$WS/FlowMatching/.codex`, then `$WS/FlowMatching/`
- Create: `$WS/archive/README.md`

- [x] **Step 1: Move and delete**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
mv "$WS/FlowMatching/anime_keyframe_fm" "$WS/archive/planning/anime_keyframe_fm"
mv "$WS/FlowMatching/Project.md" "$WS/archive/planning/Project.md"
mv "$WS/FlowMatching/progress.md" "$WS/archive/planning/progress.md"
mv "$WS/FlowMatching/Mid-Project Update.pdf" "$WS/archive/planning/Mid-Project Update.pdf"
mv "$WS/FlowMatching/evaluation.py" "$WS/archive/prototypes/evaluation.py"
rm -rf "$WS/FlowMatching/.claude" "$WS/FlowMatching/.codex"
rmdir "$WS/FlowMatching"
```

`rmdir` fails loudly on leftovers; if it does, list them, archive anything unexpected, rerun.

- [x] **Step 2: Write `$WS/archive/README.md`**

```markdown
# Archive

Historical record of the course-project era (through Aug 2026). Nothing here is live; the active codebase is `../native-animation/`.

| Path | What it is |
|---|---|
| `planning/` | Pre-pivot planning: `Project.md` (original framing), `progress.md` (session-by-session handoff log — the operational history of every early SLURM job), `anime_keyframe_fm/` (early prototyping workspace), the mid-project update PDF, the original dataset README, the retired scrape sbatch, and `stale-flat-export/` if the old export copies diverged |
| `course-report/` | The compiled course report PDF, the course presentation, misc report assets |
| `runs-2026-04/` | April 2026 run record: baseline demo `outputs/`, SLURM `slurm-logs/`, and the old `dist/` export bundle (provenance twin of the repo's `src/` layout) |
| `prototypes/` | `evaluation.py` — the shared evaluator prototype later absorbed into `src/native_animation/evaluation/evaluate.py` |
| `migration-snapshot.txt` | Workspace listing + disk state immediately before the 2026-08-28 restructure |
```

- [x] **Step 3: Verify**

```bash
! test -e "$WS/FlowMatching" && ls "$WS/archive" && echo OK
```

---

### Task 13: Rewrite the workspace CLAUDE.md

**Files:**
- Modify: `$WS/CLAUDE.md` (full replacement below — the old content is superseded; its operational knowledge now lives in the repo-level CLAUDE.md from Task 10)

- [x] **Step 1: Replace `$WS/CLAUDE.md` with:**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

The research workspace for **Native Animation** — keyframe-conditioned native-animation video generation with Flow Matching, targeting **CVPR 2027**. Restructured 2026-08-28 from an earlier course-project layout (design record: `native-animation/docs/superpowers/specs/2026-08-28-workspace-restructure-design.md`).

| Path | Role |
|---|---|
| `native-animation/` | **The canonical git repo** — all live code, paper, docs, tools. Start at its `CLAUDE.md` and `README.md`; work happens there. |
| `data/sakugabooru/` | Raw clip corpus (`clips/`, ~156 GB, 241 series), built metadata CSVs (`metadata/`), scraper state. Never versioned. |
| `models/` | Shared model-weight cache (Wan/ModelScope trees). The repo's `models` symlink points here. |
| `third_party/` | Read-only: the frozen DiffSynth fork (former implementation host) + reference clones. See its README for each tree's role. Never develop here. |
| `archive/` | Course-era historical record (planning docs, old runs, report). See its README. |

## Ground rules

- All paths flow from `native-animation/configs/paths.env` — source it, never hardcode.
- Anything model-sized or dataset-sized stays out of git and out of the repo tree (the `models` symlink and gitignored `experiments/` are the only touchpoints).
- `third_party/` and `archive/` are frozen: read, copy modules out (recording provenance in the repo's `THIRD_PARTY.md`), but never edit in place.
- Cluster/SLURM conventions and code gotchas live in `native-animation/CLAUDE.md` — read that before running anything.
```

- [x] **Step 2: Verify** — read the file back; confirm it names only paths that exist.

---

### Task 14: Final verification sweep

**Files:** none created. This is the spec's §6 checklist; every step must pass. Fix-forward small issues; STOP and report anything structural.

- [x] **Step 1: Workspace shape**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
ls -A "$WS"    # expect exactly: .claude  CLAUDE.md  archive  data  models  native-animation  third_party
```

- [x] **Step 2: Test suite green**

```bash
cd "$WS/native-animation"
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 \
  /home/aeternum/anaconda3/envs/comfy/bin/python -m pytest
```

- [x] **Step 3: Everything compiles**

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 \
  bash -c 'find src/native_animation tools -name "*.py" -exec /home/aeternum/anaconda3/envs/comfy/bin/python -m py_compile {} + && echo COMPILE-OK'
```

- [x] **Step 4: Metadata regeneration is diff-clean against the moved CSVs**

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 PYTHONPATH=$PWD/src \
  /home/aeternum/anaconda3/envs/comfy/bin/python -m native_animation.data.build_metadata \
  --input-root "$WS/data/sakugabooru/clips" --output-dir /tmp/na_meta_verify --seed 42
for f in metadata_all metadata_train metadata_val metadata_test; do
  diff -q "/tmp/na_meta_verify/$f.csv" "$WS/data/sakugabooru/metadata/$f.csv" || echo "DIFFERS: $f"
done
```

Expected: no `DIFFERS` lines → copy the fresh `summary.json` (its absolute paths now reflect the new layout): `cp /tmp/na_meta_verify/summary.json "$WS/data/sakugabooru/metadata/summary.json"`. If any CSV differs: KEEP the moved CSVs (they are the ground truth the reported numbers used), do not replace anything, and report the diff to the user — the original cluster build may have used non-default flags.

- [x] **Step 5: Real data-path exercise — extract one keyframe**

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1 PYTHONPATH=$PWD/src \
  /home/aeternum/anaconda3/envs/comfy/bin/python -m native_animation.data.extract_keyframes \
  --input-csv "$WS/data/sakugabooru/metadata/metadata_test.csv" \
  --dataset-base-path "$WS/data/sakugabooru/clips" \
  --output-dir /tmp/na_keyframe_verify --limit 1
ls /tmp/na_keyframe_verify/*.png && echo KEYFRAME-OK
```

- [x] **Step 6: No stale-path references anywhere live**

```bash
grep -rn 'FlowMatching\|/Anime/\|course_flowmatch\|course_project' \
  --include='*.py' --include='*.sh' --include='*.sbatch' --include='*.env' --include='*.toml' --include='*.md' \
  "$WS/native-animation" | grep -v 'docs/superpowers/' 
grep -rniE 'course' --include='*.py' --include='*.md' --include='*.toml' --include='*.sh' --include='*.sbatch' \
  "$WS/native-animation" | grep -v 'docs/superpowers/'
```

Expected: zero output from both.

- [x] **Step 7: Git end state**

```bash
cd "$WS/native-animation" && git status --short && git log --oneline -12 && git remote -v
```

Expected: clean status; the phase commits present; remote URL `git@github.com:eternal-f1ame/native-animation.git`; branch pushed.

- [x] **Step 8: Report + offer the optional cluster check**

Summarize results to the user and offer (do not auto-submit): `sbatch scripts/slurm/env_smoke_test.sbatch` as the final GPU-node validation, and note the two roadmap debts (`open-clip-torch` missing; CVPR template port pending).

---

## Self-review record

- **Spec coverage:** Phases 0–9 → Tasks 1–14 one-to-one (P0+P1→T1, P2→T2, P3→T3, P4→T4, P5→T5, P6.1–6.2→T6, P6.3–6.5+6.10→T7, P6.6→T8, P6.7→T9, P6.8–6.9→T10, P6.11→T11, P7→T12, P8→T13, P9→T14). Spec §5 CSV-regen verification → T14.4; §6 checklist → T14 steps 1–8 plus T14.8's smoke-test offer.
- **Placeholders:** none — every created file's content is inline; every edit has exact old/new text.
- **Consistency:** env var names (`DATA_ROOT/METADATA_DIR/MODELS_ROOT/EXPERIMENTS_ROOT`) identical across T7 scripts, T10 docs, T13; `experiments/` paths consistent everywhere; tools staging path in T3 matches T8; diff targets in T6 match what T4 produced.
