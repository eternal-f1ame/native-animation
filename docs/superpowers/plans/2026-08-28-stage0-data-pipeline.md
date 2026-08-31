# Stage-0 Data Pipeline Implementation Plan (v2 Plan 1 of 3)

> **Status (2026-08-30): EXECUTED with amendments** — object-lean pack/squash redesign (see `docs/dataset.md`), annotation validated (Qwen3-VL, ~8s/shot). Corpus run in flight. **Pending:** T11–T12 (v2 metadata rebuild + benchmark freeze) once the corpus lands; delta scrape held for packed output.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the merged ~180k-post corpus into training-ready data for Native Animation v2: single-shot clips, curation verdicts, motion profiles, Qwen3-VL directives, quality tiers, v2 metadata, and a frozen benchmark.

**Architecture:** Pure-logic cores in `src/native_animation/data/` (TDD, CPU-only tests), thin CLIs in `tools/`, all heavy work as resumable SLURM jobs. Every artifact is a file under `$WS/data/sakugabooru/` keyed by post/shot IDs; the v2 metadata builder joins them at the end. Nothing here touches the model code (Plan 2) or eval (Plan 3).

**Tech Stack:** Python 3.11 (`comfy` env for CPU/cv2 work; a NEW isolated `anno` env for Qwen3-VL), ffmpeg via `imageio_ffmpeg`, OpenCV (Farneback flow), PySceneDetect, vLLM or transformers for annotation, SLURM.

**Spec:** `docs/superpowers/specs/2026-08-28-native-fm-v2-design.md` §2 Stage 0 (+ §0 data context). Plans 2–3 (method/training, evaluation) follow this one; the long-running jobs launched here overlap with Plan-2 coding.

## Global Constraints

- `WS=/home/aeternum/Research/Comic/Cartoon`, `COMFY_PY=/home/aeternum/anaconda3/envs/comfy/bin/python`. Every python invocation prefixed `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1`.
- Heavy IO/compute NEVER on the login node — SLURM only (the login node is currently IO-starved by the 1.27 TB download; even imports time out).
- The `comfy` env is delicate: no upgrades to torch/transformers there. Annotation gets its own env (Task 2).
- Framing rule: docs and naming lead with native animation, never flow-matching branding.
- All jobs checkpoint/resume (idempotent by output-file existence). SLURM conventions from repo `CLAUDE.md`: `--export=ALL`, thread caps exported in the login shell, ≥ `1-00:00:00` wall time for GPU jobs.
- Sakugabooru is pre-curated by community — the curation cascade here is deliberately minimal (duration/decode/static/blur only; no OCR/dedup — justified in spec §2 Stage 0 vs AniMatrix's raw-episode cascade).
- Derived shots are re-encoded (libx264 CRF 16, yuv420p, downscale-only to short side 480, source fps kept) — normalizes training IO and caps derived storage at ~1 TB against 47 TB free.

---

### Task 1: Acquisition gate — confirm the merge chain, run the delta scrape

**Files:** none created (operational gate). The extractor and scraper already exist and are tested.

**Interfaces:**
- Consumes: `$WS/data/sakugabooru/extraction-summary.json` (from the in-flight babysitter chain, task `bkb28bzgy`), `tools/scrape_sakugabooru.py`.
- Produces: the merged corpus in `$WS/data/sakugabooru/clips/` (snapshot + delta + originals) and the fact set (final post count, max post ID) later tasks rely on.

- [ ] **Step 1: Wait for / verify the babysitter chain**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
tail -6 "$WS/data/sakugabooru/extraction.log" 2>/dev/null
cat "$WS/data/sakugabooru/extraction-summary.json"
```

Expected: summary exists with `extracted` ≈ 140–155k, `max_post_id` ≈ 273k, `tars_processed` ≈ number of train tars. If the chain failed at the completeness or test-gate step, rerun `bash $WS/data/sakugabooru/babysit-and-extract.sh` (idempotent). Do not proceed until the summary exists.

- [ ] **Step 2: Launch the delta scrape (SLURM, short partition — compute nodes have internet)**

Write `$WS/native-animation/scripts/slurm/delta_scrape.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=na-delta-scrape
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
MAX_ID=$(python -c "import json;print(json.load(open('$WORKSPACE_ROOT/data/sakugabooru/extraction-summary.json'))['max_post_id'])")
python tools/scrape_sakugabooru.py --anime "id:$((MAX_ID+1))..400000" --limit 999999 --min-score 0
```

Submit: `cd $WS/native-animation && sbatch scripts/slurm/delta_scrape.sbatch`. (The `id:A..B` pseudo-tag path and state-dedup were dry-run-validated on 2026-08-28; the scraper resumes safely if the job is resubmitted.)

- [ ] **Step 3: Verify the merged corpus once both finish**

```bash
WS=/home/aeternum/Research/Comic/Cartoon
find "$WS/data/sakugabooru/clips" -name '*.mp4' -o -name '*.webm' | wc -l   # expect ~165–185k
env PYTHONUTF8=1 /home/aeternum/anaconda3/envs/comfy/bin/python -c "
import json; s=json.load(open('$WS/data/sakugabooru/clips/_state.json'))
print('state ids:', len(s['downloaded_ids']))"
```

Record both numbers in the task report. This task can sit "in progress" while Tasks 2–5 (env + pure code) proceed — only Task 6 onward needs the merged corpus.

---

### Task 2: Annotation environment (`anno`) — isolated, verified

**Files:**
- Create: `$WS/native-animation/configs/anno-env.md` (exact create/verify commands, for reproducibility)

**Interfaces:**
- Produces: conda env `anno` with a Qwen3-VL-capable stack; `ANNO_PY=/home/aeternum/anaconda3/envs/anno/bin/python`. Task 9 depends on it.

- [ ] **Step 1: Create the env (new, so the delicate `comfy` stack is untouched)**

```bash
conda create -y -n anno python=3.11
/home/aeternum/anaconda3/envs/anno/bin/python -m pip install "vllm>=0.11" qwen-vl-utils pillow imageio imageio-ffmpeg
```

If the vLLM install fails on this cluster's CUDA, fall back to `pip install "transformers>=4.57" accelerate torch pillow qwen-vl-utils imageio imageio-ffmpeg` (transformers-only path; slower but sufficient).

- [ ] **Step 2: Verify model + video-input support on a GPU node (env smoke, NOT login node)**

Submit a one-off: `srun --partition=gpu --gres=gpu:1 --constraint='gmem48|gmem80' --time=1:00:00 --export=ALL` running:

```bash
/home/aeternum/anaconda3/envs/anno/bin/python - <<'EOF'
from vllm import LLM, SamplingParams   # or transformers fallback
llm = LLM(model="Qwen/Qwen3-VL-8B-Instruct", max_model_len=8192, limit_mm_per_prompt={"video": 1})
print("MODEL-OK")
EOF
```

Expected: `MODEL-OK` (first run downloads ~17 GB into the HF cache — set `HF_HOME=$WS/models/hf-cache` so it lands in the shared cache). If `Qwen/Qwen3-VL-8B-Instruct` is not the exact HF id, list `Qwen/Qwen3-VL*` on HF and pick the ~8B instruct variant; record the chosen id in `configs/anno-env.md` and `configs/stage0.yaml` (Task 3).

- [ ] **Step 3: Write `configs/anno-env.md`** documenting: env name, exact pip set actually installed, model id, `HF_HOME`, and the fallback path taken (if any). Commit.

```bash
git add configs/anno-env.md && git commit -m "Document annotation environment setup"
```

---

### Task 3: Stage-0 config + shots window math

**Files:**
- Create: `src/native_animation/data/shots.py`, `configs/stage0.yaml`
- Test: `tests/test_shots.py`

**Interfaces:**
- Produces: `plan_shot_windows(scene_list_s: list[tuple[float,float]], min_s=2.2, max_s=10.0) -> list[tuple[float,float]]` (pure); `Shot` record semantics: id `f"{post_id}_{index:02d}"`; `tools/split_shots.py` (Task 6) consumes these. `configs/stage0.yaml` is the single knob file every Stage-0 CLI reads.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shots.py
"""Shot windowing: scene list -> extractable [start, end) windows."""
from native_animation.data.shots import plan_shot_windows


def test_short_scenes_are_dropped():
    assert plan_shot_windows([(0.0, 1.5)]) == []          # < 2.2 s
    assert plan_shot_windows([(0.0, 2.4)]) == [(0.0, 2.4)]


def test_long_scene_is_tiled_without_overlap_and_remainder_kept_if_long_enough():
    windows = plan_shot_windows([(0.0, 23.0)], min_s=2.2, max_s=10.0)
    assert windows == [(0.0, 10.0), (10.0, 20.0), (20.0, 23.0)]  # 3.0 s remainder kept


def test_short_remainder_is_dropped():
    windows = plan_shot_windows([(0.0, 21.0)], min_s=2.2, max_s=10.0)
    assert windows == [(0.0, 10.0), (10.0, 20.0)]          # 1.0 s remainder dropped


def test_multiple_scenes_concatenate_in_order():
    windows = plan_shot_windows([(0.0, 3.0), (3.0, 4.0), (4.0, 9.5)])
    assert windows == [(0.0, 3.0), (4.0, 9.5)]             # middle scene too short
```

- [ ] **Step 2: Run to verify failure** — `cd $WS/native-animation && env <caps> $COMFY_PY -m pytest tests/test_shots.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `src/native_animation/data/shots.py`**

```python
"""Shot windowing for the Stage-0 pipeline.

Scene boundaries come from PySceneDetect (in tools/split_shots.py); this module
holds the pure windowing math so it is testable without video IO.
"""
from __future__ import annotations


def plan_shot_windows(
    scene_list_s: list[tuple[float, float]],
    min_s: float = 2.2,
    max_s: float = 10.0,
) -> list[tuple[float, float]]:
    """Turn detected scenes into training-clip windows.

    Scenes shorter than ``min_s`` (cannot supply 49 raw frames at 24 fps) are
    dropped; scenes longer than ``max_s`` are tiled without overlap and a
    trailing remainder is kept only if it is itself >= ``min_s``.
    """
    windows: list[tuple[float, float]] = []
    for start, end in scene_list_s:
        pos = start
        while end - pos >= min_s:
            length = min(max_s, end - pos)
            # Avoid leaving an unusably short tail: if the tail after this
            # window would be positive but < min_s, absorb nothing — drop it.
            windows.append((round(pos, 3), round(pos + length, 3)))
            pos += length
    return windows
```

- [ ] **Step 4: Run to verify pass**, then check the whole suite stays green (`pytest -q`).

- [ ] **Step 5: Write `configs/stage0.yaml`**

```yaml
# Stage-0 pipeline knobs (single source; every Stage-0 CLI reads this).
shots:
  min_seconds: 2.2
  max_seconds: 10.0
  detector_threshold: 27.0        # PySceneDetect ContentDetector default region
  reencode: {crf: 16, short_side_max: 480, pix_fmt: yuv420p}
curation:
  min_mean_framediff: 1.0         # static-clip floor (uint8 mean abs diff, 256px)
  min_laplacian_var: 20.0         # blur floor, anime-calibrated in Task 6 audit
profiling:
  frame_pairs: 8                  # sampled pairs per shot
  flow_size: 256                  # short side for Farneback
  quantiles: 4                    # Q buckets for q_m / q_d
annotation:
  model_id: Qwen/Qwen3-VL-8B-Instruct   # confirm in Task 2; update if different
  frames_per_shot: 8
  max_retries: 2
  shards: 64                      # SLURM array width
tiers: {s_quantile: 0.95, a_quantile: 0.70}   # by post score; favorites tiebreak
```

- [ ] **Step 6: Commit** — `git add src/native_animation/data/shots.py tests/test_shots.py configs/stage0.yaml && git commit -m "Add shot windowing math and Stage-0 config"`

---

### Task 4: Curation verdicts (static/blur/decode) — pure scoring + records

**Files:**
- Create: `src/native_animation/data/curation.py`
- Test: `tests/test_curation.py`

**Interfaces:**
- Produces: `static_score(frames: list[np.ndarray]) -> float` (mean abs frame diff, grayscale), `blur_score(frames) -> float` (median Laplacian variance), `curation_verdict(static: float, blur: float, duration_s: float, cfg: dict) -> dict` returning `{"pass": bool, "reasons": [str, ...]}`. `tools/curate_shots.py` (Task 6) consumes these; the metadata builder (Task 10) consumes the verdict files.

- [ ] **Step 1: Failing tests**

```python
# tests/test_curation.py
import numpy as np
from native_animation.data.curation import blur_score, curation_verdict, static_score

CFG = {"min_mean_framediff": 1.0, "min_laplacian_var": 20.0}


def _frames(n=4, value=128, jitter=0):
    rng = np.random.default_rng(0)
    return [np.clip(value + (rng.integers(-jitter, jitter + 1, (64, 64)) if jitter else 0), 0, 255).astype(np.uint8) for _ in range(n)]


def test_static_score_zero_for_identical_frames():
    assert static_score(_frames()) == 0.0
    assert static_score(_frames(jitter=30)) > 1.0


def test_blur_score_orders_sharp_above_flat():
    flat = _frames()
    checker = [np.indices((64, 64)).sum(0).astype(np.uint8) % 2 * 255] * 4
    assert blur_score(checker) > blur_score(flat)


def test_verdict_reasons():
    v = curation_verdict(static=0.1, blur=100.0, duration_s=5.0, cfg=CFG)
    assert v == {"pass": False, "reasons": ["static"]}
    v = curation_verdict(static=5.0, blur=5.0, duration_s=5.0, cfg=CFG)
    assert v == {"pass": False, "reasons": ["blur"]}
    assert curation_verdict(5.0, 100.0, 5.0, CFG) == {"pass": True, "reasons": []}
```

- [ ] **Step 2: Verify FAIL.**

- [ ] **Step 3: Implement**

```python
# src/native_animation/data/curation.py
"""Minimal curation scoring for pre-curated booru shots (spec §2 Stage 0)."""
from __future__ import annotations

import cv2
import numpy as np


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def static_score(frames: list[np.ndarray]) -> float:
    """Mean absolute inter-frame difference; ~0 for hold-only clips."""
    grays = [_gray(f).astype(np.float32) for f in frames]
    if len(grays) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(b - a)) for a, b in zip(grays, grays[1:])]))


def blur_score(frames: list[np.ndarray]) -> float:
    """Median Laplacian variance across frames; low = soft/blurry."""
    return float(np.median([cv2.Laplacian(_gray(f), cv2.CV_64F).var() for f in frames]))


def curation_verdict(static: float, blur: float, duration_s: float, cfg: dict) -> dict:
    reasons = []
    if static < cfg["min_mean_framediff"]:
        reasons.append("static")
    if blur < cfg["min_laplacian_var"]:
        reasons.append("blur")
    if duration_s <= 0:
        reasons.append("decode")
    return {"pass": not reasons, "reasons": reasons}
```

- [ ] **Step 4: Verify PASS + suite green.** **Step 5: Commit** (`"Add curation scoring"`).

---

### Task 5: Motion profiling — flow energy + non-rigid residual (the curriculum's fuel)

**Files:**
- Create: `src/native_animation/data/profiling.py`
- Test: `tests/test_profiling.py`

**Interfaces:**
- Produces: `flow_energy(flow: np.ndarray) -> float` (mean L2 magnitude, HxWx2 input); `nonrigid_residual(flow: np.ndarray) -> float` (mean magnitude of flow minus its least-squares affine fit — camera/global motion removed, deformation kept); `assign_quantile_buckets(values: list[float], q: int) -> list[int]` (1..q, ties safe). Task 7's CLI computes Farneback flow and calls these; Task 10 joins `q_m`, `q_d` into metadata; Plan 2's curriculum consumes the buckets.

- [ ] **Step 1: Failing tests**

```python
# tests/test_profiling.py
import numpy as np
from native_animation.data.profiling import assign_quantile_buckets, flow_energy, nonrigid_residual


def _affine_flow(h=32, w=32, a=0.1, b=0.05, tx=2.0, ty=-1.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([a * xs + b * ys + tx, -b * xs + a * ys + ty], axis=-1)


def test_flow_energy_zero_and_positive():
    assert flow_energy(np.zeros((8, 8, 2))) == 0.0
    assert flow_energy(np.full((8, 8, 2), 3.0)) > 0


def test_pure_affine_flow_has_near_zero_residual():
    # Camera pans/zooms are affine — the residual must ignore them.
    assert nonrigid_residual(_affine_flow()) < 1e-8


def test_localized_deformation_survives_affine_removal():
    flow = _affine_flow()
    flow[10:20, 10:20] += 5.0                    # a smear-like local deformation
    assert nonrigid_residual(flow) > 0.5


def test_quantile_buckets_cover_range():
    buckets = assign_quantile_buckets(list(range(100)), q=4)
    assert min(buckets) == 1 and max(buckets) == 4
    assert buckets[0] == 1 and buckets[-1] == 4
    assert assign_quantile_buckets([5.0, 5.0, 5.0], q=4) == [1, 1, 1]  # ties collapse low
```

- [ ] **Step 2: Verify FAIL.**

- [ ] **Step 3: Implement**

```python
# src/native_animation/data/profiling.py
"""Motion profiling: flow energy (amplitude) and non-rigid residual (deformation).

The residual removes the best-fit global affine field — pans, zooms, and
rotations — so what remains is squash/stretch/smear-style deformation, the
quantity the v2 curriculum schedules on (spec §1/§2).
"""
from __future__ import annotations

import numpy as np


def flow_energy(flow: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(flow, axis=-1)))


def nonrigid_residual(flow: np.ndarray) -> float:
    h, w, _ = flow.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    ones = np.ones_like(xs)
    basis = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=1)      # N x 3
    targets = flow.reshape(-1, 2)                                          # N x 2
    coeffs, *_ = np.linalg.lstsq(basis, targets, rcond=None)               # 3 x 2
    residual = targets - basis @ coeffs
    return float(np.mean(np.linalg.norm(residual, axis=1)))


def assign_quantile_buckets(values: list[float], q: int) -> list[int]:
    arr = np.asarray(values, dtype=np.float64)
    edges = np.quantile(arr, [i / q for i in range(1, q)])
    return [int(np.searchsorted(edges, v, side="right")) + 1 for v in arr]
```

- [ ] **Step 4: Verify PASS + suite green.** **Step 5: Commit** (`"Add motion profiling: flow energy, non-rigid residual, quantile buckets"`).

---

### Task 6: Shot-splitting + curation CLI and SLURM array

**Files:**
- Create: `tools/split_shots.py`, `scripts/slurm/split_shots.sbatch`
- Modify: none

**Interfaces:**
- Consumes: `plan_shot_windows`, `static_score`, `blur_score`, `curation_verdict`, `configs/stage0.yaml`, merged corpus from Task 1.
- Produces: `$WS/data/sakugabooru/shots/<series>/<post_id>_<idx:02d>.mp4` re-encoded shots + per-shard manifest JSONL at `$WS/data/sakugabooru/shots/manifests/shard_{k:04d}.jsonl`, one record per shot: `{"shot_id": "<post>_<idx>", "post_id": int, "series": str, "video": "shots-relative path", "start_s": float, "end_s": float, "fps": float, "curation": {"pass": bool, "reasons": []}, "static": float, "blur": float}`.

- [ ] **Step 1: Install PySceneDetect into comfy (small, pure-python + cv2 which we have)**

```bash
$COMFY_PY -m pip install scenedetect
$COMFY_PY -c "import scenedetect; print(scenedetect.__version__)"
```

- [ ] **Step 2: Write `tools/split_shots.py`** (full body):

```python
#!/usr/bin/env python3
"""Split corpus posts into single-shot re-encoded clips + curation verdicts.

Sharded by post-id modulo --num-shards so it runs as a SLURM array; each shard
is idempotent (skips shots whose output file already exists) and writes its own
manifest JSONL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.curation import blur_score, curation_verdict, static_score  # noqa: E402
from native_animation.data.shots import plan_shot_windows  # noqa: E402

from scenedetect import ContentDetector, detect  # noqa: E402


def sample_frames(path: Path, n: int = 6) -> list:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for idx in [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, (256, 256)))
    cap.release()
    return frames


def reencode(src: Path, dst: Path, start: float, end: float, cfg: dict) -> bool:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = f"scale='if(gt(iw,ih),-2,min({cfg['short_side_max']},iw))':'if(gt(iw,ih),min({cfg['short_side_max']},ih),-2)'"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-vf", vf, "-an", "-c:v", "libx264", "-crf", str(cfg["crf"]),
           "-pix_fmt", cfg["pix_fmt"], "-threads", "2", str(dst)]
    return subprocess.run(cmd, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    scfg, ccfg = cfg["shots"], cfg["curation"]
    manifest_dir = args.out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"shard_{args.shard:04d}.jsonl"
    done_ids = set()
    if manifest_path.exists():
        with manifest_path.open() as handle:
            done_ids = {json.loads(line)["shot_id"] for line in handle if line.strip()}

    with manifest_path.open("a") as manifest:
        for sidecar in sorted(args.clips_dir.rglob("*.json")):
            if sidecar.name == "_state.json" or not sidecar.stem.isdigit():
                continue
            post_id = int(sidecar.stem)
            if post_id % args.num_shards != args.shard:
                continue
            videos = list(sidecar.parent.glob(f"{post_id}_s*.mp4")) + list(sidecar.parent.glob(f"{post_id}_s*.webm"))
            if not videos:
                continue
            src = videos[0]
            series = sidecar.parent.name
            try:
                scenes = detect(str(src), ContentDetector(threshold=scfg["detector_threshold"]))
                scene_list = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
                if not scene_list:  # single-shot post: whole duration
                    cap = cv2.VideoCapture(str(src))
                    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 24)
                    cap.release()
                    scene_list = [(0.0, float(dur))]
            except Exception:
                continue
            for idx, (start, end) in enumerate(plan_shot_windows(scene_list, scfg["min_seconds"], scfg["max_seconds"])):
                shot_id = f"{post_id}_{idx:02d}"
                if shot_id in done_ids:
                    continue
                out = args.out_dir / series / f"{shot_id}.mp4"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists() and not reencode(src, out, start, end, scfg["reencode"]):
                    continue
                frames = sample_frames(out)
                if not frames:
                    continue
                cap = cv2.VideoCapture(str(out)); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; cap.release()
                st, bl = static_score(frames), blur_score(frames)
                record = {"shot_id": shot_id, "post_id": post_id, "series": series,
                          "video": str(out.relative_to(args.out_dir)), "start_s": start, "end_s": end,
                          "fps": fps, "static": st, "blur": bl,
                          "curation": curation_verdict(st, bl, end - start, ccfg)}
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
    print(f"[shard {args.shard}] done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `scripts/slurm/split_shots.sbatch`** (array job):

```bash
#!/bin/bash
#SBATCH --job-name=na-split-shots
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-63
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python tools/split_shots.py \
  --clips-dir "$WORKSPACE_ROOT/data/sakugabooru/clips" \
  --out-dir "$WORKSPACE_ROOT/data/sakugabooru/shots" \
  --config configs/stage0.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 64
```

- [ ] **Step 4: Smoke on 1 shard locally-scoped** — run `tools/split_shots.py --shard 0 --num-shards 20000` (tiny slice) via `srun` on short partition; inspect 3 output shots by eye (`ffprobe` duration/resolution) and the manifest lines. Then **audit thresholds**: from the shard manifest, print the static/blur score distribution and eyeball 5 near-threshold clips; adjust `configs/stage0.yaml` floors if anime-miscalibrated (record the decision in the commit message).

- [ ] **Step 5: Submit the full array** (`sbatch scripts/slurm/split_shots.sbatch`) — runs for hours across 64 CPU tasks; later tasks proceed meanwhile. **Step 6: Commit** tools+sbatch (`"Add shot-splitting pipeline (CLI + SLURM array)"`).

---

### Task 7: Profiling CLI + run

**Files:**
- Create: `tools/profile_motion.py`, `scripts/slurm/profile_motion.sbatch`

**Interfaces:**
- Consumes: shots + manifests (Task 6), `flow_energy` / `nonrigid_residual` (Task 5).
- Produces: `$WS/data/sakugabooru/profiles/shard_{k:04d}.jsonl`: `{"shot_id": str, "flow_energy": float, "nonrigid_residual": float}`. Task 10 computes corpus-wide quantiles from the union.

- [ ] **Step 1: Write `tools/profile_motion.py`** — same shard/idempotency pattern as Task 6 (read shard manifest, skip ids already in the profile shard). Per shot: decode `cfg["profiling"]["frame_pairs"]` evenly spaced consecutive-frame pairs at `flow_size` short side, `cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)`, record mean `flow_energy` and mean `nonrigid_residual` across pairs:

```python
#!/usr/bin/env python3
"""Per-shot motion profiling: Farneback flow energy + non-rigid residual."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.profiling import flow_energy, nonrigid_residual  # noqa: E402


def pairs_from(path: Path, n_pairs: int, short_side: int):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release(); return
    for start in np.linspace(0, total - 2, min(n_pairs, total - 1), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        ok1, a = cap.read(); ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        h, w = a.shape[:2]
        scale = short_side / min(h, w)
        size = (int(w * scale), int(h * scale))
        yield (cv2.cvtColor(cv2.resize(a, size), cv2.COLOR_BGR2GRAY),
               cv2.cvtColor(cv2.resize(b, size), cv2.COLOR_BGR2GRAY))
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()

    pcfg = yaml.safe_load(args.config.read_text())["profiling"]
    manifest = args.shots_dir / "manifests" / f"shard_{args.shard:04d}.jsonl"
    if not manifest.exists():
        print("no manifest shard; nothing to do"); return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"shard_{args.shard:04d}.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as handle:
            done = {json.loads(l)["shot_id"] for l in handle if l.strip()}
    with manifest.open() as src, out_path.open("a") as out:
        for line in src:
            rec = json.loads(line)
            if rec["shot_id"] in done or not rec["curation"]["pass"]:
                continue
            energies, residuals = [], []
            for prev, nxt in pairs_from(args.shots_dir / rec["video"], pcfg["frame_pairs"], pcfg["flow_size"]):
                flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                energies.append(flow_energy(flow)); residuals.append(nonrigid_residual(flow))
            if energies:
                out.write(json.dumps({"shot_id": rec["shot_id"],
                                      "flow_energy": float(np.mean(energies)),
                                      "nonrigid_residual": float(np.mean(residuals))}) + "\n")
                out.flush()
    print(f"[shard {args.shard}] profiled")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `scripts/slurm/profile_motion.sbatch`** — copy Task 6's sbatch shape (array 0-63, short partition, 4 CPU) with job name `na-profile`, invoking `tools/profile_motion.py --shots-dir $WORKSPACE_ROOT/data/sakugabooru/shots --out-dir $WORKSPACE_ROOT/data/sakugabooru/profiles --config configs/stage0.yaml --shard $SLURM_ARRAY_TASK_ID`.

- [ ] **Step 3: Smoke one shard via srun; submit array after Task 6's array is ≥ half done** (profiles read shot files; array shards tolerate missing manifests). **Step 4: Commit** (`"Add motion-profiling CLI + SLURM array"`).

---

### Task 8: Caption schema + prompt construction (pure logic)

**Files:**
- Create: `src/native_animation/data/captions.py`
- Test: `tests/test_captions.py`

**Interfaces:**
- Produces: `CAPTION_SYSTEM_PROMPT: str` and `build_caption_request(tags: str, series: str) -> str` (the exact instruction sent to Qwen3-VL); `parse_caption_output(text: str) -> dict | None` (extracts+validates `{"structured": {...}, "tag": str, "summary": str, "description": str}` from the model's output; None on unrecoverable parse failure); `fallback_caption(tags: str, series: str) -> dict` (tag-template directive when the model fails). Task 9's CLI consumes all three; Task 10 consumes the output records.

- [ ] **Step 1: Failing tests**

```python
# tests/test_captions.py
from native_animation.data.captions import (
    build_caption_request,
    fallback_caption,
    parse_caption_output,
)

GOOD = """Here is the annotation:
```json
{"subjects": [{"type": "Human", "appearance": "girl in red coat"}],
 "motion": [{"action": "runs left to right", "amplitude": "high"}],
 "AnimeVisualEffects": {"present": true, "effects": ["speed lines"]},
 "style": {"VideoStyle": "2D Japanese Anime", "MotionStyle": "2D Combat"},
 "camera": {"shot_type": "full shot", "camera_motion": "tracking"},
 "environment": "night rooftop"}
```
<tag> VideoStyle: 2D Japanese Anime, MotionStyle: 2D Combat, shot_type: full shot, camera_motion: tracking
<summary> A girl in a red coat sprints across a night rooftop.
<description> A girl in a red coat sprints from left to right across a rooftop at night, trailed by speed lines while the camera tracks her at full shot."""


def test_parse_extracts_all_four_parts():
    out = parse_caption_output(GOOD)
    assert out["structured"]["style"]["VideoStyle"] == "2D Japanese Anime"
    assert out["tag"].startswith("VideoStyle:")
    assert out["summary"].startswith("A girl")
    assert "speed lines" in out["description"]


def test_parse_rejects_missing_sections():
    assert parse_caption_output("no json here") is None
    assert parse_caption_output(GOOD.split("<summary>")[0]) is None


def test_request_embeds_source_tags_and_series():
    req = build_caption_request(tags="smears fighting", series="one_piece")
    assert "smears fighting" in req and "one_piece" in req


def test_fallback_is_a_usable_directive():
    fb = fallback_caption(tags="smears fighting effects", series="bleach_series")
    assert fb["summary"] and "bleach_series" in fb["description"]
    assert fb["structured"] == {}
```

- [ ] **Step 2: Verify FAIL.**

- [ ] **Step 3: Implement `src/native_animation/data/captions.py`**

```python
"""Caption schema, prompts, and parsing for Qwen3-VL shot annotation.

Follows the AniMatrix three-section directive format (<tag>/<summary>/
<description>) plus a structured JSON block, produced in a single pass.
Only text logic lives here; model inference is tools/annotate_clips.py.
"""
from __future__ import annotations

import json
import re

CAPTION_SYSTEM_PROMPT = (
    "You are a professional anime production annotator. Given a short animation "
    "clip, output BOTH of the following, exactly in this order:\n"
    "1. A JSON code block with keys: subjects (list of {type, appearance}), "
    "motion (list of {action, amplitude}), AnimeVisualEffects ({present, effects}), "
    "style ({VideoStyle, MotionStyle}), camera ({shot_type, camera_motion}), "
    "environment (string).\n"
    "2. Three tagged lines:\n"
    "<tag> comma-separated key: value pairs for VideoStyle, MotionStyle, shot_type, camera_motion\n"
    "<summary> one sentence summarizing the clip\n"
    "<description> one detailed paragraph, temporally ordered: subject appearance -> "
    "camera -> motion and expression -> visual effects -> environment. Use professional "
    "animation vocabulary (smears, impact frames, speed lines, held frames) where they "
    "genuinely appear. Describe production choices, not just visible content."
)

_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_SECTION_RE = {
    "tag": re.compile(r"<tag>\s*(.+)"),
    "summary": re.compile(r"<summary>\s*(.+)"),
    "description": re.compile(r"<description>\s*(.+?)(?:\n<|\Z)", re.DOTALL),
}


def build_caption_request(tags: str, series: str) -> str:
    return (
        f"Community tags for this clip (noisy but grounding): {tags}\n"
        f"Series: {series}\n"
        "Annotate the clip per the required format."
    )


def parse_caption_output(text: str) -> dict | None:
    json_match = _JSON_RE.search(text)
    if not json_match:
        return None
    try:
        structured = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None
    sections = {}
    for key, pattern in _SECTION_RE.items():
        match = pattern.search(text)
        if not match:
            return None
        sections[key] = match.group(1).strip()
    return {"structured": structured, **sections}


def fallback_caption(tags: str, series: str) -> dict:
    tag_list = ", ".join(tags.split()[:12])
    return {
        "structured": {},
        "tag": "",
        "summary": f"An anime clip featuring {tag_list}.",
        "description": f"native animation, anime, {series}, {tag_list}",
    }
```

- [ ] **Step 4: Verify PASS + suite green.** **Step 5: Commit** (`"Add caption schema, prompts, and parsing"`).

---

### Task 9: Annotation CLI + SLURM array (GPU)

**Files:**
- Create: `tools/annotate_clips.py`, `scripts/slurm/annotate.sbatch`

**Interfaces:**
- Consumes: `anno` env + model id (Task 2), captions module (Task 8), shots + manifests (Task 6), sidecar tags (post-level, joined by `post_id`).
- Produces: `$WS/data/sakugabooru/captions/shard_{k:04d}.jsonl`: `{"shot_id": str, "tag": str, "summary": str, "description": str, "structured": {...}, "fallback": bool}`.

- [ ] **Step 1: Write `tools/annotate_clips.py`** — vLLM path (transformers fallback documented in the file header): load model once; for each manifest record (curation-pass only, idempotent skip), sample `frames_per_shot` frames as a video input, chat with `CAPTION_SYSTEM_PROMPT` + `build_caption_request(tags, series)`; `parse_caption_output`; up to `max_retries` regenerations on parse failure, then `fallback_caption` with `"fallback": true`:

```python
#!/usr/bin/env python3
"""Qwen3-VL shot annotation: structured caption + three-section directive.

Runs under the `anno` env (see configs/anno-env.md). Sharded + idempotent like
the other Stage-0 CLIs. Transformers fallback: replace the vLLM block with
AutoModelForImageTextToText.from_pretrained(model_id, dtype="bfloat16",
device_map="cuda") and processor.apply_chat_template on the same messages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.captions import (  # noqa: E402
    CAPTION_SYSTEM_PROMPT,
    build_caption_request,
    fallback_caption,
    parse_caption_output,
)


def load_post_tags(clips_dir: Path) -> dict[int, tuple[str, str]]:
    """post_id -> (flat tags, series) from the corpus sidecars."""
    table = {}
    for sidecar in clips_dir.rglob("*.json"):
        if sidecar.name == "_state.json" or not sidecar.stem.isdigit():
            continue
        try:
            data = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        table[int(sidecar.stem)] = (data.get("tags", ""), sidecar.parent.name)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-dir", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()

    acfg = yaml.safe_load(args.config.read_text())["annotation"]
    manifest = args.shots_dir / "manifests" / f"shard_{args.shard:04d}.jsonl"
    if not manifest.exists():
        print("no manifest shard"); return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"shard_{args.shard:04d}.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as handle:
            done = {json.loads(l)["shot_id"] for l in handle if l.strip()}

    from vllm import LLM, SamplingParams
    llm = LLM(model=acfg["model_id"], max_model_len=8192, limit_mm_per_prompt={"video": 1})
    params = SamplingParams(temperature=0.2, max_tokens=1024)
    tags_table = load_post_tags(args.clips_dir)

    with manifest.open() as src, out_path.open("a") as out:
        for line in src:
            rec = json.loads(line)
            if rec["shot_id"] in done or not rec["curation"]["pass"]:
                continue
            tags, series = tags_table.get(rec["post_id"], ("", rec["series"]))
            video_path = str(args.shots_dir / rec["video"])
            messages = [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "video", "video": video_path, "num_frames": acfg["frames_per_shot"]},
                    {"type": "text", "text": build_caption_request(tags, series)},
                ]},
            ]
            caption, fallback = None, False
            for _ in range(acfg["max_retries"] + 1):
                result = llm.chat(messages, params)[0].outputs[0].text
                caption = parse_caption_output(result)
                if caption:
                    break
            if caption is None:
                caption, fallback = fallback_caption(tags, series), True
            out.write(json.dumps({"shot_id": rec["shot_id"], **caption, "fallback": fallback}) + "\n")
            out.flush()
    print(f"[shard {args.shard}] annotated")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `scripts/slurm/annotate.sbatch`** — GPU array:

```bash
#!/bin/bash
#SBATCH --job-name=na-annotate
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gmem48|gmem80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-63%8
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$MODELS_ROOT/hf-cache"
/home/aeternum/anaconda3/envs/anno/bin/python tools/annotate_clips.py \
  --shots-dir "$WORKSPACE_ROOT/data/sakugabooru/shots" \
  --clips-dir "$WORKSPACE_ROOT/data/sakugabooru/clips" \
  --out-dir "$WORKSPACE_ROOT/data/sakugabooru/captions" \
  --config configs/stage0.yaml --shard "$SLURM_ARRAY_TASK_ID"
```

(`%8` caps concurrent GPUs at 8 — be a good citizen; raise if the queue is empty.)

- [ ] **Step 3: Smoke on 30 shots** (srun, one GPU): eyeball 10 captions for format compliance, hallucination, and vocabulary; measure per-shot latency and record projected total GPU-hours in the task report. Tune `temperature`/`frames_per_shot` if outputs are weak. **Step 4: Submit the array.** **Step 5: Commit** (`"Add Qwen3-VL annotation CLI + SLURM array"`).

---

### Task 10: Tiers + v2 metadata builder

**Files:**
- Create: `src/native_animation/data/tiers.py`
- Modify: `src/native_animation/data/build_metadata.py` (add a v2 entrypoint alongside v1; do not break v1's tested behavior)
- Test: `tests/test_tiers.py`, `tests/test_metadata_v2.py`

**Interfaces:**
- Consumes: shot manifests, profile shards, caption shards, post sidecars (rating/score/favorites), `assign_quantile_buckets`.
- Produces: `assign_tiers(posts: list[dict], s_quantile: float, a_quantile: float) -> dict[int, str]` (post_id → "S"/"A"/"B", ranked by (score, favorite_count)); `build_metadata_v2(...) -> None` writing `$WS/data/sakugabooru/metadata/v2/metadata_{all,train,val,test}.csv` with columns `video,prompt,summary,series,post_id,shot_id,tier,q_motion,q_deform,fps,score,rating,tags,fallback_caption` — `video` relative to `shots/`, `prompt` = caption description, split by series (seed 42, v1's leakage-free splitter), **rating == Safe only**. Plan 2 trains on these CSVs; Plan 3 freezes the benchmark from the test split.

- [ ] **Step 1: Failing tests**

```python
# tests/test_tiers.py
from native_animation.data.tiers import assign_tiers


def _posts(n=100):
    return [{"post_id": i, "score": i, "favorite_count": 0} for i in range(n)]


def test_tier_proportions_and_ordering():
    tiers = assign_tiers(_posts(), s_quantile=0.95, a_quantile=0.70)
    assert tiers[99] == "S" and tiers[96] == "S"
    assert tiers[80] == "A" and tiers[71] == "A"
    assert tiers[10] == "B"
    counts = {t: list(tiers.values()).count(t) for t in "SAB"}
    assert counts["S"] == 5 and counts["A"] == 25 and counts["B"] == 70


def test_favorites_break_score_ties():
    posts = [{"post_id": 1, "score": 10, "favorite_count": 9},
             {"post_id": 2, "score": 10, "favorite_count": 1}] + \
            [{"post_id": i + 10, "score": 0, "favorite_count": 0} for i in range(38)]
    tiers = assign_tiers(posts, s_quantile=0.975, a_quantile=0.9)
    assert tiers[1] == "S" and tiers[2] != "S"
```

```python
# tests/test_metadata_v2.py
import csv
import json
from pathlib import Path

from native_animation.data.build_metadata import build_metadata_v2


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fixture(tmp_path):
    ws = tmp_path
    clips = ws / "clips" / "demo_series"
    clips.mkdir(parents=True)
    for pid, score, rating in [(1, 500, "Safe"), (2, 5, "Safe"), (3, 900, "Explicit")]:
        (clips / f"{pid}.json").write_text(json.dumps(
            {"id": pid, "score": score, "favorite_count": 0, "rating": rating,
             "tags": "animated smears demo_series"}))
    _write(ws / "shots" / "manifests" / "shard_0000.jsonl", [
        {"shot_id": f"{pid}_00", "post_id": pid, "series": "demo_series",
         "video": f"demo_series/{pid}_00.mp4", "fps": 24.0,
         "curation": {"pass": pid != 2, "reasons": [] if pid != 2 else ["static"]}}
        for pid in (1, 2, 3)])
    _write(ws / "profiles" / "shard_0000.jsonl",
           [{"shot_id": "1_00", "flow_energy": 2.0, "nonrigid_residual": 0.5},
            {"shot_id": "3_00", "flow_energy": 9.0, "nonrigid_residual": 4.0}])
    _write(ws / "captions" / "shard_0000.jsonl",
           [{"shot_id": "1_00", "tag": "t", "summary": "s",
             "description": "a girl runs across the rooftop", "structured": {}, "fallback": False}])
    return ws


def test_v2_metadata_joins_filters_and_splits(tmp_path):
    ws = _fixture(tmp_path)
    out = ws / "metadata" / "v2"
    build_metadata_v2(clips_dir=ws / "clips", shots_dir=ws / "shots",
                      profiles_dir=ws / "profiles", captions_dir=ws / "captions",
                      output_dir=out, seed=42, s_quantile=0.95, a_quantile=0.70)
    with (out / "metadata_all.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    ids = {r["shot_id"] for r in rows}
    assert ids == {"1_00"}            # 2_00 fails curation, 3_00 is Explicit
    row = rows[0]
    assert row["prompt"] == "a girl runs across the rooftop"
    assert row["q_motion"] and row["tier"] in "SAB"
    for split in ("train", "val", "test"):
        assert (out / f"metadata_{split}.csv").exists()
```

- [ ] **Step 2: Verify FAIL.**

- [ ] **Step 3: Implement** — `tiers.py`:

```python
"""Quality tiers from community signals (spec §2 Stage 0)."""
from __future__ import annotations


def assign_tiers(posts: list[dict], s_quantile: float, a_quantile: float) -> dict[int, str]:
    ranked = sorted(posts, key=lambda p: (p.get("score", 0), p.get("favorite_count", 0)))
    n = len(ranked)
    s_cut, a_cut = int(n * s_quantile), int(n * a_quantile)
    tiers = {}
    for idx, post in enumerate(ranked):
        tiers[post["post_id"]] = "S" if idx >= s_cut else "A" if idx >= a_cut else "B"
    return tiers
```

`build_metadata_v2` in `build_metadata.py`: load sidecars (rating/score/favorites/tags per post) → `assign_tiers` → read all manifest/profile/caption shards into dicts keyed by shot_id → join curation-pass ∧ Safe ∧ has-profile rows (caption optional: missing → v1-style tag prompt, `fallback_caption` semantics) → corpus-wide `assign_quantile_buckets` on flow_energy and nonrigid_residual → reuse v1's `build_series_split` (seed) on the series set → write the four CSVs with the column set from **Interfaces**. Reuse `write_csv` patterns from v1; keep v1's `main()` untouched.

- [ ] **Step 4: Verify PASS + whole suite green (30 tests).** **Step 5: Commit** (`"Add tiers and v2 metadata builder"`).

---

### Task 11: Build v2 metadata + freeze the benchmark split

**Files:**
- Create: `tools/build_metadata_v2.py` (thin CLI over `build_metadata_v2`), `$WS/data/benchmarks/na-bench-v1/` (data, not git)

**Interfaces:**
- Consumes: everything above, complete.
- Produces: `$WS/data/sakugabooru/metadata/v2/metadata_{all,train,val,test}.csv` + `summary.json`; frozen benchmark at `$WS/data/benchmarks/na-bench-v1/` = the v2 **test** CSV + a copy of its shot files + `README.md` stating: series-disjoint, frozen 2026-09, never regenerated (spec §3).

- [ ] **Step 1:** Write the CLI (argparse defaults from `paths.env`-relative locations, `--config configs/stage0.yaml` for quantiles) and run it via srun once Tasks 6/7/9 arrays are complete. Inspect `summary.json`: total shots, per-tier counts, per-split counts, fallback-caption rate (flag to user if > 10%).
- [ ] **Step 2:** Freeze the benchmark: `mkdir -p $WS/data/benchmarks/na-bench-v1 && cp metadata_test.csv there`, copy the test-split shot files (`rsync --files-from`), write the README. Record clip count.
- [ ] **Step 3:** Update `docs/dataset.md` (v2 section: shots/profiles/captions/metadata-v2 layout, tier definition, benchmark pointer). **Commit** (`"Add v2 metadata CLI; document Stage-0 outputs and frozen benchmark"`).

---

### Task 12: Stage-0 verification sweep

**Files:** none (verification).

- [ ] **Step 1:** Full test suite green (`pytest -q`, expect 30+).
- [ ] **Step 2:** Corpus accounting table printed and saved to the task report: posts (merged) → shots planned → shots extracted → curation-pass → profiled → captioned (+fallback rate) → v2 rows (train/val/test) → tier counts. Every stage's count must be explainable (no silent loss > 5% between adjacent stages).
- [ ] **Step 3:** Spot-check 10 random v2 rows end-to-end: shot file plays (`ffprobe`), prompt reads as a directive, q buckets populated, tier sane vs score.
- [ ] **Step 4:** `git status` clean; push; report to user with the accounting table. Plan 2 (method/training) may already be executing by now — its Stage-1 launch gates on this task's completion.

---

## Self-review record

- **Spec coverage (§2 Stage 0):** delta scrape → T1; shot split → T3/T6; curation → T4/T6; profiling → T5/T7; annotation → T2/T8/T9; tiers + metadata + benchmark freeze → T10/T11; verification → T12. Rebalancing/curriculum *consume* q buckets in Plan 2 (spec §2 Stage 2) — correctly out of scope here.
- **Placeholders:** none; every code step is complete; Task 7 Step 2 and Task 9's transformers fallback reference concrete, already-shown patterns with exact deltas stated.
- **Type consistency:** shard file naming (`shard_{k:04d}.jsonl`), record keys (`shot_id`, `post_id`, `video`, `curation.pass`), and function signatures match across T3→T6→T7→T9→T10; `assign_quantile_buckets` (T5) is the same symbol T10 imports.
- **Reality checks encoded:** login-node IO starvation → everything on SLURM; `comfy` fragility → separate `anno` env with verify-first steps; model id uncertainty → explicit confirm step (T2.2); scenedetect not yet installed → explicit install step (T6.1); annotation cost unknown → measured in T9.3 before the array.
