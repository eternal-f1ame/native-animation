# Workspace Restructure: Course Project → CVPR 2027 Research Codebase

- **Date:** 2026-08-28
- **Status:** Approved (design review with project owner)
- **Scope:** The entire `~/Research/Comic/Cartoon/` workspace
- **Driver:** The project is no longer a course project. It targets CVPR 2027 (submission deadline ~mid-November 2026, ~11 weeks out). The restructure must be a days-not-weeks operation that leaves maximum runway for experiments.

## 1. Context

The workspace grew organically around a course deliverable and now has:

- **Two live code copies** with different layouts: the DiffSynth fork (`FlowMatching/DiffSynth/`, branch `course-flowmatch-i2v-minimal`, flat files under `course_project/`) where cluster runs happen, and the standalone repo (`FlowMatching/native-animation-flowmatching/`, clean `src/` package) produced by a one-way exporter. The fork holds all runtime state (11 GB Wan weights, built CSVs, outputs, SLURM logs).
- Seven read-only reference clones mixed into `FlowMatching/` alongside live code.
- The dataset (`Anime/`, ~11.9k Sakugabooru clips, ~156 GB, 241 series) with its scraper at top level.
- A ~200 GB shared ComfyUI install unrelated to the paper.
- Course-era artifacts (reports, presentation, planning docs, progress log) scattered throughout.
- Known defects: the standalone repo's `DATA_ROOT` defaults point one directory level too high (`../Anime/` instead of `../../Anime/`); stale untracked `course_project/` and `diffsynth/` copies sit in the repo root; no tests anywhere.

**Constraints established:** single user now — no teammate depends on any path; ComfyUI and the dataset are movable. Everything is on one filesystem, so all moves are instant metadata-only `mv` operations.

## 2. Decisions (settled in design review)

1. **Single source of truth:** `native-animation-flowmatching` evolves into the canonical repo, keeping its git history. The DiffSynth fork is frozen into `third_party/` and the exporter pipeline dies.
2. **Name:** `native-animation` — repo dir `native-animation/`, pyproject name `native-animation`, GitHub repo renamed from `Dynamic-Panel-Animation` to `native-animation`. Python package stays `native_animation` (zero import churn).
3. **Structure:** repo-centric monorepo. Everything versionable and paper-relevant (code, paper, dataset tooling, docs) lives in the repo; the workspace around it holds only heavy non-versioned material (data, model weights, reference clones, archive).
4. Approved judgment calls: `artifacts/` → `experiments/`; ComfyUI moves out to `../`; fork dir renamed `DiffSynth-fork`; a pytest suite is part of this restructure; Claude runs `gh repo rename` (fallback: owner renames on github.com).
5. `Cartoon/` root rename: **deferred**, optional, only ever as a final standalone step (it breaks the active session's path mapping).
6. No "course" identity anywhere in the end state: no `course_project` paths, no course/team-contribution language in docs, no course-flavored filenames.

## 3. Target layout

### Workspace

```
Cartoon/
├── CLAUDE.md                 # rewritten: workspace geography (data, models, third_party, archive)
├── native-animation/         # ★ canonical git repo (see below)
├── data/
│   └── sakugabooru/
│       ├── clips/<series>/{post_id}_s{score}.mp4 + {post_id}.json   # moved as-is, incl. _state.json
│       ├── metadata/metadata_{all,train,val,test}.csv + summary.json
│       ├── cf_cookies.json   # scraper auth — never enters git
│       └── scrape-logs/
├── models/                   # shared weight cache (ModelScope/Wan trees from the fork)
├── third_party/              # frozen fork + 7 reference clones + README of roles
└── archive/                  # course-era record + README index
```

`ComfyUI/` and `README_comfyui_setup.md` relocate to `~/Research/Comic/` (workspace sibling), untouched internally.

### Repo (`native-animation/`)

```
native-animation/
├── pyproject.toml            # name=native-animation, v0.2.0, [dev] extra with pytest
├── README.md                 # research-facing rewrite
├── THIRD_PARTY.md            # vendored-diffsynth provenance (absorbs export_manifest.json)
├── CLAUDE.md                 # repo-level: dev commands, conventions (repo may be cloned bare)
├── configs/
│   └── paths.env             # single source: DATA_ROOT, MODELS_ROOT, METADATA_DIR — sourced by all scripts
├── src/native_animation/     # package, unchanged layout: data/ modeling/ training/ inference/ evaluation/
├── src/diffsynth/            # vendored runtime subset; grows from third_party/DiffSynth-fork as needed
├── tools/                    # dataset pipeline = claimed contribution #1, so it lives in the repo
│   ├── scrape_sakugabooru.py
│   ├── extract_cf_cookies.py
│   └── build_pair_dataset.py # renamed from Anime/build_dataset.py
├── scripts/
│   ├── train_native_animation.sh          # sources configs/paths.env
│   └── slurm/*.sbatch                     # same; artifacts/ → experiments/
├── tests/                    # pytest, CPU-only, no model weights required
├── paper/                    # from Final_Project_Report/: main.tex, sample.bib, images/, figures, scripts/
├── docs/
│   ├── method.md             # kept
│   ├── dataset.md            # new — from Anime/README.md, updated paths, growth plan
│   ├── roadmap.md            # new — skeleton (content is the next brainstorm)
│   └── superpowers/specs/    # this spec
├── models -> ../models       # symlink; DiffSynth's cwd-relative downloads hit the shared cache
└── experiments/              # gitignored: logs/ checkpoints/ demo/ eval/
```

## 4. Migration phases

Each phase leaves the workspace in a consistent, resumable state. Order matters: runtime state is extracted from the fork **before** the fork is frozen.

**Phase 0 — Preflight.** `squeue --me` must show no jobs (none may reference old paths). Record `tree -L 2` and `df -h` snapshots into `archive/migration-snapshot.txt`.

**Phase 1 — Skeleton.** Create `data/sakugabooru/{metadata,scrape-logs}/`, `models/`, `third_party/`, `archive/{course-report,planning,runs-2026-04,prototypes}/` at workspace root.

**Phase 2 — ComfyUI out.** `ComfyUI/` → `~/Research/Comic/ComfyUI/`; `README_comfyui_setup.md` → `~/Research/Comic/`.

**Phase 3 — Dataset.** From `Anime/`:

| Source | Destination |
|---|---|
| `sakugabooru_clips/` (incl. `_state.json`) | `data/sakugabooru/clips/` |
| `scrape_sakugabooru.py`, `extract_cf_cookies.py` | repo `tools/` |
| `build_dataset.py` | repo `tools/build_pair_dataset.py` |
| `README.md` | source text for repo `docs/dataset.md`; original → `archive/planning/anime-dataset-README.md` |
| `scrape.log`, `scrape_335809.log` | `data/sakugabooru/scrape-logs/` |
| `cf_cookies.json` | `data/sakugabooru/cf_cookies.json` |
| `slurm_scrape.sh` (stale paths) | `archive/planning/` |
| `__pycache__/` | delete |

`Anime/` is then removed.

**Phase 4 — Extract fork runtime state.** From `FlowMatching/DiffSynth/`:

| Source | Destination |
|---|---|
| `models/*` (~11 GB: `DiffSynth-Studio/…`, `Wan-AI/…`) | `models/` |
| `data/course_flowmatch_i2v/*` (4 CSVs + summary.json) | `data/sakugabooru/metadata/` |
| `outputs/` | `archive/runs-2026-04/outputs/` |
| `course_project/flowmatch_i2v/logs/` | `archive/runs-2026-04/slurm-logs/` |
| `dist/` (old export bundle) | `archive/runs-2026-04/dist/` |

**Phase 5 — Freeze references.** `FlowMatching/DiffSynth/` → `third_party/DiffSynth-fork/` (branch and working tree left exactly as-is — it is a historical artifact). `DiffSynth-Studio/`, `goku/`, `Sana/`, `Pyramid-Flow/`, `flowception/`, `Janus/`, `CausVid/` → `third_party/`. Write `third_party/README.md` stating each tree's role (fork = frozen former implementation host; Studio = pristine upstream for diffing; goku = rectified-flow concept reference; Sana/Pyramid-Flow = secondary design references; flowception = frame-insertion direction; Janus/CausVid = unused).

**Phase 6 — Repo transform.** `FlowMatching/native-animation-flowmatching/` → `Cartoon/native-animation/`, then, as a sequence of clean commits on `main`:

1. **Purge stale state:** delete untracked root `course_project/` and `diffsynth/` (verify by diff against `third_party/DiffSynth-fork` first — they must be duplicates); `presentation/` → `archive/course-report/`; `Final_Project_Report_Aaditya.pdf`, `frog.jpg` → `archive/course-report/`; commit the already-deleted `main.pdf`.
2. **`Final_Project_Report/` → `paper/`** (`git mv`), keeping `main.tex`, `sample.bib`, `images/`, generated figure PNGs, and `scripts/`. Update `.gitignore`'s LaTeX-artifact paths accordingly.
3. **Identity:** pyproject → `name = "native-animation"`, `version = "0.2.0"`, add `[project.optional-dependencies] dev = ["pytest"]`; console entrypoints unchanged.
4. **`configs/paths.env`** with `WORKSPACE_ROOT` (repo-relative `..`), `DATA_ROOT=$WORKSPACE_ROOT/data/sakugabooru/clips`, `METADATA_DIR=$WORKSPACE_ROOT/data/sakugabooru/metadata`, `MODELS_ROOT=$WORKSPACE_ROOT/models`. All five shell entrypoints (`scripts/train_native_animation.sh`, 4 sbatch files) source it and drop their private `DATA_ROOT` defaults. This permanently fixes the `../Anime` off-by-one.
5. **`artifacts/` → `experiments/`** in every script and `.gitignore` (add `/experiments/`, `/models`, drop `/artifacts/`).
6. **Tools in:** the three scraper/pair-builder scripts under `tools/`, header comments updated to new paths, committed.
7. **Tests in** (`tests/`, CPU-only):
   - `test_imports.py` — import every `native_animation` module.
   - `test_native_flowmatch.py` — scheduler `shift` forwarding; `_weighted_mse` normalizer invariant (`alpha=0` ≡ plain `.mean()`); `_motion_frame_weights` shapes with/without anchor frames and per-clip normalization.
   - `test_build_metadata.py` — tmp fixture tree of fake series/JSON/mp4-stub files → CSV columns, prompt format, and the series-split leakage guarantee (no series spans two splits).
   - `test_evaluate_metrics.py` — `diffusion_failure_score`, `temporal_consistency`, `worst_segment`, `final_score` on synthetic curves (flat-good, mid-collapse, jittery).
8. **Docs:** rewrite `README.md` (research framing, method summary, quickstart, no course/team language — authorship lives in the paper); update `THIRD_PARTY.md` (vendored `src/diffsynth/` provenance: originated in `third_party/DiffSynth-fork`, migrated 2026-08); delete `export_manifest.json`; add `docs/dataset.md` and `docs/roadmap.md` (skeleton only: dataset/training scale-up, per-component ablations, baseline sweep, evaluation expansion, paper timeline to ~2026-11); add repo-level `CLAUDE.md`.
9. **String audit:** `grep -ri "course\|submission bundle\|team contribution"` across the repo; rewrite every hit.
10. **`models -> ../models`** symlink (gitignored).
11. **Remote:** `gh repo rename native-animation -R eternal-f1ame/Dynamic-Panel-Animation` (fallback: owner renames in the GitHub UI); update `origin` URL; push.

**Phase 7 — Dissolve `FlowMatching/`.** `anime_keyframe_fm/`, `Project.md`, `progress.md`, `Mid-Project Update.pdf` → `archive/planning/`; root `evaluation.py` → `archive/prototypes/`; delete `FlowMatching/.claude/` and the empty `.codex` file; remove the emptied `FlowMatching/` dir. Write `archive/README.md` indexing what is where and why.

**Phase 8 — Workspace CLAUDE.md.** Rewrite for the new geography; the pre-restructure CLAUDE.md content is superseded (its operational knowledge — SLURM conventions, gotchas — carries into the repo-level CLAUDE.md where it belongs).

**Phase 9 — Verification** (see §6).

## 5. Data and model conventions going forward

- Metadata CSVs store paths **relative to the clips root**, so the moved CSVs remain valid. As verification, `build_metadata` is re-run with `--seed 42` against the new root and the output diffed against the moved CSVs (the split is deterministic; only `summary.json`'s absolute-path fields may differ and are regenerated).
- Named growth points (documented in `docs/dataset.md`, not built now): `data/sakugabooru/pairs/` for windowed keyframe-pair exports via `tools/build_pair_dataset.py`; `data/benchmarks/` for frozen evaluation sets.
- All model downloads land in workspace `models/` via the repo symlink; nothing model-sized is ever written inside the repo.

## 6. Verification checklist (end of migration)

1. `pytest` green in the `comfy` env.
2. `python -m py_compile` over all `src/native_animation` entrypoints and `tools/`.
3. Metadata regeneration diff-clean against moved CSVs (modulo `summary.json` absolute paths).
4. One real keyframe extraction (`native_animation.data.extract_keyframes`) against `data/sakugabooru/clips` succeeds.
5. `git status` clean; `git log` shows the phase commits; remote renamed and pushed.
6. `grep -ri course` in repo returns nothing; no path containing `course_project`, `Anime/`, or `FlowMatching/` remains outside `archive/` and `third_party/`.
7. Workspace root contains exactly: `CLAUDE.md`, `native-animation/`, `data/`, `models/`, `third_party/`, `archive/` (plus `.claude/`).
8. Optional cluster check (owner-triggered): `sbatch scripts/slurm/env_smoke_test.sbatch` passes on a GPU node.

## 7. Out of scope (deliberately)

- `Cartoon/` root rename (deferred; standalone final step if ever).
- CVPR LaTeX template switch in `paper/` (roadmap item — `main.tex` is currently article-class).
- A real config/experiment-tracking system and per-run `experiments/<run>/` layout (roadmap; `paths.env` is the only config artifact now).
- Installing the missing CLIP backend (`open-clip-torch`) into `comfy` — environment work for the first eval run, not restructure.
- `docs/roadmap.md` *content* — the research plan is the next brainstorming session.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Interrupted mid-migration | Phases are ordered so every intermediate state is consistent; each phase is a short batch of `mv`s; this spec + phase commits are the resume record |
| Stale root `course_project`/`diffsynth` dirs are not actually duplicates | Diff against `third_party/DiffSynth-fork` before deleting; anything divergent goes to `archive/` instead |
| Hidden absolute paths break after moves | §6 checks 3–4 exercise the real data path; grep for `/Cartoon/FlowMatching` and `/Cartoon/Anime` across repo + metadata after moving |
| GitHub rename breaks clones elsewhere | GitHub redirects renamed repos; local `origin` updated explicitly |
