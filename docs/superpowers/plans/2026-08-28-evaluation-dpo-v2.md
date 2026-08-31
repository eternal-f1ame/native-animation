# Evaluation Suite + GT-Anchored Preference Implementation Plan (v2 Plan 3 of 3)

> **Status (2026-08-30): NOT STARTED.** Next major implementation block after Stage-0.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v2 evaluation program (GT-anchored suite + JEDi distributional metric + benchmark/baseline runners + human-study tooling) and the GT-anchored preference stage (pair construction + Diffusion-DPO on a LoRA), completing spec §2 Stage 3 and §3.

**Architecture:** Pure-math cores TDD'd on CPU (MMD kernel, Krippendorff's α, pair building, the DPO loss estimator); model-facing runners as thin CLIs on SLURM; three external integrations (open-clip, JEDi/V-JEPA, AniSora-V3.2 baseline) handled as probe-then-adapt tasks with explicit fallbacks. DPO *code* lands now; its *run* gates on the Stage-2 checkpoint.

**Tech Stack:** comfy env (+`open-clip-torch`), vLLM-independent; JEDi via its published package or a vendored V-JEPA+MMD minimal; Diffusion-DPO per Wallace et al. estimator on the DiffSynth LoRA path.

**Spec:** `docs/superpowers/specs/2026-08-28-native-fm-v2-design.md` §2 Stage 3, §3, §8 items 2–3. Companions: Plans 1–2 (data, method/training).

## Global Constraints

- Same fleet rules as Plans 1–2: absolute `$COMFY_PY`/`$COMFY_ACCEL`, `SAKUGA_ROOT` data paths, everything heavy on SLURM, tests via `run_tests.sbatch`.
- The frozen benchmark (`$SAKUGA_ROOT/../benchmarks/na-bench-v1` → created in Plan 1 Task 11) is read-only once written; runners must never mutate it.
- Anti-reward-hacking invariants (spec §2 Stage 3): metric-diverse pair scoring, loser floor, gap weighting, 50-sample human spot-check gate before accepting any DPO checkpoint.
- Framing rule: animation problem first in all docs/strings.

---

### Task 1: CLIP backend for the GT suite (env debt)

**Files:** none (env + verification).

- [ ] **Step 1:** `$COMFY_PY -m pip install --no-deps open-clip-torch` then verify its runtime deps are already present: `$COMFY_PY -c "import open_clip, ftfy, regex; print(open_clip.__version__)"`. If `ftfy`/`regex` missing, install those two exactly (`pip install ftfy regex` — pure-python, safe).
- [ ] **Step 2:** GPU sanity via a one-off sbatch (pattern of `run_tests.sbatch` with `--partition=gpu --gpus-per-node=1`): load `open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")`, encode one image, print shape. Record model-cache location (`HF_HOME=$MODELS_ROOT/hf-cache`).
- [ ] **Step 3:** Run the existing evaluator end-to-end once on 2 GT/GT pairs (same video as both sides → near-perfect scores expected): `$COMFY_PY -m native_animation.evaluation.evaluate --ground-truth-video X --generated-video X`. Expected: `CFS>0.97`, label `[OK]`. Commit nothing (env-only) but record versions in the task report.

---

### Task 2: Distributional metric — JEDi (probe-then-adapt)

**Files:**
- Create: `src/native_animation/evaluation/jedi.py`
- Test: `tests/test_jedi_mmd.py` (the MMD core is ours and pure regardless of which feature extractor wins)

**Interfaces:**
- Produces: `poly_mmd2(x: torch.Tensor, y: torch.Tensor, degree=3, coef=1.0) -> float` (unbiased MMD² with polynomial kernel, (n,d)/(m,d) inputs) and `jedi_score(real_dir: Path, gen_dir: Path, num_videos: int | None) -> dict` (feature extraction + MMD; extractor per Step 1's outcome). Benchmark runners (Task 3) consume `jedi_score`.

- [ ] **Step 1 (probe):** On a GPU node, try the published JEDi package: `pip install videojedi` into a THROWAWAY venv (not comfy) and run its example; simultaneously check V-JEPA-2 availability on HF (`facebook/vjepa2-vitl-fpc64-256` or current id). Decide: (a) package works → wrap it; (b) package fights the env → vendored minimal: V-JEPA features via transformers + our `poly_mmd2`. Record the decision + exact ids in the module docstring.
- [ ] **Step 2 (RED):** MMD tests:

```python
# tests/test_jedi_mmd.py
import torch

from native_animation.evaluation.jedi import poly_mmd2


def test_identical_distributions_give_near_zero():
    torch.manual_seed(0)
    x = torch.randn(256, 32)
    assert abs(poly_mmd2(x, x.clone())) < 1e-6


def test_shifted_distributions_give_positive_and_ordered():
    torch.manual_seed(0)
    x = torch.randn(512, 32)
    near = torch.randn(512, 32) + 0.1
    far = torch.randn(512, 32) + 1.0
    assert poly_mmd2(x, far) > poly_mmd2(x, near) > 0


def test_sample_size_stability():
    torch.manual_seed(1)
    x, y = torch.randn(600, 16), torch.randn(600, 16) + 0.5
    small = poly_mmd2(x[:150], y[:150])
    large = poly_mmd2(x, y)
    assert abs(small - large) / large < 0.5   # same order of magnitude
```

- [ ] **Step 3 (GREEN):** implement `poly_mmd2` (unbiased estimator: kernel `k(a,b) = (a·b/d + coef)^degree`; `MMD² = mean(Kxx off-diag) + mean(Kyy off-diag) − 2·mean(Kxy)`), then `jedi_score` per the Step-1 decision (features per video: uniform 16-frame clip → extractor → mean-pool tokens → one vector/video). **Step 4:** suite green. **Step 5:** commit (`"Add JEDi distributional metric (poly-MMD core + feature wrapper)"`).

---

### Task 3: Benchmark + baseline runners

**Files:**
- Create: `tools/run_benchmark.py`, `scripts/slurm/benchmark.sbatch`
- Test: `tests/test_benchmark_manifest.py`

**Interfaces:**
- Consumes: `anchored_generate` (Plan 2 T8), the evaluator (`evaluate_pair`), `jedi_score`, frozen benchmark CSV.
- Produces: `plan_benchmark_runs(rows, modes=("keyframe","first_last","storyboard"), limit=None) -> list[dict]` (pure; each run item: shot_id, mode, anchor latent-frame indices computed from the GT shot length, prompt, gt path, out path); `tools/run_benchmark.py --checkpoint <path|"base"> --modes ... --limit ...` generating videos + `summary.json` (evaluator-compatible), then scoring: per-sample GT suite + set-level JEDi → `experiments/eval/<run-name>/report.json`.

- [ ] **Step 1 (RED):**

```python
# tests/test_benchmark_manifest.py
from native_animation.evaluation.benchmark import plan_benchmark_runs

ROWS = [{"shot_id": "1_00", "video": "s/1_00.mp4", "prompt": "p", "fps": 24.0}]


def test_modes_produce_correct_anchor_sets():
    runs = plan_benchmark_runs(ROWS, modes=("keyframe", "first_last", "storyboard"),
                               latent_frames=13, seed=0)
    by_mode = {r["mode"]: r for r in runs}
    assert by_mode["keyframe"]["anchors"] == [0]
    assert by_mode["first_last"]["anchors"] == [0, 12]
    sb = by_mode["storyboard"]["anchors"]
    assert sb[0] == 0 and len(sb) >= 2 and all(0 < a < 12 for a in sb[1:])


def test_out_paths_are_unique_per_mode():
    runs = plan_benchmark_runs(ROWS, modes=("keyframe", "first_last"),
                               latent_frames=13, seed=0)
    assert len({r["out_name"] for r in runs}) == len(runs)
```

- [ ] **Step 2 (GREEN):** `src/native_animation/evaluation/benchmark.py` with `plan_benchmark_runs` (storyboard anchors via seeded `sample_anchor_set` restricted to the storyboard mode); `tools/run_benchmark.py` iterating runs: extract anchor frames from the GT shot at the mapped raw-frame indices (latent idx k → raw frame 1+4·(k−1), k>0; idx 0 → frame 0), call `anchored_generate`, save, then evaluate (suite + JEDi) into `report.json`. `benchmark.sbatch`: 1×gmem48/80 GPU, `CHECKPOINT`/`MODES`/`LIMIT` env-driven.
- [ ] **Step 3:** Baseline adapters inside `run_benchmark.py`: `--model base` (vanilla Wan2.2-TI2V-5B, stock pipeline, keyframe mode only), `--model v1-lora <path>`. **AniSora-V3.2 (probe-then-adapt):** check `IndexTeam/Index-anisora` HF layout — if its V3.x weights load through DiffSynth `ModelConfig` (Wan2.2-family state dict), add `--model anisora`; if not, document the exact run procedure via `third_party` clone of their repo as an external baseline (their outputs dropped into a runs dir and scored by the same evaluator — the scoring path is model-agnostic by design). Also fetch their **948-clip benchmark** (repo `data/`/HF) into `$SAKUGA_ROOT/../benchmarks/anisora-948/` and write an adapter row-loader if the format differs. Timebox each probe to ~1h; record outcomes.
- [ ] **Step 4:** suite green; commit (`"Add benchmark/baseline runners with multi-anchor eval modes"`).

---

### Task 4: GT-anchored preference pairs

**Files:**
- Create: `src/native_animation/preference/pairs.py`, `src/native_animation/preference/__init__.py`, `tools/build_dpo_pairs.py`, `scripts/slurm/dpo_candidates.sbatch`
- Test: `tests/test_dpo_pairs.py`

**Interfaces:**
- Produces: `build_pairs(scored: list[dict], min_gap: float = 0.05, loser_floor: float = 0.2) -> list[dict]` — input items `{prompt_id, candidate_path, final_score}`; output pairs `{prompt_id, winner, loser, gap}` (all ordered pairs within a prompt group with `gap ≥ min_gap` and `loser_score ≥ loser_floor` — degenerate losers rejected so the model never learns from garbage-vs-worse); `gap` feeds GAPO-style weighting in Task 5. `tools/build_dpo_pairs.py`: N=4 candidates per held-out-TRAIN keyframe (never benchmark rows) via sbatch array → score each against GT with the suite → `pairs.jsonl`.

- [ ] **Step 1 (RED):**

```python
# tests/test_dpo_pairs.py
from native_animation.preference.pairs import build_pairs


def _scored(scores, prompt="p1"):
    return [{"prompt_id": prompt, "candidate_path": f"{prompt}_{i}.mp4", "final_score": s}
            for i, s in enumerate(scores)]


def test_all_ordered_pairs_above_gap():
    pairs = build_pairs(_scored([0.9, 0.7, 0.65]), min_gap=0.1, loser_floor=0.2)
    combos = {(p["winner"], p["loser"]) for p in pairs}
    assert ("p1_0.mp4", "p1_1.mp4") in combos and ("p1_0.mp4", "p1_2.mp4") in combos
    assert ("p1_1.mp4", "p1_2.mp4") not in combos            # gap 0.05 < 0.1


def test_loser_floor_rejects_degenerate_pairs():
    pairs = build_pairs(_scored([0.9, 0.05]), min_gap=0.1, loser_floor=0.2)
    assert pairs == []                                        # loser below floor


def test_groups_do_not_cross_prompts():
    scored = _scored([0.9, 0.5]) + _scored([0.8, 0.4], prompt="p2")
    pairs = build_pairs(scored, min_gap=0.1, loser_floor=0.2)
    assert all(p["winner"].startswith(p["prompt_id"]) for p in pairs)


def test_gap_recorded():
    pairs = build_pairs(_scored([0.9, 0.6]), min_gap=0.1, loser_floor=0.2)
    assert abs(pairs[0]["gap"] - 0.3) < 1e-9
```

- [ ] **Step 2 (GREEN):** implement; CLI generates candidates (seeded 4 per keyframe via the Stage-2 checkpoint through `anchored_generate`), scores with `evaluate_pair`, writes `pairs.jsonl` (+ per-pair video paths + gaps). sbatch array shards prompts across GPUs. **Step 3:** suite green; commit (`"Add GT-anchored preference pair construction"`).

---

### Task 5: Diffusion-DPO stage (LoRA on frozen Stage-2)

**Files:**
- Create: `src/native_animation/preference/dpo_loss.py`, `src/native_animation/training/stages/dpo_stage.py`, `configs/dpo.yaml`, `scripts/slurm/dpo_train.sbatch`
- Test: `tests/test_dpo_loss.py`

**Interfaces:**
- Consumes: pairs.jsonl (Task 4), Plan-2 runner/module machinery, v2 objective's noising conventions.
- Produces: `dpo_pair_loss(err_w_theta, err_l_theta, err_w_ref, err_l_ref, beta: float, gap_weight: float = 1.0) -> torch.Tensor` implementing `−gap_weight·log σ(−β/2·[(‖e_wθ‖²−‖e_wref‖²) − (‖e_lθ‖²−‖e_lref‖²)])` (Wallace et al. per-timestep estimator; inputs are velocity-prediction error tensors at a shared (σ, ε) draw); `dpo_stage.py`: loads Stage-2 pipe frozen + LoRA (rank 64, targets as v1), ref = same weights with LoRA disabled (`with pipe.lora_disabled()`-style toggle or a second frozen forward — implementation picks the memory-cheaper of the two and documents it), per step: sample pair → shared σ,ε → four forward passes (θ/ref × w/l) → loss; β from config, gap weighting `1 + gap` normalized.

- [ ] **Step 1 (RED):**

```python
# tests/test_dpo_loss.py
import torch

from native_animation.preference.dpo_loss import dpo_pair_loss


def _errs(w_good=True):
    base = torch.randn(1, 4, 5, 3, 3)
    small, big = base * 0.1, base * 1.0
    return (small if w_good else big), (big if w_good else small)


def test_preferring_the_better_sample_lowers_loss():
    torch.manual_seed(0)
    ref_w, ref_l = torch.randn(1, 4, 5, 3, 3) * 0.5, torch.randn(1, 4, 5, 3, 3) * 0.5
    good_w, good_l = _errs(w_good=True)
    bad_w, bad_l = _errs(w_good=False)
    loss_aligned = dpo_pair_loss(good_w, good_l, ref_w, ref_l, beta=500.0)
    loss_inverted = dpo_pair_loss(bad_w, bad_l, ref_w, ref_l, beta=500.0)
    assert loss_aligned < loss_inverted


def test_zero_margin_gives_log2():
    e = torch.randn(1, 4, 5, 3, 3)
    loss = dpo_pair_loss(e, e, e, e, beta=1000.0)
    assert torch.isclose(loss, torch.log(torch.tensor(2.0)), atol=1e-5)


def test_gap_weight_scales_loss():
    torch.manual_seed(1)
    w, l = _errs()
    ref = torch.randn_like(w) * 0.5
    base = dpo_pair_loss(w, l, ref, ref, beta=500.0, gap_weight=1.0)
    heavy = dpo_pair_loss(w, l, ref, ref, beta=500.0, gap_weight=2.0)
    assert torch.isclose(heavy, 2 * base, atol=1e-6)
```

- [ ] **Step 2 (GREEN):** implement loss (mean-per-element ‖·‖² for scale stability; β swept {500, 2000, 5000} via config), then the stage entrypoint reusing `runner_v2.train_loop` with a pair-dataset (each item: both videos VAE-encoded through the pipeline units once, cached to disk on first epoch — pairs corpus is small). `dpo.yaml` + `dpo_train.sbatch` (1–4×gmem80, LoRA-scale). **Step 3:** suite green; commit (`"Add Diffusion-DPO stage with gap weighting on frozen Stage-2 + LoRA"`).
- [ ] **Step 4 (gate, run-time):** the 50-sample human spot-check protocol is written into `docs/eval-protocol.md` (Task 6) — no DPO checkpoint is "accepted" without it.

---

### Task 6: Human study + metric-validation tooling

**Files:**
- Create: `src/native_animation/evaluation/agreement.py`, `tools/make_rating_sheets.py`, `tools/analyze_ratings.py`, `docs/eval-protocol.md`
- Test: `tests/test_agreement.py`

**Interfaces:**
- Produces: `krippendorff_alpha(ratings: list[list[float | None]], level="interval") -> float` (raters × items, None = missing); `spearman(xs, ys) -> float`; rating-sheet generator (writes an HTML page per rater: shuffled clip grid, the five AniMatrix dimensions on 1–5 scales, CSV template) and the analyzer (α per dimension, metric↔human Spearman table, FVD/JEDi anti-correlation check).

- [ ] **Step 1 (RED):**

```python
# tests/test_agreement.py
from native_animation.evaluation.agreement import krippendorff_alpha, spearman


def test_perfect_agreement_is_one():
    ratings = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    assert abs(krippendorff_alpha(ratings) - 1.0) < 1e-9


def test_random_disagreement_is_near_zero():
    ratings = [[1, 5, 2, 4, 3, 1, 5, 2, 4, 3],
               [5, 1, 4, 2, 3, 5, 1, 4, 2, 3]]
    assert krippendorff_alpha(ratings) < 0.2


def test_missing_values_are_tolerated():
    ratings = [[1, 2, None, 4], [1, 2, 3, 4], [None, 2, 3, 4]]
    assert krippendorff_alpha(ratings) > 0.9


def test_spearman_monotonic():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9
```

- [ ] **Step 2 (GREEN):** implement interval-level Krippendorff (observed vs expected disagreement over coincidence pairs; ~40 lines) and rank-based Spearman. **Step 3:** sheet generator (HTML with embedded `<video>` tags referencing benchmark outputs; model identity blinded via shuffled ids + a private key CSV) and analyzer. `docs/eval-protocol.md`: the 5 dimensions verbatim from the spec, rater instructions, 100–200 prompts × 3 raters, spot-check protocol for DPO. **Step 4:** suite green; commit (`"Add human-study tooling: agreement stats, blinded rating sheets, protocol"`).

---

### Task 7: Verification + docs

- [ ] **Step 1:** Full suite green (expect ~90 tests).
- [ ] **Step 2:** Dry-run wiring check on GPU: `tools/run_benchmark.py --model base --limit 2 --modes keyframe` end-to-end (generate → score → report.json exists with all metric keys).
- [ ] **Step 3:** Update repo `CLAUDE.md` (eval/DPO commands) and `docs/roadmap.md` (mark eval workstream tooling done; note run-gates: DPO awaits Stage-2, human study awaits benchmark outputs + raters).
- [ ] **Step 4:** `git status` clean, push, report: what runs today vs what gates on checkpoints/data.

---

## Self-review record

- **Spec coverage:** §3 baselines→T3; benchmarks (ours + AniSora-948 + JEDi)→T2/T3; human study + α + anti-correlation→T6; multi-anchor eval→T3 modes; §2 Stage 3 pairs/DPO/anti-hacking→T4/T5 (+T6 spot-check protocol); §8 open items 2–3 (AniSora variant check, AnimeReward availability)→T3 probe covers AniSora; AnimeReward probe folded into T4 Step 2 report (if public, add as second scorer column — noted, optional).
- **Placeholders:** none — probes are explicit verify-then-decide steps with fallbacks and timeboxes; all pure-math steps carry code.
- **Type consistency:** `plan_benchmark_runs` lives in `evaluation/benchmark.py` and is imported accordingly in its test; `evaluate_pair`/`final_score` names match the existing evaluator; `sample_anchor_set` reused from Plan-2's anchoring; pair dict keys consistent between T4 (`winner/loser/gap/prompt_id`) and T5's dataset reader.
