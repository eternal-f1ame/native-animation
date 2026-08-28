# Research Roadmap — CVPR 2027

Status: skeleton. Content to be filled in a dedicated planning session.
Positioning and adoption plan relative to Tencent AniMatrix (arXiv:2605.03652): see `related/animatrix.md`.
All experiments run afresh — no prior numbers carry over (predecessor report archived).
Target: CVPR 2027 submission (~mid-November 2026).

## Workstreams

1. **Training scale-up** — multi-epoch schedules, LoRA-rank sweep, full fine-tune feasibility, 81-frame training.
2. **Ablations** — leave-one-out over the three method components (scheduler shift, motion weighting, delta consistency) plus sensitivity on `shift`, `alpha`, `lambda`.
3. **Baselines** — untuned Wan2.2-TI2V-5B, Wan2.1-I2V-14B preset, and at least one non-Wan I2V model.
4. **Evaluation** — freeze a benchmark split under `data/benchmarks/`; add FVD and/or a small user study; calibrate DFS thresholds against labeled clips.
5. **Paper** — write the CVPR 2027 manuscript from scratch in `paper/` (CVPR author kit); figure pipeline from `experiments/`; fresh related-work pass.

## Known environment debts

- `open-clip-torch` is not installed in the `comfy` env (the evaluator needs a CLIP backend).
- `experiments/` has no per-run directory convention yet; introduce one alongside a config system.
