# Research Roadmap — CVPR 2027

Target: CVPR 2027 submission (~mid-November 2026). All experiments run afresh — no prior numbers carry over.
Positioning vs. Tencent AniMatrix (arXiv:2605.03652): see `related/animatrix.md` — cite as concurrent corroboration, never defensively.
Method spec: `superpowers/specs/2026-08-28-native-fm-v2-design.md`; execution plans in `superpowers/plans/` (each carries a status line).

## Workstreams

1. **Stage-0 data** *(in flight)* — full-snapshot streaming extraction → shot packs → motion profiling → Qwen3-VL annotation → `shots.sqsh`. Then: rebuild v2 metadata (tiers S/A/B) and freeze the benchmark split (Plan 1 T11–T12). Delta scrape resumes after its output path is packed (object quota).
2. **Training** — CT-a (256×448×17, fits 4×80GB, FSDP verified) → CT-b (480×832×49; needs 8×80GB and/or T5+VAE offload + FSDP CPU offload) → SFT (tiers S/A + curriculum) from `configs/{ct_a,ct_b,sft}.yaml`.
3. **Ablations** — anchor-mode mix, delta_mode (vspace vs off vs legacy), motion-weight α, timestep density (m,s,tail), curriculum on/off; v1 objective as a baseline arm.
4. **Baselines** — untuned Wan2.2-TI2V-5B; AniSora-V3.2 (external SOTA arm).
5. **Evaluation + DPO** *(Plan 3, not started)* — JEDi (V-JEPA + poly-MMD), AniSora-948 benchmark protocol, anchor-fidelity + flicker metrics, GT-anchored DPO after SFT; small user study.
6. **Paper** — CVPR author kit in `paper/`; figure pipeline from `experiments/`; fresh related-work pass.

## Known environment debts

- `open-clip-torch` not in `comfy` (only the v1 evaluator needs it).
- `experiments/` per-run directory convention still informal for v2 stages.
- Optional admin ask: object quota 2M → 5M (softened by pack/squash, still the clean fix).
