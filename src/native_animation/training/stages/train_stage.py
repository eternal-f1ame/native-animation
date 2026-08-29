#!/usr/bin/env python3
"""Config-driven training entrypoint for CT/SFT stages (Native Animation v2).

The stage is entirely defined by its YAML config (see configs/ct_a.yaml,
ct_b.yaml, sft.yaml): data CSV, resolution/frames, curriculum, density,
objective, and optimizer settings. Launch under accelerate (FSDP config in
configs/accelerate_fsdp.yaml) via scripts/slurm/train_v2.sbatch.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import accelerate
import yaml

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import ModelLogger

from native_animation.training.curriculum import CurriculumDataset, CurriculumSampler
from native_animation.training.module_v2 import NativeAnimationV2Module
from native_animation.training.runner_v2 import train_loop

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_rows(csv_path: Path, limit: int | None) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", default=None, choices=[None, "auto"])
    parser.add_argument("--smoke-steps", type=int, default=None,
                        help="Override total_steps for smoke runs.")
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, train_cfg = cfg["data"], cfg["training"]
    if args.smoke_steps:
        train_cfg["total_steps"] = args.smoke_steps
        train_cfg["warmup_steps"] = min(train_cfg.get("warmup_steps", 0), 1)
    if args.resume:
        train_cfg["resume"] = args.resume

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1))

    metadata_path = Path(data_cfg["metadata_csv"])
    rows = load_rows(metadata_path, args.dataset_limit)
    tiers = set(data_cfg.get("tiers", []))
    if tiers:
        keep = [i for i, r in enumerate(rows) if r.get("tier") in tiers]
        rows = [rows[i] for i in keep]
    accelerator.print(f"dataset rows after tier filter: {len(rows)}")

    dataset = UnifiedDataset(
        base_path=data_cfg["base_path"],
        metadata_path=str(metadata_path),
        repeat=data_cfg.get("repeat", 1),
        data_file_keys=["video"],
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=data_cfg["base_path"],
            max_pixels=data_cfg.get("max_pixels", 1920 * 1080),
            height=data_cfg["height"],
            width=data_cfg["width"],
            height_division_factor=16,
            width_division_factor=16,
            num_frames=data_cfg["num_frames"],
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )

    curriculum = None
    if cfg.get("curriculum", {}).get("enabled", False):
        cur_cfg = cfg["curriculum"]
        curriculum = CurriculumSampler(
            rows, seed=cur_cfg.get("seed", 0),
            gamma=cur_cfg.get("gamma", 8.0), beta=cur_cfg.get("beta", 0.25))
        dataset = CurriculumDataset(dataset, curriculum)

    model = NativeAnimationV2Module(
        model_id_with_origin_paths=cfg["model"]["model_id_with_origin_paths"],
        trainable_models=cfg["model"].get("trainable_models", "dit"),
        lora_base_model=cfg["model"].get("lora_base_model"),
        lora_target_modules=cfg["model"].get("lora_target_modules", ""),
        lora_rank=cfg["model"].get("lora_rank", 32),
        use_gradient_checkpointing=True,
        extra_inputs=None,
        task="sft",
        device="cpu" if cfg["model"].get("initialize_on_cpu", True) else accelerator.device,
        v2_cfg=cfg.get("objective", {}),
        density_cfg=cfg.get("density", {}),
        text_dropout=cfg.get("objective", {}).get("text_dropout", 0.1),
        seed=train_cfg.get("seed", 0),
    )

    model_logger = ModelLogger(train_cfg["output_path"],
                               remove_prefix_in_ckpt="pipe.dit.")
    train_loop(accelerator, model, dataset, model_logger, train_cfg, curriculum)


if __name__ == "__main__":
    main()
