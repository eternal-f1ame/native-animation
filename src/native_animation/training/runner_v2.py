"""Runner v2: the training loop the campaign needs and the stock runner lacks.

Adds over ``diffsynth.diffusion.runner.launch_training_task`` (whose structure
this mirrors): cosine LR with warmup, gradient clipping, EMA of trainable
parameters, curriculum refresh, JSONL loss logging, and full state
save/resume via ``accelerator.save_state``/``load_state`` — every long job on
this cluster must be resumable (spec §2 Stage 1).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm


def cosine_warmup_lambda(warmup_steps: int, total_steps: int, floor: float = 0.1):
    """LR multiplier: linear 0->1 over warmup, cosine 1->floor after."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


class EMA:
    """Exponential moving average of parameters, shadowed on CPU float32."""

    def __init__(self, params, decay: float = 0.995):
        self.decay = decay
        self.shadow = [p.detach().clone().float().cpu() for p in params]

    def update(self, params) -> None:
        with torch.no_grad():
            for shadow, param in zip(self.shadow, params):
                shadow.mul_(self.decay).add_(param.detach().float().cpu(), alpha=1 - self.decay)

    def copy_to(self, params) -> None:
        with torch.no_grad():
            for shadow, param in zip(self.shadow, params):
                param.data.copy_(shadow.to(dtype=param.dtype, device=param.device))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = [tensor.clone() for tensor in state["shadow"]]


def _latest_state_dir(state_root: Path) -> Path | None:
    if not state_root.exists():
        return None
    candidates = sorted(state_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return candidates[-1] if candidates else None


def train_loop(accelerator, model, dataset, model_logger, config: dict,
               curriculum=None) -> None:
    """Config keys: learning_rate, weight_decay, warmup_steps, total_steps,
    grad_clip, ema_decay, save_steps, save_state_steps, curriculum_refresh_steps,
    num_workers, output_path, resume ("auto" | None)."""
    optimizer = torch.optim.AdamW(model.trainable_modules(),
                                  lr=config["learning_rate"],
                                  weight_decay=config.get("weight_decay", 1e-2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(config["warmup_steps"], config["total_steps"]))
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0],
        num_workers=config.get("num_workers", 1))
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler)
    accelerator.print(f"[runner] wrapped model type: {type(model).__name__}")
    fsdp_children = sum(1 for m in model.modules() if "FullyShard" in type(m).__name__)
    accelerator.print(f"[runner] FSDP-wrapped submodules: {fsdp_children}")

    state_root = Path(config["output_path"]) / "state"
    log_path = Path(config["output_path"]) / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    if config.get("resume") == "auto":
        latest = _latest_state_dir(state_root)
        if latest is not None:
            accelerator.load_state(str(latest))
            step = int(latest.name.split("_")[1])
            accelerator.print(f"resumed from {latest} at step {step}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    ema = EMA(trainable, decay=config.get("ema_decay", 0.995)) if config.get("ema_decay") else None

    done = False
    while not done:
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                if config.get("grad_clip"):
                    accelerator.clip_grad_norm_(trainable, config["grad_clip"])
                optimizer.step()
                scheduler.step()
            if accelerator.sync_gradients:
                step += 1
                if ema is not None:
                    ema.update(trainable)
                if curriculum is not None and step % config.get("curriculum_refresh_steps", 200) == 0:
                    curriculum.refresh(min(step / config["total_steps"], 1.0))
                if accelerator.is_main_process:
                    with log_path.open("a") as handle:
                        handle.write(json.dumps({"step": step, "loss": float(loss.detach()),
                                                 "lr": scheduler.get_last_lr()[0],
                                                 "time": time.time()}) + "\n")
                if step % config.get("save_state_steps", 500) == 0:
                    accelerator.save_state(str(state_root / f"step_{step}"))
                    if accelerator.is_main_process:   # keep last two states only
                        states = sorted(state_root.glob("step_*"),
                                        key=lambda p: int(p.name.split("_")[1]))
                        for old in states[:-2]:
                            import shutil
                            shutil.rmtree(old, ignore_errors=True)
                if config.get("save_steps") and step % config["save_steps"] == 0:
                    model_logger.on_step_end(accelerator, model, config["save_steps"], loss=loss)
                if step >= config["total_steps"]:
                    done = True
                    break
        else:
            continue
    model_logger.on_training_end(accelerator, model, config.get("save_steps"))
