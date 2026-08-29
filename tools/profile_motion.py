#!/usr/bin/env python3
"""Per-shot motion profiling: Farneback flow energy + non-rigid residual.

Reads the shot manifest shard produced by tools/split_shots.py, writes a
profile shard with the same sharding; idempotent by shot_id.
"""
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
        cap.release()
        return
    for start in np.linspace(0, total - 2, min(n_pairs, total - 1), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        ok1, a = cap.read()
        ok2, b = cap.read()
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
        print("no manifest shard; nothing to do")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"shard_{args.shard:04d}.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as handle:
            done = {json.loads(line)["shot_id"] for line in handle if line.strip()}

    processed = 0
    with manifest.open() as src, out_path.open("a") as out:
        for line in src:
            rec = json.loads(line)
            if rec["shot_id"] in done or not rec["curation"]["pass"]:
                continue
            energies, residuals = [], []
            for prev, nxt in pairs_from(args.shots_dir / rec["video"],
                                        pcfg["frame_pairs"], pcfg["flow_size"]):
                flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                energies.append(flow_energy(flow))
                residuals.append(nonrigid_residual(flow))
            if energies:
                out.write(json.dumps({"shot_id": rec["shot_id"],
                                      "flow_energy": float(np.mean(energies)),
                                      "nonrigid_residual": float(np.mean(residuals))}) + "\n")
                out.flush()
                processed += 1
    print(f"[shard {args.shard}] profiled {processed} shots")


if __name__ == "__main__":
    main()
