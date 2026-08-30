#!/usr/bin/env python3
"""Consolidate per-batch shot pack tars into a single shots.sqsh.

The object-count endgame: N pack tars -> 1 squashfs image (mksquashfs appends
into an existing image, so this runs incrementally as waves of packs land).
Single-writer by design — run as one SLURM job, never an array. Each batch:
extract pack tars to node-local staging, mksquashfs-append staging into
shots.sqsh, record the packs in _state_squash.json, delete them. Packs are
only deleted after their content is confirmed inside the image, so a killed
run never loses data — rerunning resumes from the state file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"done_packs": []}


def save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(state_path)


def mksquashfs_append(staging: Path, sqsh: Path) -> None:
    cmd = ["mksquashfs", str(staging), str(sqsh),
           "-no-progress", "-processors", "4", "-noappend"]
    if sqsh.exists():
        cmd.remove("-noappend")  # append mode: merge staging into existing image
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def verify_members(sqsh: Path, samples: list[str]) -> None:
    listing = subprocess.run(["unsquashfs", "-l", str(sqsh)],
                             capture_output=True, text=True, check=True).stdout
    for rel in samples:
        if rel not in listing:
            raise RuntimeError(f"post-append verification failed: {rel} not in {sqsh}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True,
                        help="Node-local scratch; sized for one batch of extracted packs.")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Pack tars consolidated per mksquashfs append.")
    args = parser.parse_args()

    packs_dir = args.shots_dir / "packs"
    sqsh = args.shots_dir / "shots.sqsh"
    state_path = args.shots_dir / "_state_squash.json"
    state = load_state(state_path)
    done = set(state["done_packs"])

    pending = sorted(p for p in packs_dir.glob("shots_*.tar") if p.name not in done) \
        if packs_dir.exists() else []
    if not pending:
        print("no pending packs; nothing to do")
        return

    consolidated = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        staging = args.staging_dir / "extract"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        samples: list[str] = []
        for pack in batch:
            with tarfile.open(pack) as handle:
                handle.extractall(staging, filter="data")
                names = [m.name.lstrip("./") for m in handle.getmembers() if m.isfile()]
            if names:
                samples.append(names[-1])
        mksquashfs_append(staging, sqsh)
        verify_members(sqsh, samples)
        for pack in batch:
            pack.unlink()
            done.add(pack.name)
        state["done_packs"] = sorted(done)
        save_state(state_path, state)
        consolidated += len(batch)
        print(f"consolidated {consolidated}/{len(pending)} packs into {sqsh.name}", flush=True)
    shutil.rmtree(args.staging_dir / "extract", ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
