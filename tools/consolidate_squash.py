#!/usr/bin/env python3
"""Consolidate per-batch shot pack tars into squashfs images.

The object-count endgame: N pack tars -> a handful of squashfs images.
Each batch builds a FRESH image shots_<serial>.sqsh (mksquashfs append renames
colliding top-level series dirs to <name>_1 instead of merging — never append).
Readers mount every image (scripts/slurm/lib/mount_shots.sh) and resolve
series/file against each mount root. Single-writer by design — one SLURM job,
never an array, and NEVER concurrent with pack readers (profile/annotate):
consolidated packs are deleted, which yanks them from under a running reader.
Packs are only deleted after their content is verified inside the image, so a
killed run never loses data — rerunning resumes from the state file.
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


def mksquashfs_fresh(staging: Path, sqsh: Path) -> None:
    # -no-compression: members are already-compressed mp4s; squashfs gzip over
    # ~500G would burn hours of CPU for ~zero size gain.
    cmd = ["mksquashfs", str(staging), str(sqsh),
           "-no-progress", "-processors", "4", "-no-compression", "-noappend"]
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
    state_path = args.shots_dir / "_state_squash.json"
    state = load_state(state_path)
    state.setdefault("images", [])
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
        serial = max((int(n[6:10]) for n in state["images"]), default=-1) + 1
        sqsh = args.shots_dir / f"shots_{serial:04d}.sqsh"
        mksquashfs_fresh(staging, sqsh)
        verify_members(sqsh, samples)
        for pack in batch:
            pack.unlink()
            done.add(pack.name)
        state["done_packs"] = sorted(done)
        state["images"].append(sqsh.name)
        save_state(state_path, state)
        consolidated += len(batch)
        print(f"consolidated {consolidated}/{len(pending)} packs into {sqsh.name}", flush=True)
    shutil.rmtree(args.staging_dir / "extract", ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
