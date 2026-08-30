#!/usr/bin/env python3
"""Split corpus posts into single-shot re-encoded clips + curation verdicts.

Sharded by post-id modulo --num-shards so it runs as a SLURM array; each shard
is idempotent (skips shots already in its manifest) and appends to its own
manifest JSONL under <out-dir>/manifests/. The per-video core lives in
native_animation.data.shot_extraction (shared with the streaming processor).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.shot_extraction import append_manifest, split_video_into_shots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--pack-shots", action="store_true",
                        help="Object-lean: shots into one per-shard tar via local staging.")
    parser.add_argument("--staging-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    scfg, ccfg = cfg["shots"], cfg["curation"]
    pack_rel = f"packs/shots_b{args.shard:04d}.tar"
    if args.pack_shots:
        staging = Path(args.staging_dir or tempfile.mkdtemp(prefix="na_split_"))
        effective_out = staging / "shots_out"
        effective_out.mkdir(parents=True, exist_ok=True)
    else:
        effective_out = args.out_dir
    manifest_path = args.out_dir / "manifests" / f"shard_{args.shard:04d}.jsonl"
    done_ids = set()
    if manifest_path.exists():
        with manifest_path.open() as handle:
            done_ids = {json.loads(line)["shot_id"] for line in handle if line.strip()}

    processed = 0
    for sidecar in sorted(args.clips_dir.rglob("*.json")):
        if sidecar.name == "_state.json" or not sidecar.stem.isdigit():
            continue
        post_id = int(sidecar.stem)
        if post_id % args.num_shards != args.shard:
            continue
        videos = (list(sidecar.parent.glob(f"{post_id}_s*.mp4"))
                  + list(sidecar.parent.glob(f"{post_id}_s*.webm")))
        if not videos:
            continue
        records = split_video_into_shots(videos[0], post_id, sidecar.parent.name,
                                         effective_out, scfg, ccfg, done_ids)
        if args.pack_shots:
            for record in records:
                record["pack"] = pack_rel
        append_manifest(manifest_path, records)
        processed += len(records)
    if args.pack_shots and processed:
        packs_dir = args.out_dir / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        pack_path = args.out_dir / pack_rel
        mode = "a" if pack_path.exists() else "w"
        with tarfile.open(pack_path, mode) as pack:
            for series_dir in sorted(effective_out.iterdir()):
                pack.add(series_dir, arcname=f"./{series_dir.name}")
        shutil.rmtree(effective_out, ignore_errors=True)
    print(f"[shard {args.shard}] wrote {processed} new shots")


if __name__ == "__main__":
    main()
