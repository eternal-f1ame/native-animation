#!/usr/bin/env python3
"""Storage-lean streaming processor: one snapshot tar -> shots, then delete it.

Per tar: extract each valid post's media to node-local staging, write the
enriched sidecar to the home clips tree (annotation/metadata need it), split
into re-encoded 480p shots written to the home shots tree, then remove the
staged media. On completion: manifest renamed into place, scraper state
updated, the HOME TAR DELETED (unless --keep-tar), and a done-marker written.
Home usage strictly decreases per tar; the mirror on HF remains the recovery
path for full-res media.

Crash-safe: no marker -> the whole tar reprocesses (extract/encode steps skip
existing outputs, so a rerun is fast).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.shot_extraction import split_video_into_shots  # noqa: E402

from extract_snapshot import (  # noqa: E402  (sibling tool module)
    MEMBER_RE,
    VIDEO_EXTS,
    _load_state,
    build_sidecar,
    series_dir_from_tags,
)


def _known_ids(clips_dir: Path) -> set[int]:
    """Union of the scraper's state and every per-tar stream state file.

    Stream tasks run 16-wide; each writes only its own _state_stream_<tar>.json
    (atomic tmp+replace, unique name) so concurrent tars never race.
    """
    known = set(_load_state(clips_dir).get("downloaded_ids", []))
    for state_file in clips_dir.glob("_state_stream_*.json"):
        try:
            known.update(json.loads(state_file.read_text()).get("ids", []))
        except json.JSONDecodeError:
            continue
    return known


def process_tar(tar_path: Path, clips_dir: Path, shots_dir: Path, staging_dir: Path,
                cfg: dict, include_explicit: bool = False, keep_tar: bool = False,
                pack_shots: bool = False) -> dict:
    tar_path, clips_dir = Path(tar_path), Path(clips_dir)
    shots_dir, staging_dir = Path(shots_dir), Path(staging_dir)
    manifests = shots_dir / "manifests"
    marker = manifests / f".done_{tar_path.stem}"
    if marker.exists():
        return {"skipped_done": True}

    manifests.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    # Pack mode (object-quota-lean): shots land in local staging, then one tar
    # per source-tar on shared storage; sidecars become one JSONL per tar.
    effective_shots_dir = (staging_dir / "shots_out") if pack_shots else shots_dir
    effective_shots_dir.mkdir(parents=True, exist_ok=True)
    pack_rel = f"packs/shots_t{tar_path.stem}.tar"
    scfg, ccfg = cfg["shots"], cfg["curation"]
    known = _known_ids(clips_dir)

    summary = {"extracted": 0, "shots": 0, "skipped_existing": 0,
               "skipped_explicit": 0, "skipped_no_media": 0, "max_post_id": 0}
    records_all: list[dict] = []
    sidecar_records: list[dict] = []
    new_ids: set[int] = set()

    with tarfile.open(tar_path) as tar:
        posts: dict[int, dict] = {}
        for member in tar.getmembers():
            match = MEMBER_RE.search(member.name)
            if not match:
                continue
            pid, ext = int(match.group(1)), match.group(2).lower()
            entry = posts.setdefault(pid, {"json": None, "media": []})
            if ext == "json":
                entry["json"] = member
            else:
                entry["media"].append((ext, member))

        for pid in sorted(posts):
            entry = posts[pid]
            if entry["json"] is None:
                continue
            summary["max_post_id"] = max(summary["max_post_id"], pid)
            video = next(((e, m) for e, m in entry["media"] if e in VIDEO_EXTS), None)
            if video is None:
                summary["skipped_no_media"] += 1
                continue
            if pid in known:
                summary["skipped_existing"] += 1
                continue
            post = json.load(tar.extractfile(entry["json"]))
            rating = str(post.get("rating") or "").lower()
            if rating.startswith("e") and not include_explicit:
                summary["skipped_explicit"] += 1
                continue

            series = series_dir_from_tags(post.get("tags", {}) or {})
            if pack_shots:
                sidecar_records.append({**build_sidecar(post), "series": series})
            else:
                sidecar_path = clips_dir / series / f"{pid}.json"
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                if not sidecar_path.exists():
                    sidecar_path.write_text(json.dumps(build_sidecar(post)))

            ext, media_member = video
            staged = staging_dir / f"{pid}.{ext}"
            with tar.extractfile(media_member) as src, staged.open("wb") as out:
                shutil.copyfileobj(src, out)
            records = split_video_into_shots(staged, pid, series, effective_shots_dir, scfg, ccfg)
            staged.unlink(missing_ok=True)
            if pack_shots:
                for record in records:
                    record["pack"] = pack_rel

            records_all.extend(records)
            summary["extracted"] += 1
            summary["shots"] += len(records)
            new_ids.add(pid)

    if pack_shots:
        packs_dir = shots_dir / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        pack_tmp = packs_dir / f"shots_t{tar_path.stem}.tar.tmp"
        with tarfile.open(pack_tmp, "w") as pack:
            pack.add(effective_shots_dir, arcname=".")
        pack_tmp.replace(shots_dir / pack_rel)
        sidecars_dir = clips_dir / "sidecars"
        sidecars_dir.mkdir(parents=True, exist_ok=True)
        side_tmp = sidecars_dir / f"sidecars_t{tar_path.stem}.jsonl.tmp"
        with side_tmp.open("w") as handle:
            for record in sidecar_records:
                handle.write(json.dumps(record) + "\n")
        side_tmp.replace(sidecars_dir / f"sidecars_t{tar_path.stem}.jsonl")
        shutil.rmtree(effective_shots_dir, ignore_errors=True)

    manifest_tmp = manifests / f"shard_t{tar_path.stem}.jsonl.tmp"
    with manifest_tmp.open("w") as handle:
        for record in records_all:
            handle.write(json.dumps(record) + "\n")
    manifest_tmp.replace(manifests / f"shard_t{tar_path.stem}.jsonl")

    stream_state = clips_dir / f"_state_stream_{tar_path.stem}.json"
    tmp_state = stream_state.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps({"ids": sorted(new_ids)}))
    tmp_state.replace(stream_state)

    if not keep_tar:
        tar_path.unlink(missing_ok=True)
    marker.touch()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--shots-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--include-explicit", action="store_true")
    parser.add_argument("--keep-tar", action="store_true",
                        help="Pilot mode: process but do not delete the tar.")
    parser.add_argument("--pack-shots", action="store_true",
                        help="Object-lean mode: shots into one tar, sidecars into one JSONL.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    summary = process_tar(args.tar, args.clips_dir, args.shots_dir, args.staging_dir,
                          cfg, include_explicit=args.include_explicit,
                          keep_tar=args.keep_tar, pack_shots=args.pack_shots)
    print(json.dumps({"tar": args.tar.name, **summary}))


if __name__ == "__main__":
    main()
