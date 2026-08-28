#!/usr/bin/env python3
"""Extract the sakugabooru2025 WebDataset snapshot into the project clips layout.

Streams each ``train/*.tar`` shard, pairs ``sakuga_{id}.mp4`` media with its
``sakuga_{id}.json`` metadata, and writes clips as
``clips/<series>/{post_id}_s{score}.{ext}`` plus an enriched sidecar
``{post_id}.json`` — the same layout the scraper produces, so the two sources
merge into one corpus.

Key behaviors:
- series directory derived from ``tag-type-copyright`` (re-underscored, so it
  merges exactly into the scraper's existing directories);
- image posts and (by default) Explicit-rated posts are skipped;
- post IDs already held (scraper state or on-disk sidecars) are skipped;
- the scraper's ``_state.json`` is extended with extracted IDs so future
  scrapes never re-download snapshot content;
- idempotent: re-running extracts nothing new.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path

MEMBER_RE = re.compile(r"sakuga_(\d+)\.(\w+)$")
VIDEO_EXTS = {"mp4", "webm", "mkv", "avi", "m4v"}


def _underscore(tag: str) -> str:
    """Mirror tags are de-underscored; reverse that, and defuse path hazards."""
    return tag.strip().replace(" ", "_").replace("/", "_")


def series_dir_from_tags(tags_typed: dict) -> str:
    """Pick the series directory from typed tags.

    Prefer a ``*_series`` copyright tag (the booru's franchise-wide tag),
    else the first copyright tag, else ``_other``.
    """
    copyrights = [_underscore(t) for t in tags_typed.get("tag-type-copyright", []) if t.strip()]
    if not copyrights:
        return "_other"
    for tag in copyrights:
        if tag.endswith("_series"):
            return tag
    return copyrights[0]


def build_sidecar(post: dict) -> dict:
    """Convert a mirror post JSON into our sidecar format (enriched).

    The flat ``tags`` string deliberately excludes artist tags — they are
    prompt noise — but the full typed dict is preserved in ``tags_typed``.
    """
    tags_typed = post.get("tags", {}) or {}
    flat = [
        _underscore(t)
        for key in ("tag-type-general", "tag-type-copyright")
        for t in tags_typed.get(key, [])
        if t.strip()
    ]
    return {
        "id": post.get("id"),
        "score": post.get("score", 0),
        "tags": " ".join(flat),
        "width": post.get("width"),
        "height": post.get("height"),
        "source": post.get("source"),
        "rating": post.get("rating"),
        "framerate": post.get("framerate"),
        "favorite_count": post.get("favorite_count"),
        "posted": post.get("posted"),
        "post_url": post.get("post_url"),
        "tags_typed": tags_typed,
    }


def _load_state(clips_dir: Path) -> dict:
    state_path = clips_dir / "_state.json"
    if state_path.exists():
        with state_path.open() as handle:
            return json.load(handle)
    return {"downloaded_ids": [], "failed_ids": [], "stats": {}}


def _save_state(clips_dir: Path, state: dict) -> None:
    tmp = clips_dir / "_state.json.tmp"
    with tmp.open("w") as handle:
        json.dump(state, handle)
    tmp.replace(clips_dir / "_state.json")


def _existing_ids(clips_dir: Path, state: dict) -> set:
    """IDs we already hold: scraper state plus on-disk sidecars."""
    ids = set(state.get("downloaded_ids", []))
    for sidecar in clips_dir.rglob("*.json"):
        if sidecar.name == "_state.json":
            continue
        stem = sidecar.stem
        if stem.isdigit():
            ids.add(int(stem))
    return ids


def extract_snapshot(
    snapshot_dir: Path,
    clips_dir: Path,
    include_explicit: bool = False,
    limit_tars: int | None = None,
) -> dict:
    snapshot_dir, clips_dir = Path(snapshot_dir), Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(clips_dir)
    existing = _existing_ids(clips_dir, state)

    summary = {
        "extracted": 0,
        "skipped_existing": 0,
        "skipped_explicit": 0,
        "skipped_no_media": 0,
        "skipped_non_video": 0,
        "skipped_orphan_media": 0,
        "max_post_id": 0,
        "tars_processed": 0,
    }
    new_ids: set = set()

    tars = sorted((snapshot_dir / "train").glob("*.tar"))
    if limit_tars is not None:
        tars = tars[:limit_tars]

    for tar_path in tars:
        with tarfile.open(tar_path) as tar:
            # Group members by post id: one json plus zero or more media files.
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
                    summary["skipped_orphan_media"] += 1
                    continue
                summary["max_post_id"] = max(summary["max_post_id"], pid)

                if not entry["media"]:
                    summary["skipped_no_media"] += 1
                    continue
                video = next(((ext, m) for ext, m in entry["media"] if ext in VIDEO_EXTS), None)
                if video is None:
                    summary["skipped_non_video"] += 1
                    continue
                if pid in existing or pid in new_ids:
                    summary["skipped_existing"] += 1
                    continue

                post = json.load(tar.extractfile(entry["json"]))
                rating = str(post.get("rating") or "").lower()
                if rating.startswith("e") and not include_explicit:
                    summary["skipped_explicit"] += 1
                    continue

                ext, media_member = video
                series = series_dir_from_tags(post.get("tags", {}) or {})
                series_path = clips_dir / series
                series_path.mkdir(parents=True, exist_ok=True)
                dest = series_path / f"{pid}_s{post.get('score', 0)}.{ext}"
                if dest.exists():
                    summary["skipped_existing"] += 1
                    continue

                tmp_dest = dest.with_suffix(dest.suffix + ".part")
                with tar.extractfile(media_member) as src, tmp_dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                tmp_dest.replace(dest)
                with (series_path / f"{pid}.json").open("w") as handle:
                    json.dump(build_sidecar(post), handle)
                summary["extracted"] += 1
                new_ids.add(pid)

        # Persist state per tar so an interrupted run stays resumable and
        # the scraper never re-downloads what we already extracted.
        state["downloaded_ids"] = sorted(set(state.get("downloaded_ids", [])) | new_ids)
        _save_state(clips_dir, state)
        summary["tars_processed"] += 1
        print(f"[{summary['tars_processed']}/{len(tars)}] {tar_path.name}: "
              f"extracted={summary['extracted']} existing={summary['skipped_existing']} "
              f"explicit={summary['skipped_explicit']} max_id={summary['max_post_id']}", flush=True)

    with (clips_dir.parent / "extraction-summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=workspace / "data" / "sakugabooru" / "snapshot-2025")
    parser.add_argument("--clips-dir", type=Path, default=workspace / "data" / "sakugabooru" / "clips")
    parser.add_argument("--include-explicit", action="store_true", help="Also extract Explicit-rated posts.")
    parser.add_argument("--limit-tars", type=int, default=None, help="Process only the first N tars (testing).")
    args = parser.parse_args()

    summary = extract_snapshot(
        snapshot_dir=args.snapshot_dir,
        clips_dir=args.clips_dir,
        include_explicit=args.include_explicit,
        limit_tars=args.limit_tars,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
