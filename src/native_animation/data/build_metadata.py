#!/usr/bin/env python3
"""Convert raw Sakugabooru clips into DiffSynth metadata CSV files.

This script targets the project use case:
- keep the format extremely simple
- split by series to reduce leakage
- create prompt text from series + tags
- write DiffSynth-friendly CSVs with at least `video` and `prompt`
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Root of sakugabooru clip directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write train/val/test CSV files")
    parser.add_argument("--seed", type=int, default=42, help="Series split seed")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio by series")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio by series")
    parser.add_argument("--limit", type=int, help="Optional cap on number of metadata entries")
    parser.add_argument("--min-score", type=int, default=0, help="Drop clips below this Sakugabooru score")
    parser.add_argument("--max-tags", type=int, default=20, help="Maximum number of tags to include in prompt text")
    parser.add_argument(
        "--prompt-prefix",
        default="native animation, anime",
        help="Prefix prepended to every generated prompt.",
    )
    return parser.parse_args()


def normalize_tags(tags: str) -> List[str]:
    """Split Sakugabooru's whitespace-separated tag string into a clean list."""
    return [tag.strip() for tag in tags.split() if tag.strip()]


def build_prompt(series: str, tags: List[str], max_tags: int, prompt_prefix: str) -> str:
    """Compose a training prompt: ``"<prefix>, <series>, <tag1>, <tag2>, ..."``."""
    if tags:
        return f"{prompt_prefix}, {series}, " + ", ".join(tags[:max_tags])
    return f"{prompt_prefix}, {series}"


def find_video_for_json(metadata_path: Path) -> Optional[Path]:
    """Locate the .mp4 that corresponds to a Sakugabooru JSON metadata file.

    Tries an exact-stem match first, then ``<stem>_s*.mp4`` (split clips), and
    finally any prefix match. Returns ``None`` if nothing plausible is found.
    """
    stem = metadata_path.stem
    parent = metadata_path.parent
    direct = parent / f"{stem}.mp4"
    if direct.exists():
        return direct
    suffix_matches = sorted(parent.glob(f"{stem}_s*.mp4"))
    if suffix_matches:
        return suffix_matches[0]
    fuzzy_matches = sorted(parent.glob(f"{stem}*.mp4"))
    if fuzzy_matches:
        return fuzzy_matches[0]
    return None


def build_series_split(series_names: Iterable[str], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, str]:
    """Assign each unique series to train/val/test.

    Splitting *by series* (not by clip) prevents leakage where the train and
    eval sets share frames from the same show. Small bumps guarantee at least
    one series in val/test when the requested ratio would otherwise round to 0.
    """
    names = sorted(set(series_names))
    rng = random.Random(seed)
    rng.shuffle(names)

    total = len(names)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)

    # Guarantee at least one series in val/test when feasible.
    if total >= 3 and test_ratio > 0 and test_count == 0:
        test_count = 1
    if total - test_count >= 2 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if test_count + val_count > total:
        val_count = max(0, total - test_count)

    split_map: Dict[str, str] = {}
    for idx, name in enumerate(names):
        if idx < test_count:
            split_map[name] = "test"
        elif idx < test_count + val_count:
            split_map[name] = "val"
        else:
            split_map[name] = "train"
    return split_map


def load_json(path: Path) -> Dict:
    """Read a single Sakugabooru JSON sidecar file."""
    with path.open() as handle:
        return json.load(handle)


def gather_json_paths(input_root: Path, limit: Optional[int]) -> List[Path]:
    """Collect every Sakugabooru clip-sidecar JSON under ``input_root`` (optionally capped).

    Root-level JSON files (e.g. the scraper's ``_state.json`` resume state) are
    excluded: they are not clip sidecars, and because the series split is keyed
    on each JSON's parent-directory name, a root-level file would leak the
    dataset directory's own name into the split shuffle — silently reshuffling
    train/val/test whenever the dataset folder is renamed.
    """
    paths = sorted(path for path in input_root.rglob("*.json") if path.parent != input_root)
    if limit is not None:
        paths = paths[:limit]
    return paths


def write_csv(path: Path, rows: List[Dict]) -> None:
    """Write ``rows`` to ``path`` with the canonical DiffSynth-friendly columns."""
    fieldnames = [
        "video",
        "prompt",
        "series",
        "tags",
        "score",
        "clip_id",
        "width",
        "height",
        "source",
        "split",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _read_jsonl_shards(directory: Path, key: str = "shot_id") -> Dict[str, Dict]:
    """Union all shard_*.jsonl files in ``directory`` into a dict keyed by ``key``."""
    table: Dict[str, Dict] = {}
    if not directory.exists():
        return table
    for shard in sorted(directory.glob("shard_*.jsonl")):
        with shard.open() as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    table[record[key]] = record
    return table


def build_metadata_v2(
    clips_dir: Path,
    shots_dir: Path,
    profiles_dir: Path,
    captions_dir: Path,
    output_dir: Path,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    s_quantile: float = 0.95,
    a_quantile: float = 0.70,
) -> None:
    """Stage-0 v2 metadata: join shots + profiles + captions + sidecars.

    Emits shot-level rows (video paths relative to ``shots_dir``) filtered to
    curation-pass and rating==Safe posts, with quality tier, motion/deformation
    quantile buckets, and the VLM directive as the training prompt.
    """
    from native_animation.data.profiling import assign_quantile_buckets
    from native_animation.data.tiers import assign_tiers

    output_dir.mkdir(parents=True, exist_ok=True)

    # Post-level facts from corpus sidecars.
    posts: Dict[int, Dict] = {}
    for sidecar in clips_dir.rglob("*.json"):
        if sidecar.name == "_state.json" or not sidecar.stem.isdigit():
            continue
        try:
            info = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        posts[int(sidecar.stem)] = info
    tiers = assign_tiers(
        [{"post_id": pid, "score": int(p.get("score", 0) or 0),
          "favorite_count": int(p.get("favorite_count", 0) or 0)} for pid, p in posts.items()],
        s_quantile=s_quantile, a_quantile=a_quantile,
    )

    shots = _read_jsonl_shards(shots_dir / "manifests")
    profiles = _read_jsonl_shards(profiles_dir)
    captions = _read_jsonl_shards(captions_dir)

    joined = []
    for shot_id, shot in shots.items():
        post = posts.get(shot["post_id"])
        if post is None or not shot["curation"]["pass"]:
            continue
        rating = str(post.get("rating") or "Safe")
        if not rating.lower().startswith("s"):
            continue
        profile = profiles.get(shot_id)
        if profile is None:
            continue
        caption = captions.get(shot_id)
        joined.append((shot, post, profile, caption))

    q = 4
    q_motion = assign_quantile_buckets([p["flow_energy"] for _, _, p, _ in joined], q) if joined else []
    q_deform = assign_quantile_buckets([p["nonrigid_residual"] for _, _, p, _ in joined], q) if joined else []

    split_map = build_series_split((s["series"] for s, _, _, _ in joined), val_ratio, test_ratio, seed)

    rows: List[Dict] = []
    for idx, (shot, post, profile, caption) in enumerate(joined):
        tags = post.get("tags", "")
        if caption is not None:
            prompt, summary, fallback = caption["description"], caption["summary"], caption.get("fallback", False)
        else:
            prompt = f"native animation, anime, {shot['series']}, " + ", ".join(tags.split()[:12])
            summary, fallback = "", True
        rows.append({
            "video": shot["video"],
            "prompt": prompt,
            "summary": summary,
            "series": shot["series"],
            "post_id": shot["post_id"],
            "shot_id": shot["shot_id"],
            "tier": tiers.get(shot["post_id"], "B"),
            "q_motion": q_motion[idx],
            "q_deform": q_deform[idx],
            "fps": shot.get("fps", 24.0),
            "score": int(post.get("score", 0) or 0),
            "rating": post.get("rating", "Safe"),
            "tags": tags,
            "fallback_caption": fallback,
            "split": split_map[shot["series"]],
        })

    fieldnames = ["video", "prompt", "summary", "series", "post_id", "shot_id", "tier",
                  "q_motion", "q_deform", "fps", "score", "rating", "tags",
                  "fallback_caption", "split"]

    def write(path: Path, subset: List[Dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in subset:
                writer.writerow(row)

    write(output_dir / "metadata_all.csv", rows)
    for split in ("train", "val", "test"):
        write(output_dir / f"metadata_{split}.csv", [r for r in rows if r["split"] == split])

    summary_payload = {
        "num_rows": len(rows),
        "num_train": sum(1 for r in rows if r["split"] == "train"),
        "num_val": sum(1 for r in rows if r["split"] == "val"),
        "num_test": sum(1 for r in rows if r["split"] == "test"),
        "num_series": len({r["series"] for r in rows}),
        "tier_counts": {t: sum(1 for r in rows if r["tier"] == t) for t in "SAB"},
        "fallback_caption_rate": (sum(1 for r in rows if r["fallback_caption"]) / len(rows)) if rows else 0.0,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary_payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = gather_json_paths(input_root, args.limit)
    if not json_paths:
        raise SystemExit(f"No JSON metadata found under {input_root}")

    # Series name is taken from the parent directory; the split is computed
    # once up front so every clip from a given series lands in the same split.
    split_map = build_series_split((path.parent.name for path in json_paths), args.val_ratio, args.test_ratio, args.seed)

    rows: List[Dict] = []
    skipped_missing_video = 0
    skipped_low_score = 0

    for json_path in json_paths:
        info = load_json(json_path)
        score = int(info.get("score", 0) or 0)
        if score < args.min_score:
            skipped_low_score += 1
            continue

        video_path = find_video_for_json(json_path)
        if video_path is None:
            skipped_missing_video += 1
            continue

        series = json_path.parent.name
        tags = info.get("tags", "")
        tag_list = normalize_tags(tags)
        rows.append(
            {
                # Stored relative to dataset root so the CSV stays portable.
                "video": str(video_path.relative_to(input_root)),
                "prompt": build_prompt(series, tag_list, args.max_tags, args.prompt_prefix),
                "series": series,
                "tags": tags,
                "score": score,
                "clip_id": info.get("id", json_path.stem),
                "width": info.get("width"),
                "height": info.get("height"),
                "source": info.get("source"),
                "split": split_map[series],
            }
        )

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]

    # Always write all four CSVs so downstream jobs can pick whichever split
    # they need without rerunning the build.
    write_csv(output_dir / "metadata_all.csv", rows)
    write_csv(output_dir / "metadata_train.csv", train_rows)
    write_csv(output_dir / "metadata_val.csv", val_rows)
    write_csv(output_dir / "metadata_test.csv", test_rows)

    summary = {
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "num_series": len(set(row["series"] for row in rows)),
        "top_series": Counter(row["series"] for row in rows).most_common(20),
        "skipped_missing_video": skipped_missing_video,
        "skipped_low_score": skipped_low_score,
        "input_root": str(input_root),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
