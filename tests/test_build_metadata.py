"""Metadata builder: split determinism, series-level leakage guarantee, CSV shape."""
import csv
import json
import sys
from pathlib import Path

from native_animation.data.build_metadata import (
    build_prompt,
    build_series_split,
    find_video_for_json,
    gather_json_paths,
    main,
)


def _make_clip(root: Path, series: str, post_id: int, score: int = 100) -> None:
    series_dir = root / series
    series_dir.mkdir(parents=True, exist_ok=True)
    (series_dir / f"{post_id}.json").write_text(
        json.dumps(
            {
                "id": post_id,
                "score": score,
                "tags": "animated smears fighting",
                "width": 852,
                "height": 480,
                "source": "test",
            }
        )
    )
    # The builder only checks file existence; an empty file is enough.
    (series_dir / f"{post_id}_s{score}.mp4").write_bytes(b"")


def test_series_split_is_deterministic_and_covers_all():
    names = [f"series_{i}" for i in range(10)]
    split_a = build_series_split(names, val_ratio=0.1, test_ratio=0.1, seed=42)
    split_b = build_series_split(names, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert split_a == split_b
    assert set(split_a) == set(names)
    assert sorted(set(split_a.values())) == ["test", "train", "val"]


def test_find_video_matches_score_suffixed_files(tmp_path):
    _make_clip(tmp_path, "series_a", 111, score=250)
    json_path = tmp_path / "series_a" / "111.json"
    assert find_video_for_json(json_path).name == "111_s250.mp4"
    (tmp_path / "series_a" / "999.json").write_text("{}")
    assert find_video_for_json(tmp_path / "series_a" / "999.json") is None


def test_prompt_format():
    prompt = build_prompt("one_piece", ["smears", "fighting"], max_tags=20, prompt_prefix="native animation, anime")
    assert prompt == "native animation, anime, one_piece, smears, fighting"


def test_end_to_end_build_has_no_series_leakage(tmp_path, monkeypatch):
    clips_root = tmp_path / "clips"
    for i in range(10):
        _make_clip(clips_root, f"series_{i}", post_id=1000 + 2 * i)
        _make_clip(clips_root, f"series_{i}", post_id=1001 + 2 * i)
    out_dir = tmp_path / "meta"

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_metadata", "--input-root", str(clips_root), "--output-dir", str(out_dir), "--seed", "42"],
    )
    main()

    with (out_dir / "metadata_all.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    # Leakage guarantee: every series lives in exactly one split.
    series_to_splits = {}
    for row in rows:
        series_to_splits.setdefault(row["series"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in series_to_splits.values())
    # The per-split CSVs partition the full set.
    sizes = {}
    for name in ("train", "val", "test"):
        with (out_dir / f"metadata_{name}.csv").open() as handle:
            sizes[name] = len(list(csv.DictReader(handle)))
    assert sum(sizes.values()) == 20
    assert sizes["val"] >= 2 and sizes["test"] >= 2  # at least one series each (2 clips/series)


def test_gather_json_paths_skips_root_level_state_file(tmp_path):
    # The scraper keeps its resume state as a JSON at the dataset root; it is
    # not a clip sidecar and must never be treated as one.
    _make_clip(tmp_path, "series_a", 1)
    (tmp_path / "_state.json").write_text("{}")
    paths = gather_json_paths(tmp_path, limit=None)
    assert [p.name for p in paths] == ["1.json"]


def _build_split_map(root: Path, out_dir: Path, monkeypatch) -> dict:
    """Run the full builder against ``root`` and return its series→split map."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_metadata", "--input-root", str(root), "--output-dir", str(out_dir), "--seed", "42"],
    )
    main()
    with (out_dir / "metadata_all.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return {row["series"]: row["split"] for row in rows}


def test_split_is_independent_of_dataset_root_name(tmp_path, monkeypatch):
    # Regression: a root-level _state.json used to leak the root directory's
    # own name into the series-split shuffle, so renaming the dataset folder
    # silently reshuffled train/val/test.
    # Root names chosen to sort before vs. after the series names, mirroring
    # the real incident (sakugabooru_clips vs clips): the phantom's sort
    # position shifts every series' index, changing the whole assignment.
    maps = {}
    for root_name in ("aardvark_clips", "zebra_clips"):
        root = tmp_path / root_name
        for i in range(12):
            _make_clip(root, f"series_{i}", post_id=100 + i)
        (root / "_state.json").write_text(json.dumps({"downloaded": []}))
        maps[root_name] = _build_split_map(root, tmp_path / f"meta_{root_name}", monkeypatch)
    assert set(maps["aardvark_clips"]) == {f"series_{i}" for i in range(12)}  # no phantom series
    assert maps["aardvark_clips"] == maps["zebra_clips"]
