"""v2 metadata builder: join shards, filter, tier, bucket, split."""
import csv
import json
from pathlib import Path

from native_animation.data.build_metadata import build_metadata_v2


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fixture(tmp_path):
    ws = tmp_path
    clips = ws / "clips" / "demo_series"
    clips.mkdir(parents=True)
    for pid, score, rating in [(1, 500, "Safe"), (2, 5, "Safe"), (3, 900, "Explicit")]:
        (clips / f"{pid}.json").write_text(json.dumps(
            {"id": pid, "score": score, "favorite_count": 0, "rating": rating,
             "tags": "animated smears demo_series"}))
    _write(ws / "shots" / "manifests" / "shard_0000.jsonl", [
        {"shot_id": f"{pid}_00", "post_id": pid, "series": "demo_series",
         "video": f"demo_series/{pid}_00.mp4", "fps": 24.0,
         "curation": {"pass": pid != 2, "reasons": [] if pid != 2 else ["static"]}}
        for pid in (1, 2, 3)])
    _write(ws / "profiles" / "shard_0000.jsonl",
           [{"shot_id": "1_00", "flow_energy": 2.0, "nonrigid_residual": 0.5},
            {"shot_id": "3_00", "flow_energy": 9.0, "nonrigid_residual": 4.0}])
    _write(ws / "captions" / "shard_0000.jsonl",
           [{"shot_id": "1_00", "tag": "t", "summary": "s",
             "description": "a girl runs across the rooftop", "structured": {}, "fallback": False}])
    return ws


def test_v2_metadata_joins_filters_and_splits(tmp_path):
    ws = _fixture(tmp_path)
    out = ws / "metadata" / "v2"
    build_metadata_v2(clips_dir=ws / "clips", shots_dir=ws / "shots",
                      profiles_dir=ws / "profiles", captions_dir=ws / "captions",
                      output_dir=out, seed=42, s_quantile=0.95, a_quantile=0.70)
    with (out / "metadata_all.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    ids = {r["shot_id"] for r in rows}
    assert ids == {"1_00"}            # 2_00 fails curation, 3_00 is Explicit
    row = rows[0]
    assert row["prompt"] == "a girl runs across the rooftop"
    assert row["q_motion"] and row["tier"] in "SAB"
    for split in ("train", "val", "test"):
        assert (out / f"metadata_{split}.csv").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["num_rows"] == 1
