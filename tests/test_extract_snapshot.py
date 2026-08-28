"""Snapshot extractor: series derivation, sidecar conversion, end-to-end tar extraction."""
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from extract_snapshot import (  # noqa: E402
    build_sidecar,
    extract_snapshot,
    series_dir_from_tags,
)


def _typed(copyright=(), general=(), artist=()):
    return {
        "tag-type-copyright": list(copyright),
        "tag-type-general": list(general),
        "tag-type-artist": list(artist),
    }


def test_series_prefers_series_suffixed_copyright_tag():
    tags = _typed(copyright=["sengoku basara", "basara series"])
    assert series_dir_from_tags(tags) == "basara_series"


def test_series_reunderscores_to_match_existing_layout():
    # Mirror de-underscores tags; re-underscoring must reproduce our dirs exactly,
    # including punctuation ("jojo's_bizarre_adventure_series", ".hack_series").
    assert series_dir_from_tags(_typed(copyright=["jojo's bizarre adventure series"])) == "jojo's_bizarre_adventure_series"
    assert series_dir_from_tags(_typed(copyright=[".hack series"])) == ".hack_series"


def test_series_fallbacks():
    assert series_dir_from_tags(_typed(copyright=["fastening days"])) == "fastening_days"
    assert series_dir_from_tags(_typed()) == "_other"
    # Slash is a path hazard (the 22/7 anime) — replaced, not nested.
    assert "/" not in series_dir_from_tags(_typed(copyright=["22/7 series"]))


def test_sidecar_keeps_tags_clean_of_artists():
    post = {
        "id": 5, "score": 42, "width": 852, "height": 480, "rating": "Safe",
        "framerate": 23.976, "favorite_count": 7, "posted": "x", "post_url": "u",
        "source": None,
        "tags": _typed(copyright=["basara series"], general=["animated", "smears"], artist=["some artist"]),
    }
    side = build_sidecar(post)
    assert side["id"] == 5 and side["score"] == 42
    assert "some_artist" not in side["tags"]
    assert side["tags"] == "animated smears basara_series"
    assert side["tags_typed"]["tag-type-artist"] == ["some artist"]
    assert side["rating"] == "Safe" and side["framerate"] == 23.976


def _add(tar, name, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _post_json(pid, rating="Safe", copyright=("demo series",), general=("animated", "smears")):
    return json.dumps({
        "id": pid, "score": 10 + pid, "width": 852, "height": 480,
        "rating": rating, "framerate": 24.0, "favorite_count": 1,
        "posted": "d", "post_url": f"https://example/{pid}", "source": None,
        "tags": _typed(copyright=copyright, general=general),
    }).encode()


def _make_snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snapshot" / "train"
    snap.mkdir(parents=True)
    with tarfile.open(snap / "0.tar", "w") as tar:
        _add(tar, "sakuga_1.mp4", b"FAKEVIDEO1")
        _add(tar, "sakuga_1.json", _post_json(1))
        _add(tar, "sakuga_2.mp4", b"FAKEVIDEO2")
        _add(tar, "sakuga_2.json", _post_json(2, rating="Explicit"))
        _add(tar, "sakuga_3.jpg", b"FAKEIMAGE")           # image post -> skipped
        _add(tar, "sakuga_3.json", _post_json(3))
        _add(tar, "sakuga_4.json", _post_json(4))          # media deleted -> skipped
        _add(tar, "sakuga_9.mp4", b"FAKEVIDEO9")
        _add(tar, "sakuga_9.json", _post_json(9))
    return snap.parent


def test_end_to_end_extraction(tmp_path):
    snapshot = _make_snapshot(tmp_path)
    clips = tmp_path / "clips"
    # Pre-existing holding: post 9 already downloaded by the scraper.
    have = clips / "demo_series"
    have.mkdir(parents=True)
    (have / "9_s19.mp4").write_bytes(b"OLD")
    (have / "9.json").write_text("{}")
    (clips / "_state.json").write_text(json.dumps({"downloaded_ids": [9], "failed_ids": [], "stats": {}}))

    summary = extract_snapshot(snapshot_dir=snapshot, clips_dir=clips)

    # Post 1: extracted with our naming + enriched sidecar.
    out = clips / "demo_series" / "1_s11.mp4"
    assert out.read_bytes() == b"FAKEVIDEO1"
    side = json.loads((clips / "demo_series" / "1.json").read_text())
    assert side["tags"] == "animated smears demo_series"
    # Post 2 (Explicit), post 3 (image), post 4 (no media), post 9 (dupe): all skipped.
    assert not (clips / "demo_series" / "2_s12.mp4").exists()
    assert summary["extracted"] == 1
    assert summary["skipped_explicit"] == 1
    assert summary["skipped_non_video"] == 1
    assert summary["skipped_no_media"] == 1
    assert summary["skipped_existing"] == 1
    assert summary["max_post_id"] == 9

    # Scraper state now covers the extracted post so future scrapes dedupe.
    state = json.loads((clips / "_state.json").read_text())
    assert set(state["downloaded_ids"]) == {1, 9}

    # Idempotent: a second run extracts nothing.
    again = extract_snapshot(snapshot_dir=snapshot, clips_dir=clips)
    assert again["extracted"] == 0
