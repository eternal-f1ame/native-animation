"""Streaming per-tar processor: extract -> split -> manifest/state -> cleanup."""
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from stream_process_tar import process_tar  # noqa: E402

CFG = {
    "shots": {"min_seconds": 2.2, "max_seconds": 10.0, "detector_threshold": 27.0,
              "reencode": {"crf": 20, "short_side_max": 480, "pix_fmt": "yuv420p"}},
    "curation": {"min_mean_framediff": 1.0, "min_laplacian_var": 20.0},
}


def _tiny_video_bytes(frames=80, size=64, fps=24) -> bytes:
    """A real H.264 clip: textured background + moving square (passes curation)."""
    import imageio.v2 as imageio

    rng = np.random.default_rng(0)
    background = rng.integers(0, 120, (size, size, 3), dtype=np.uint8)
    path = Path("/tmp") / f"na_test_clip_{frames}.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=1)
    for i in range(frames):
        frame = background.copy()
        x = 4 + (i * 40 // frames)
        frame[20:36, x:x + 16] = 255
        writer.append_data(frame)
    writer.close()
    return path.read_bytes()


def _post_json(pid, rating="Safe"):
    return json.dumps({
        "id": pid, "score": 42, "width": 64, "height": 64, "rating": rating,
        "framerate": 24.0, "favorite_count": 0, "posted": "d", "post_url": "u",
        "source": None,
        "tags": {"tag-type-copyright": ["demo series"],
                 "tag-type-general": ["animated", "smears"],
                 "tag-type-artist": []},
    }).encode()


def _add(tar, name, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_env(tmp_path: Path) -> dict:
    video = _tiny_video_bytes()
    snapshot = tmp_path / "snapshot" / "train"
    snapshot.mkdir(parents=True)
    with tarfile.open(snapshot / "7.tar", "w") as tar:
        _add(tar, "sakuga_7001.mp4", video)
        _add(tar, "sakuga_7001.json", _post_json(7001))
        _add(tar, "sakuga_7002.mp4", video)
        _add(tar, "sakuga_7002.json", _post_json(7002, rating="Explicit"))
        _add(tar, "sakuga_7003.json", _post_json(7003))          # media deleted
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "_state.json").write_text(json.dumps({"downloaded_ids": [], "failed_ids": [], "stats": {}}))
    return {"tar": snapshot / "7.tar", "clips": clips,
            "shots": tmp_path / "shots", "staging": tmp_path / "staging"}


def test_stream_process_end_to_end(tmp_path):
    env = _make_env(tmp_path)
    summary = process_tar(env["tar"], env["clips"], env["shots"], env["staging"], CFG)

    shot = env["shots"] / "demo_series" / "7001_00.mp4"
    assert shot.exists() and shot.stat().st_size > 1000
    manifest = env["shots"] / "manifests" / "shard_t7.jsonl"
    records = [json.loads(l) for l in manifest.read_text().splitlines()]
    assert any(r["shot_id"] == "7001_00" and r["curation"]["pass"] for r in records)

    sidecar = json.loads((env["clips"] / "demo_series" / "7001.json").read_text())
    assert "smears" in sidecar["tags"]

    stream_state = json.loads((env["clips"] / "_state_stream_7.json").read_text())
    assert 7001 in stream_state["ids"]
    scraper_state = json.loads((env["clips"] / "_state.json").read_text())
    assert scraper_state["downloaded_ids"] == []   # shared state untouched (race-free)
    assert summary["extracted"] == 1 and summary["skipped_explicit"] == 1

    assert not env["tar"].exists()                       # tar deleted by default
    assert list(env["staging"].glob("*")) == []          # staging cleaned
    assert (env["shots"] / "manifests" / ".done_7").exists()


def test_stream_process_keep_tar_and_idempotent(tmp_path):
    env = _make_env(tmp_path)
    first = process_tar(env["tar"], env["clips"], env["shots"], env["staging"], CFG,
                        keep_tar=True)
    assert env["tar"].exists()                           # pilot mode preserves the tar
    again = process_tar(env["tar"], env["clips"], env["shots"], env["staging"], CFG,
                        keep_tar=True)
    assert first["extracted"] == 1 and again.get("skipped_done") is True


def test_stream_process_pack_mode(tmp_path):
    env = _make_env(tmp_path)
    summary = process_tar(env["tar"], env["clips"], env["shots"], env["staging"], CFG,
                          keep_tar=True, pack_shots=True)
    assert summary["extracted"] == 1

    # No loose shot files or per-post sidecar JSONs on shared storage.
    assert not (env["shots"] / "demo_series").exists()
    assert not (env["clips"] / "demo_series" / "7001.json").exists()

    # One pack tar containing the shot at its series-relative path.
    pack = env["shots"] / "packs" / "shots_t7.tar"
    assert pack.exists()
    with tarfile.open(pack) as t:
        names = [n.lstrip("./") for n in t.getnames()]
        assert "demo_series/7001_00.mp4" in names

    # One sidecar JSONL carrying the tags.
    lines = (env["clips"] / "sidecars" / "sidecars_t7.jsonl").read_text().splitlines()
    records = [json.loads(l) for l in lines]
    assert records[0]["id"] == 7001 and "smears" in records[0]["tags"]

    # Manifest records point at the pack.
    manifest = env["shots"] / "manifests" / "shard_t7.jsonl"
    rec = json.loads(manifest.read_text().splitlines()[0])
    assert rec["pack"] == "packs/shots_t7.tar"
