"""Reader-side shot materialization: loose-first, pack-extract fallback."""
import tarfile

from native_animation.data.shot_access import materialize_shot


def test_loose_file_wins(tmp_path):
    shots = tmp_path / "shots"
    (shots / "s").mkdir(parents=True)
    (shots / "s" / "1_00.mp4").write_bytes(b"LOOSE")
    path, is_temp = materialize_shot({"video": "s/1_00.mp4"}, shots, tmp_path / "tmp")
    assert path.read_bytes() == b"LOOSE" and is_temp is False


def test_pack_extraction(tmp_path):
    shots = tmp_path / "shots"
    (shots / "packs").mkdir(parents=True)
    src = tmp_path / "stage" / "s"
    src.mkdir(parents=True)
    (src / "2_00.mp4").write_bytes(b"PACKED")
    with tarfile.open(shots / "packs" / "shots_t2.tar", "w") as t:
        t.add(tmp_path / "stage", arcname=".")
    rec = {"video": "s/2_00.mp4", "pack": "packs/shots_t2.tar"}
    path, is_temp = materialize_shot(rec, shots, tmp_path / "tmp")
    assert path is not None and path.read_bytes() == b"PACKED" and is_temp is True


def test_missing_everywhere_returns_none(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    path, is_temp = materialize_shot({"video": "s/9.mp4"}, shots, tmp_path / "tmp")
    assert path is None and is_temp is False
