"""Squash consolidation: pack tars -> fresh per-run images, object-count endgame."""
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.shot_access import materialize_shot  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools"

needs_squashfs = pytest.mark.skipif(
    shutil.which("mksquashfs") is None or shutil.which("unsquashfs") is None,
    reason="squashfs-tools not on this node",
)


def _make_pack(packs_dir: Path, stem: str, series: str, names: list[str], tmp: Path) -> Path:
    stage = tmp / f"stage_{stem}"
    (stage / series).mkdir(parents=True)
    for name in names:
        (stage / series / name).write_bytes(f"{stem}:{name}".encode())
    packs_dir.mkdir(parents=True, exist_ok=True)
    pack = packs_dir / f"shots_t{stem}.tar"
    with tarfile.open(pack, "w") as handle:
        handle.add(stage / series, arcname=f"./{series}")
    shutil.rmtree(stage)
    return pack


def _run(shots_dir: Path, staging: Path, batch: int = 100) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOLS / "consolidate_squash.py"),
         "--shots-dir", str(shots_dir), "--staging-dir", str(staging),
         "--batch-size", str(batch)],
        capture_output=True, text=True, check=False)


def _listing(sqsh: Path) -> str:
    return subprocess.run(["unsquashfs", "-l", str(sqsh)],
                          capture_output=True, text=True, check=True).stdout


@needs_squashfs
def test_consolidate_creates_image_and_removes_packs(tmp_path):
    shots = tmp_path / "shots"
    _make_pack(shots / "packs", "0001", "series_a", ["1001_00.mp4"], tmp_path)
    _make_pack(shots / "packs", "0002", "series_b", ["2001_00.mp4", "2001_01.mp4"], tmp_path)

    result = _run(shots, tmp_path / "staging")
    assert result.returncode == 0, result.stderr

    sqsh = shots / "shots_0000.sqsh"
    assert sqsh.exists()
    listing = _listing(sqsh)
    assert "series_a/1001_00.mp4" in listing
    assert "series_b/2001_01.mp4" in listing
    # consolidated packs are deleted (the object-count win) and state records them
    assert not (shots / "packs" / "shots_t0001.tar").exists()
    state = json.loads((shots / "_state_squash.json").read_text())
    assert set(state["done_packs"]) == {"shots_t0001.tar", "shots_t0002.tar"}
    assert state["images"] == ["shots_0000.sqsh"]


@needs_squashfs
def test_second_wave_gets_fresh_image_even_with_colliding_series(tmp_path):
    """Regression: mksquashfs append renames colliding top-level dirs to
    <name>_1 instead of merging — waves must land in fresh images."""
    shots = tmp_path / "shots"
    _make_pack(shots / "packs", "0001", "series_a", ["1001_00.mp4"], tmp_path)
    assert _run(shots, tmp_path / "s1").returncode == 0

    # second wave REUSES series_a — the exact collision that corrupted appends
    _make_pack(shots / "packs", "0003", "series_a", ["3001_00.mp4"], tmp_path)
    result = _run(shots, tmp_path / "s2")
    assert result.returncode == 0, result.stderr

    first, second = shots / "shots_0000.sqsh", shots / "shots_0001.sqsh"
    assert "series_a/1001_00.mp4" in _listing(first)
    listing2 = _listing(second)
    assert "series_a/3001_00.mp4" in listing2
    assert "series_a_1" not in listing2
    state = json.loads((shots / "_state_squash.json").read_text())
    assert state["images"] == ["shots_0000.sqsh", "shots_0001.sqsh"]
    assert len(state["done_packs"]) == 2


def test_materialize_shot_multiple_extra_roots(tmp_path):
    """Reader resolves across several mounted-image roots in order."""
    shots = tmp_path / "shots"
    shots.mkdir()
    mnt0, mnt1 = tmp_path / "m0", tmp_path / "m1"  # two squashfuse mountpoints
    (mnt0 / "series_a").mkdir(parents=True)
    (mnt0 / "series_a" / "1001_00.mp4").write_bytes(b"wave0")
    (mnt1 / "series_a").mkdir(parents=True)
    (mnt1 / "series_a" / "3001_00.mp4").write_bytes(b"wave1")

    rec = {"video": "series_a/3001_00.mp4", "pack": "packs/gone.tar"}
    local, is_temp = materialize_shot(rec, shots, tmp_path / "tmp",
                                      extra_roots=(mnt0, mnt1))
    assert local is not None and local.read_bytes() == b"wave1"
    assert not is_temp


def test_materialize_shot_missing_pack_returns_none(tmp_path):
    shots = tmp_path / "shots"
    (shots / "packs").mkdir(parents=True)
    rec = {"video": "series_a/9_00.mp4", "pack": "packs/shots_t9.tar"}
    local, is_temp = materialize_shot(rec, shots, tmp_path / "tmp")
    assert local is None and not is_temp
