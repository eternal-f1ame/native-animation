"""Squash consolidation: pack tars -> single shots.sqsh, object-count endgame."""
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


@needs_squashfs
def test_consolidate_creates_sqsh_and_removes_packs(tmp_path):
    shots = tmp_path / "shots"
    _make_pack(shots / "packs", "0001", "series_a", ["1001_00.mp4"], tmp_path)
    _make_pack(shots / "packs", "0002", "series_b", ["2001_00.mp4", "2001_01.mp4"], tmp_path)

    result = _run(shots, tmp_path / "staging")
    assert result.returncode == 0, result.stderr

    sqsh = shots / "shots.sqsh"
    assert sqsh.exists()
    listing = subprocess.run(["unsquashfs", "-l", str(sqsh)],
                             capture_output=True, text=True, check=True).stdout
    assert "series_a/1001_00.mp4" in listing
    assert "series_b/2001_01.mp4" in listing
    # consolidated packs are deleted (the object-count win) and state records them
    assert not (shots / "packs" / "shots_t0001.tar").exists()
    state = json.loads((shots / "_state_squash.json").read_text())
    assert set(state["done_packs"]) == {"shots_t0001.tar", "shots_t0002.tar"}


@needs_squashfs
def test_consolidate_appends_and_skips_done(tmp_path):
    shots = tmp_path / "shots"
    _make_pack(shots / "packs", "0001", "series_a", ["1001_00.mp4"], tmp_path)
    assert _run(shots, tmp_path / "s1").returncode == 0

    # second wave: one new pack; rerun must append without disturbing wave 1
    _make_pack(shots / "packs", "0003", "series_c", ["3001_00.mp4"], tmp_path)
    result = _run(shots, tmp_path / "s2")
    assert result.returncode == 0, result.stderr
    listing = subprocess.run(["unsquashfs", "-l", str(shots / "shots.sqsh")],
                             capture_output=True, text=True, check=True).stdout
    assert "series_a/1001_00.mp4" in listing and "series_c/3001_00.mp4" in listing
    state = json.loads((shots / "_state_squash.json").read_text())
    assert len(state["done_packs"]) == 2


def test_materialize_shot_extra_roots(tmp_path):
    """Reader finds shots under a mounted-sqsh root passed as extra root."""
    shots = tmp_path / "shots"
    shots.mkdir()
    mount = tmp_path / "mnt"  # stands in for the squashfuse mountpoint
    (mount / "series_a").mkdir(parents=True)
    (mount / "series_a" / "1001_00.mp4").write_bytes(b"squashed")

    rec = {"video": "series_a/1001_00.mp4", "pack": "packs/shots_t0001.tar"}
    local, is_temp = materialize_shot(rec, shots, tmp_path / "tmp", extra_roots=(mount,))
    assert local is not None and local.read_bytes() == b"squashed"
    assert not is_temp
