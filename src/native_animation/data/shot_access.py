"""Shot materialization for readers (profiling, annotation, training prep).

Shots live either loose under the shots root (legacy/pre-pack output) or
inside per-batch pack tars (object-quota-lean mode). Readers call
``materialize_shot`` per manifest record and clean up temps they receive.
"""
from __future__ import annotations

import tarfile
from pathlib import Path


def materialize_shot(record: dict, shots_dir: Path, tmp_dir: Path) -> tuple[Path | None, bool]:
    """Return (local_path, is_temp) for a manifest record's video, or (None, False).

    Loose file wins; otherwise the shot is extracted from its pack tar into
    ``tmp_dir``. Callers unlink the returned path when ``is_temp`` is True.
    """
    loose = shots_dir / record["video"]
    if loose.exists():
        return loose, False
    pack_rel = record.get("pack")
    if not pack_rel:
        return None, False
    pack_path = shots_dir / pack_rel
    if not pack_path.exists():
        return None, False
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / Path(record["video"]).name
    with tarfile.open(pack_path) as pack:
        for name in (record["video"], f"./{record['video']}"):
            try:
                member = pack.getmember(name)
            except KeyError:
                continue
            with pack.extractfile(member) as src, out.open("wb") as dst:
                dst.write(src.read())
            return out, True
    return None, False
