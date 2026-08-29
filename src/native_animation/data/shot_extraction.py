"""Per-video shot extraction: scene detection -> windows -> re-encoded shots.

Shared by the batch splitter (tools/split_shots.py) and the streaming per-tar
processor (tools/stream_process_tar.py). Video-IO heavy — the pure windowing
math stays in ``shots.py``.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

from native_animation.data.curation import blur_score, curation_verdict, static_score
from native_animation.data.shots import plan_shot_windows

from scenedetect import ContentDetector, detect


def sample_frames(path: Path, n: int = 6) -> list:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for idx in [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, (256, 256)))
    cap.release()
    return frames


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return float(frames / fps) if fps else 0.0


def reencode(src: Path, dst: Path, start: float, end: float, cfg: dict) -> bool:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    side = cfg["short_side_max"]
    vf = (f"scale='if(gt(iw,ih),-2,min({side},iw))':"
          f"'if(gt(iw,ih),min({side},ih),-2)'")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-vf", vf, "-an", "-c:v", "libx264", "-crf", str(cfg["crf"]),
           "-pix_fmt", cfg["pix_fmt"], "-threads", "2", str(dst)]
    return subprocess.run(cmd, check=False).returncode == 0


def split_video_into_shots(src: Path, post_id: int, series: str, out_dir: Path,
                           shots_cfg: dict, curation_cfg: dict,
                           done_ids: set | None = None) -> list[dict]:
    """Split one source video into re-encoded shots + manifest records."""
    done_ids = done_ids or set()
    try:
        scenes = detect(str(src), ContentDetector(threshold=shots_cfg["detector_threshold"]))
        scene_list = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    except Exception:
        return []
    if not scene_list:  # single-shot post: use the whole duration
        scene_list = [(0.0, video_duration(src))]

    records: list[dict] = []
    for idx, (start, end) in enumerate(
        plan_shot_windows(scene_list, shots_cfg["min_seconds"], shots_cfg["max_seconds"])
    ):
        shot_id = f"{post_id}_{idx:02d}"
        if shot_id in done_ids:
            continue
        out = out_dir / series / f"{shot_id}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() and not reencode(src, out, start, end, shots_cfg["reencode"]):
            continue
        frames = sample_frames(out)
        if not frames:
            continue
        cap = cv2.VideoCapture(str(out))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        st, bl = static_score(frames), blur_score(frames)
        records.append({"shot_id": shot_id, "post_id": post_id, "series": series,
                        "video": str(out.relative_to(out_dir)),
                        "start_s": start, "end_s": end, "fps": float(fps),
                        "static": st, "blur": bl,
                        "curation": curation_verdict(st, bl, end - start, curation_cfg)})
    return records


def append_manifest(manifest_path: Path, records: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
