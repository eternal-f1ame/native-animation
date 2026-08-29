#!/usr/bin/env python3
"""Split corpus posts into single-shot re-encoded clips + curation verdicts.

Sharded by post-id modulo --num-shards so it runs as a SLURM array; each shard
is idempotent (skips shots already in its manifest) and appends to its own
manifest JSONL under <out-dir>/manifests/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.curation import blur_score, curation_verdict, static_score  # noqa: E402
from native_animation.data.shots import plan_shot_windows  # noqa: E402

from scenedetect import ContentDetector, detect  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    scfg, ccfg = cfg["shots"], cfg["curation"]
    manifest_dir = args.out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"shard_{args.shard:04d}.jsonl"
    done_ids = set()
    if manifest_path.exists():
        with manifest_path.open() as handle:
            done_ids = {json.loads(line)["shot_id"] for line in handle if line.strip()}

    processed = 0
    with manifest_path.open("a") as manifest:
        for sidecar in sorted(args.clips_dir.rglob("*.json")):
            if sidecar.name == "_state.json" or not sidecar.stem.isdigit():
                continue
            post_id = int(sidecar.stem)
            if post_id % args.num_shards != args.shard:
                continue
            videos = (list(sidecar.parent.glob(f"{post_id}_s*.mp4"))
                      + list(sidecar.parent.glob(f"{post_id}_s*.webm")))
            if not videos:
                continue
            src = videos[0]
            series = sidecar.parent.name
            try:
                scenes = detect(str(src), ContentDetector(threshold=scfg["detector_threshold"]))
                scene_list = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
            except Exception:
                continue
            if not scene_list:  # single-shot post: use the whole duration
                scene_list = [(0.0, video_duration(src))]
            for idx, (start, end) in enumerate(
                plan_shot_windows(scene_list, scfg["min_seconds"], scfg["max_seconds"])
            ):
                shot_id = f"{post_id}_{idx:02d}"
                if shot_id in done_ids:
                    continue
                out = args.out_dir / series / f"{shot_id}.mp4"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists() and not reencode(src, out, start, end, scfg["reencode"]):
                    continue
                frames = sample_frames(out)
                if not frames:
                    continue
                cap = cv2.VideoCapture(str(out))
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                cap.release()
                st, bl = static_score(frames), blur_score(frames)
                record = {"shot_id": shot_id, "post_id": post_id, "series": series,
                          "video": str(out.relative_to(args.out_dir)),
                          "start_s": start, "end_s": end, "fps": float(fps),
                          "static": st, "blur": bl,
                          "curation": curation_verdict(st, bl, end - start, ccfg)}
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                processed += 1
    print(f"[shard {args.shard}] wrote {processed} new shots")


if __name__ == "__main__":
    main()
