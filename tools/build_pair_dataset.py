#!/usr/bin/env python3
"""
Keyframe-to-Animation Dataset Builder
======================================
Turns sakugabooru clips into (keyframe, short_clip) pairs for training
image-to-video models (e.g., Stable Video Diffusion, AnimateDiff).

For each source video, slides a window and extracts:
  - keyframe.jpg  : first frame of the window (conditioning image)
  - clip.mp4      : short clip starting from that frame (target video)

Output structure:
  dataset/
    keyframes/{id}.jpg
    clips/{id}.mp4
    manifest.jsonl      # one JSON object per pair with all metadata

Usage:
    python build_dataset.py                           # Default: 1s clips, 0.1s stride
    python build_dataset.py --clip-duration 2.0       # 2-second clips
    python build_dataset.py --stride 0.5              # Less overlap, fewer pairs
    python build_dataset.py --fps 8 --resolution 256  # SVD-style: 8fps, 256px
    python build_dataset.py --workers 8               # Parallel processing
    python build_dataset.py --estimate                # Show expected output size
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "clips"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sakugabooru" / "pairs"

DEFAULT_CLIP_DURATION = 1.0
DEFAULT_STRIDE = 0.1
DEFAULT_FPS = 24
DEFAULT_RESOLUTION = 512
DEFAULT_QUALITY = 23


def get_video_info(path: str) -> dict:
    """Get duration, fps, width, height from a video file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,duration",
             "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})
        duration = float(stream.get("duration", 0)) or float(fmt.get("duration", 0))
        fps_str = stream.get("r_frame_rate", "24/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        return {
            "duration": duration, "fps": fps,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
        }
    except Exception:
        return None


def load_clip_metadata(video_path: Path) -> dict:
    """Load the Sakugabooru metadata JSON for a clip."""
    stem = video_path.stem
    post_id = stem.split("_s")[0]
    meta_path = video_path.parent / f"{post_id}.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def extract_single_pair(args_tuple):
    """Extract one (keyframe, clip) pair. Designed for parallel dispatch."""
    (video_path, start_time, pair_id, keyframe_path, clip_path,
     clip_duration, target_fps, resolution, quality) = args_tuple

    video_path = str(video_path)

    # Scale filter: short side to resolution, divisible by 2
    # Probe not needed here — let ffmpeg figure it out with generic filter
    scale_filter = (
        f"scale='if(gte(iw,ih),{resolution},-2)':'if(gte(iw,ih),-2,{resolution})'"
    )

    # 1) Extract keyframe
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet",
             "-ss", f"{start_time:.3f}",
             "-i", video_path,
             "-vframes", "1",
             "-vf", scale_filter,
             "-q:v", "2",
             str(keyframe_path)],
            timeout=15, check=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    # 2) Extract clip
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet",
             "-ss", f"{start_time:.3f}",
             "-i", video_path,
             "-t", f"{clip_duration:.3f}",
             "-vf", f"{scale_filter},fps={target_fps}",
             "-c:v", "libx264", "-crf", str(quality),
             "-preset", "fast",
             "-pix_fmt", "yuv420p",
             "-an",
             str(clip_path)],
            timeout=30, check=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        for p in [keyframe_path, clip_path]:
            if os.path.exists(p):
                os.unlink(p)
        return None

    # Verify
    if (not os.path.exists(keyframe_path) or os.path.getsize(keyframe_path) < 100 or
            not os.path.exists(clip_path) or os.path.getsize(clip_path) < 500):
        for p in [keyframe_path, clip_path]:
            if os.path.exists(p):
                os.unlink(p)
        return None

    return pair_id


def find_source_videos(source_dir: Path) -> list[Path]:
    videos = []
    for ext in ("*.mp4", "*.webm"):
        videos.extend(source_dir.rglob(ext))
    videos.sort()
    return videos


def plan_pairs(videos, output_dir, clip_duration, stride, resume):
    """Build the full list of (video, start_time, pair_id, paths) work items.

    Also returns metadata per pair for the manifest.
    """
    keyframes_dir = output_dir / "keyframes"
    clips_dir = output_dir / "clips"
    work_items = []
    metadata_map = {}

    for video_path in videos:
        info = get_video_info(str(video_path))
        if not info or info["duration"] < clip_duration + 0.05:
            continue

        meta = load_clip_metadata(video_path)
        anime_tag = video_path.parent.name
        post_id = video_path.stem.split("_s")[0]

        duration = info["duration"]
        t = 0.0
        while t + clip_duration <= duration + 0.01:
            start_ms = int(round(t * 1000))
            pair_id = f"{post_id}_{start_ms:06d}"
            kf_path = keyframes_dir / f"{pair_id}.jpg"
            cl_path = clips_dir / f"{pair_id}.mp4"

            if resume and kf_path.exists() and cl_path.exists():
                t += stride
                continue

            work_items.append((
                str(video_path), t, pair_id, str(kf_path), str(cl_path)
            ))

            metadata_map[pair_id] = {
                "id": pair_id,
                "keyframe": f"keyframes/{pair_id}.jpg",
                "clip": f"clips/{pair_id}.mp4",
                "source_video": str(video_path),
                "start_time": round(t, 3),
                "clip_duration": clip_duration,
                "anime": anime_tag,
                "post_id": int(post_id),
                "score": meta.get("score", 0),
                "tags": meta.get("tags", ""),
                "source_fps": round(info["fps"], 2),
                "source_resolution": f"{info['width']}x{info['height']}",
            }

            t += stride

    return work_items, metadata_map


def estimate(source_dir, clip_duration, stride, resolution, fps):
    import random
    videos = find_source_videos(source_dir)
    print(f"Scanning {len(videos)} source videos...\n")
    sample = random.sample(videos, min(200, len(videos)))
    total_pairs = 0
    for v in sample:
        info = get_video_info(str(v))
        if info and info["duration"] >= clip_duration:
            total_pairs += max(0, int((info["duration"] - clip_duration) / stride) + 1)
    est_pairs = int(total_pairs * len(videos) / len(sample))
    frames_per_clip = int(clip_duration * fps)
    avg_kf_kb, avg_clip_kb = 50, frames_per_clip * 15
    total_gb = est_pairs * (avg_kf_kb + avg_clip_kb) / 1024 / 1024

    print(f"Source videos:        {len(videos)}")
    print(f"Clip:                 {clip_duration}s at {fps}fps ({frames_per_clip} frames)")
    print(f"Stride:               {stride}s")
    print(f"Resolution:           {resolution}px (short side)")
    print(f"")
    print(f"Estimated pairs:      ~{est_pairs:,}")
    print(f"Estimated total size: ~{total_gb:.0f} GB")
    print(f"")
    if stride < 0.5:
        print(f"Dense sampling. Consider --stride 0.5 for {est_pairs * stride / 0.5:.0f} pairs "
              f"or --stride 1.0 for non-overlapping.")


def main():
    parser = argparse.ArgumentParser(
        description="Build keyframe-to-animation dataset from sakuga clips"
    )
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--clip-duration", type=float, default=DEFAULT_CLIP_DURATION,
                        help="Seconds per output clip (default: 1.0)")
    parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE,
                        help="Sliding window stride in seconds (default: 0.1)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Output framerate (default: 24)")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help="Short-side resolution in px (default: 512)")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                        help="x264 CRF (default: 23)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel ffmpeg workers (default: cpu_count)")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs whose output files already exist")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N source videos (testing)")

    args = parser.parse_args()
    source_dir = Path(args.source) if args.source else SOURCE_DIR
    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    if args.estimate:
        estimate(source_dir, args.clip_duration, args.stride,
                 args.resolution, args.fps)
        return

    keyframes_dir = output_dir / "keyframes"
    clips_dir = output_dir / "clips"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    videos = find_source_videos(source_dir)
    if args.limit:
        videos = videos[:args.limit]

    workers = args.workers or os.cpu_count()

    print(f"Keyframe-to-Animation Dataset Builder")
    print(f"  Source:       {source_dir} ({len(videos)} videos)")
    print(f"  Output:       {output_dir}")
    print(f"  Clip:         {args.clip_duration}s at {args.fps}fps "
          f"({int(args.clip_duration * args.fps)} frames)")
    print(f"  Stride:       {args.stride}s")
    print(f"  Resolution:   {args.resolution}px (short side)")
    print(f"  Quality:      CRF {args.quality}")
    print(f"  Workers:      {workers}")
    print()

    # Phase 1: Scan all videos and plan extraction pairs
    print("Scanning source videos and planning pairs...")
    work_items, metadata_map = plan_pairs(
        videos, output_dir, args.clip_duration, args.stride, args.resume)
    print(f"  Planned {len(work_items)} pairs to extract\n")

    if not work_items:
        print("Nothing to do.")
        return

    # Phase 2: Extract in parallel (each work item = 2 ffmpeg calls)
    # Append common params to each work item
    full_items = [
        (vi, st, pid, kfp, clp,
         args.clip_duration, args.fps, args.resolution, args.quality)
        for vi, st, pid, kfp, clp in work_items
    ]

    manifest_path = output_dir / "manifest.jsonl"
    start_time = time.time()
    completed = 0
    errors = 0

    with open(manifest_path, "a" if args.resume else "w") as mf:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_single_pair, item): item[2]
                       for item in full_items}

            for future in as_completed(futures):
                pair_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        entry = metadata_map[result]
                        mf.write(json.dumps(entry) + "\n")
                        completed += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1

                total_done = completed + errors
                if total_done % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = total_done / elapsed
                    remaining = len(full_items) - total_done
                    eta_min = remaining / rate / 60 if rate > 0 else 0
                    mf.flush()
                    print(f"  [{total_done}/{len(full_items)}] "
                          f"{completed} ok, {errors} err | "
                          f"{rate:.0f} pairs/s | "
                          f"ETA: {eta_min:.0f}min")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"  Pairs extracted:  {completed}")
    print(f"  Errors:           {errors}")
    print(f"  Manifest:         {manifest_path}")
    print(f"  Elapsed:          {elapsed/3600:.1f} hours")
    kf_count = len(list(keyframes_dir.glob("*.jpg")))
    cl_count = len(list(clips_dir.glob("*.mp4")))
    print(f"  Keyframes:        {kf_count}")
    print(f"  Clips:            {cl_count}")
    try:
        total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        print(f"  Total size:       {total_bytes/1024/1024/1024:.1f} GB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
