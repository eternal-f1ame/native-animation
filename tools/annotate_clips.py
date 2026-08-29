#!/usr/bin/env python3
"""Qwen3-VL shot annotation: structured caption + three-section directive.

Runs under the `anno` env (see configs/anno-env.md). Sharded + idempotent like
the other Stage-0 CLIs.

Transformers fallback (if vLLM is unavailable on this CUDA): replace the vLLM
block with AutoModelForImageTextToText.from_pretrained(model_id,
dtype="bfloat16", device_map="cuda") + AutoProcessor, apply_chat_template on
the same messages, and model.generate(max_new_tokens=1024).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_animation.data.captions import (  # noqa: E402
    CAPTION_SYSTEM_PROMPT,
    build_caption_request,
    fallback_caption,
    parse_caption_output,
)


def load_post_tags(clips_dir: Path) -> dict[int, tuple[str, str]]:
    """post_id -> (flat tags, series) from the corpus sidecars."""
    table: dict[int, tuple[str, str]] = {}
    for sidecar in clips_dir.rglob("*.json"):
        if sidecar.name == "_state.json" or not sidecar.stem.isdigit():
            continue
        try:
            data = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        table[int(sidecar.stem)] = (data.get("tags", ""), sidecar.parent.name)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots-dir", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Cap shots this run (smoke tests).")
    args = parser.parse_args()

    acfg = yaml.safe_load(args.config.read_text())["annotation"]
    manifest = args.shots_dir / "manifests" / f"shard_{args.shard:04d}.jsonl"
    if not manifest.exists():
        print("no manifest shard; nothing to do")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"shard_{args.shard:04d}.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as handle:
            done = {json.loads(line)["shot_id"] for line in handle if line.strip()}

    from vllm import LLM, SamplingParams

    llm = LLM(model=acfg["model_id"], max_model_len=8192, limit_mm_per_prompt={"video": 1})
    params = SamplingParams(temperature=0.2, max_tokens=1024)
    tags_table = load_post_tags(args.clips_dir)

    processed = 0
    with manifest.open() as src, out_path.open("a") as out:
        for line in src:
            rec = json.loads(line)
            if rec["shot_id"] in done or not rec["curation"]["pass"]:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            tags, series = tags_table.get(rec["post_id"], ("", rec["series"]))
            video_path = str(args.shots_dir / rec["video"])
            messages = [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "video", "video": video_path, "num_frames": acfg["frames_per_shot"]},
                    {"type": "text", "text": build_caption_request(tags, series)},
                ]},
            ]
            caption, fallback = None, False
            for _ in range(acfg["max_retries"] + 1):
                result = llm.chat(messages, params)[0].outputs[0].text
                caption = parse_caption_output(result)
                if caption:
                    break
            if caption is None:
                caption, fallback = fallback_caption(tags, series), True
            out.write(json.dumps({"shot_id": rec["shot_id"], **caption, "fallback": fallback}) + "\n")
            out.flush()
            processed += 1
    print(f"[shard {args.shard}] annotated {processed} shots")


if __name__ == "__main__":
    main()
