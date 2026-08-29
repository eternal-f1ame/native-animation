"""Caption schema, prompts, and parsing for Qwen3-VL shot annotation.

Follows the AniMatrix three-section directive format (<tag>/<summary>/
<description>) plus a structured JSON block, produced in a single pass.
Only text logic lives here; model inference is tools/annotate_clips.py.
"""
from __future__ import annotations

import json
import re

CAPTION_SYSTEM_PROMPT = (
    "You are a professional anime production annotator. Given a short animation "
    "clip, output BOTH of the following, exactly in this order:\n"
    "1. A JSON code block with keys: subjects (list of {type, appearance}), "
    "motion (list of {action, amplitude}), AnimeVisualEffects ({present, effects}), "
    "style ({VideoStyle, MotionStyle}), camera ({shot_type, camera_motion}), "
    "environment (string).\n"
    "2. Three tagged lines:\n"
    "<tag> comma-separated key: value pairs for VideoStyle, MotionStyle, shot_type, camera_motion\n"
    "<summary> one sentence summarizing the clip\n"
    "<description> one detailed paragraph, temporally ordered: subject appearance -> "
    "camera -> motion and expression -> visual effects -> environment. Use professional "
    "animation vocabulary (smears, impact frames, speed lines, held frames) where they "
    "genuinely appear. Describe production choices, not just visible content."
)

_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_SECTION_RE = {
    "tag": re.compile(r"<tag>\s*(.+)"),
    "summary": re.compile(r"<summary>\s*(.+)"),
    "description": re.compile(r"<description>\s*(.+?)(?:\n<|\Z)", re.DOTALL),
}


def build_caption_request(tags: str, series: str) -> str:
    return (
        f"Community tags for this clip (noisy but grounding): {tags}\n"
        f"Series: {series}\n"
        "Annotate the clip per the required format."
    )


def parse_caption_output(text: str) -> dict | None:
    """Extract the structured JSON + three sections; None if any part is missing."""
    json_match = _JSON_RE.search(text)
    if not json_match:
        return None
    try:
        structured = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None
    sections = {}
    for key, pattern in _SECTION_RE.items():
        match = pattern.search(text)
        if not match:
            return None
        sections[key] = match.group(1).strip()
    return {"structured": structured, **sections}


def fallback_caption(tags: str, series: str) -> dict:
    """Tag-template directive used when the model output cannot be parsed."""
    tag_list = ", ".join(tags.split()[:12])
    return {
        "structured": {},
        "tag": "",
        "summary": f"An anime clip featuring {tag_list}.",
        "description": f"native animation, anime, {series}, {tag_list}",
    }
