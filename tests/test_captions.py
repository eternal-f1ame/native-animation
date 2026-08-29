"""Caption prompts and parsing for Qwen3-VL shot annotation."""
from native_animation.data.captions import (
    build_caption_request,
    fallback_caption,
    parse_caption_output,
)

GOOD = """Here is the annotation:
```json
{"subjects": [{"type": "Human", "appearance": "girl in red coat"}],
 "motion": [{"action": "runs left to right", "amplitude": "high"}],
 "AnimeVisualEffects": {"present": true, "effects": ["speed lines"]},
 "style": {"VideoStyle": "2D Japanese Anime", "MotionStyle": "2D Combat"},
 "camera": {"shot_type": "full shot", "camera_motion": "tracking"},
 "environment": "night rooftop"}
```
<tag> VideoStyle: 2D Japanese Anime, MotionStyle: 2D Combat, shot_type: full shot, camera_motion: tracking
<summary> A girl in a red coat sprints across a night rooftop.
<description> A girl in a red coat sprints from left to right across a rooftop at night, trailed by speed lines while the camera tracks her at full shot."""


def test_parse_extracts_all_four_parts():
    out = parse_caption_output(GOOD)
    assert out["structured"]["style"]["VideoStyle"] == "2D Japanese Anime"
    assert out["tag"].startswith("VideoStyle:")
    assert out["summary"].startswith("A girl")
    assert "speed lines" in out["description"]


def test_parse_rejects_missing_sections():
    assert parse_caption_output("no json here") is None
    assert parse_caption_output(GOOD.split("<summary>")[0]) is None


def test_request_embeds_source_tags_and_series():
    req = build_caption_request(tags="smears fighting", series="one_piece")
    assert "smears fighting" in req and "one_piece" in req


def test_fallback_is_a_usable_directive():
    fb = fallback_caption(tags="smears fighting effects", series="bleach_series")
    assert fb["summary"] and "bleach_series" in fb["description"]
    assert fb["structured"] == {}
