"""Shot windowing for the Stage-0 pipeline.

Scene boundaries come from PySceneDetect (in tools/split_shots.py); this module
holds the pure windowing math so it is testable without video IO.
"""
from __future__ import annotations


def plan_shot_windows(
    scene_list_s: list[tuple[float, float]],
    min_s: float = 2.2,
    max_s: float = 10.0,
) -> list[tuple[float, float]]:
    """Turn detected scenes into training-clip windows.

    Scenes shorter than ``min_s`` (cannot supply 49 raw frames at 24 fps) are
    dropped; scenes longer than ``max_s`` are tiled without overlap and a
    trailing remainder is kept only if it is itself >= ``min_s``.
    """
    windows: list[tuple[float, float]] = []
    for start, end in scene_list_s:
        pos = start
        while end - pos >= min_s:
            length = min(max_s, end - pos)
            windows.append((round(pos, 3), round(pos + length, 3)))
            pos += length
    return windows
