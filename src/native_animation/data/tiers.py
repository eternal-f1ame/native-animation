"""Quality tiers from community signals (spec §2 Stage 0).

Sakugabooru score (community rating of the animation craft) with favorites as
tiebreak stands in for AniMatrix's expert-review tiers: S = top slice used for
Stage-3 preference data, A = high-quality SFT pool, B = broad CT pool.
"""
from __future__ import annotations


def assign_tiers(posts: list[dict], s_quantile: float, a_quantile: float) -> dict[int, str]:
    """post_id -> 'S' | 'A' | 'B', ranked by (score, favorite_count)."""
    ranked = sorted(posts, key=lambda p: (p.get("score", 0), p.get("favorite_count", 0)))
    n = len(ranked)
    s_cut, a_cut = int(n * s_quantile), int(n * a_quantile)
    tiers: dict[int, str] = {}
    for idx, post in enumerate(ranked):
        tiers[post["post_id"]] = "S" if idx >= s_cut else "A" if idx >= a_cut else "B"
    return tiers
