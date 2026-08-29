"""Quality-tier assignment from community score/favorites."""
from native_animation.data.tiers import assign_tiers


def _posts(n=100):
    return [{"post_id": i, "score": i, "favorite_count": 0} for i in range(n)]


def test_tier_proportions_and_ordering():
    tiers = assign_tiers(_posts(), s_quantile=0.95, a_quantile=0.70)
    assert tiers[99] == "S" and tiers[96] == "S"
    assert tiers[80] == "A" and tiers[71] == "A"
    assert tiers[10] == "B"
    counts = {t: list(tiers.values()).count(t) for t in "SAB"}
    assert counts["S"] == 5 and counts["A"] == 25 and counts["B"] == 70


def test_favorites_break_score_ties():
    posts = [{"post_id": 1, "score": 10, "favorite_count": 9},
             {"post_id": 2, "score": 10, "favorite_count": 1}] + \
            [{"post_id": i + 10, "score": 0, "favorite_count": 0} for i in range(38)]
    tiers = assign_tiers(posts, s_quantile=0.975, a_quantile=0.9)
    assert tiers[1] == "S" and tiers[2] != "S"
