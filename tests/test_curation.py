"""Curation scoring: static, blur, and verdict composition."""
import numpy as np

from native_animation.data.curation import blur_score, curation_verdict, static_score

CFG = {"min_mean_framediff": 1.0, "min_laplacian_var": 20.0}


def _frames(n=4, value=128, jitter=0):
    rng = np.random.default_rng(0)
    base = np.full((64, 64), value, dtype=np.int64)
    return [
        np.clip(base + (rng.integers(-jitter, jitter + 1, (64, 64)) if jitter else 0), 0, 255).astype(np.uint8)
        for _ in range(n)
    ]


def test_static_score_zero_for_identical_frames():
    assert static_score(_frames()) == 0.0
    assert static_score(_frames(jitter=30)) > 1.0


def test_blur_score_orders_sharp_above_flat():
    flat = _frames()
    checker = [np.indices((64, 64)).sum(0).astype(np.uint8) % 2 * 255] * 4
    assert blur_score(checker) > blur_score(flat)


def test_verdict_reasons():
    v = curation_verdict(static=0.1, blur=100.0, duration_s=5.0, cfg=CFG)
    assert v == {"pass": False, "reasons": ["static"]}
    v = curation_verdict(static=5.0, blur=5.0, duration_s=5.0, cfg=CFG)
    assert v == {"pass": False, "reasons": ["blur"]}
    assert curation_verdict(5.0, 100.0, 5.0, CFG) == {"pass": True, "reasons": []}
