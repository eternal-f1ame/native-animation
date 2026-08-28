"""Aggregate metric behavior on synthetic score curves (no CLIP, no video IO)."""
import numpy as np

from native_animation.evaluation.evaluate import (
    classify_result,
    continuation_fidelity,
    diffusion_failure_score,
    final_score,
    temporal_consistency,
    worst_segment,
)


def _flat(value: float = 0.9, n: int = 50) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def test_flat_curve_is_healthy():
    scores = _flat()
    dfs, *_ = diffusion_failure_score(scores)
    assert dfs == 0.0
    assert np.isclose(temporal_consistency(scores), 0.9, atol=1e-6)
    assert np.isclose(worst_segment(scores), 0.9, atol=1e-6)
    assert np.isclose(continuation_fidelity(scores), 0.9, atol=1e-6)


def test_mid_clip_collapse_trips_dfs():
    collapse = np.concatenate(
        [np.full(17, 0.9), np.full(16, 0.4), np.full(17, 0.9)]
    ).astype(np.float32)
    dfs, mid_drop, _, start, middle, end = diffusion_failure_score(collapse)
    assert dfs >= 0.7  # verified 0.777 on the live implementation
    assert mid_drop > 0.3
    assert middle < start and middle < end


def test_jitter_hurts_tcs_more_than_dfs():
    jitter = np.tile([0.9, 0.7], 25).astype(np.float32)
    dfs, *_ = diffusion_failure_score(jitter)
    assert 0.25 <= dfs <= 0.5  # smoothness term only; no mid-drop
    assert temporal_consistency(jitter) < temporal_consistency(_flat(0.8))


def test_worst_segment_finds_the_dip():
    curve = np.concatenate([np.full(10, 0.9), np.full(5, 0.2), np.full(10, 0.9)]).astype(np.float32)
    assert np.isclose(worst_segment(curve, window=5), 0.2, atol=1e-6)


def test_final_score_formula_and_classification():
    assert np.isclose(final_score(1.0, 1.0, 1.0, 0.0), 0.85)
    assert np.isclose(final_score(0.0, 0.0, 0.0, 1.0), -0.5)
    failing = {"CFS": 0.8, "TCS": 0.8, "WorstSegment": 0.8, "DFS": 0.75}
    assert classify_result(failing) == "[FAIL] Diffusion failure"
