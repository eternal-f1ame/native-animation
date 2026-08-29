"""Curriculum: difficulty gate, rebalance weights, resampling behavior."""
from native_animation.training.curriculum import (
    CurriculumSampler,
    curriculum_weight,
    difficulty,
    rebalance_weights,
)


def _rows():
    rows = []
    for i in range(400):
        rows.append({"q_motion": (i % 4) + 1, "q_deform": (i // 100) + 1,
                     "series": f"s{i % 8}"})
    return rows


def test_difficulty_normalizes_to_unit_interval():
    assert difficulty({"q_motion": 1, "q_deform": 1}) == 0.0
    assert difficulty({"q_motion": 4, "q_deform": 4}) == 1.0
    assert abs(difficulty({"q_motion": 1, "q_deform": 4}) - 0.5) < 1e-9


def test_curriculum_gate_opens_with_tau():
    hard = 1.0
    assert curriculum_weight(hard, tau=0.0) < 0.01
    assert curriculum_weight(hard, tau=1.0) > 0.85
    easy = 0.0
    assert curriculum_weight(easy, tau=0.0) > 0.85


def test_rebalance_upweights_rare_series():
    rows = [{"series": "big", "q_motion": 1}] * 90 + [{"series": "small", "q_motion": 1}] * 10
    weights = rebalance_weights(rows, axis_cols=("series",), exponent=0.7)
    assert weights[-1] > weights[0]


def test_sampler_hard_fraction_grows_with_tau():
    rows = _rows()
    sampler = CurriculumSampler(rows, seed=0)

    def hard_fraction(tau, draws=4000):
        sampler.refresh(tau)
        hard = sum(difficulty(rows[sampler.sample_index()]) > 0.66 for _ in range(draws))
        return hard / draws

    assert hard_fraction(0.05) < hard_fraction(0.95) - 0.1
