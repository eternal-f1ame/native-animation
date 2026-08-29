"""Motion profiling: flow energy, non-rigid residual, quantile buckets."""
import numpy as np

from native_animation.data.profiling import assign_quantile_buckets, flow_energy, nonrigid_residual


def _affine_flow(h=32, w=32, a=0.1, b=0.05, tx=2.0, ty=-1.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([a * xs + b * ys + tx, -b * xs + a * ys + ty], axis=-1)


def test_flow_energy_zero_and_positive():
    assert flow_energy(np.zeros((8, 8, 2))) == 0.0
    assert flow_energy(np.full((8, 8, 2), 3.0)) > 0


def test_pure_affine_flow_has_near_zero_residual():
    # Camera pans/zooms are affine — the residual must ignore them.
    assert nonrigid_residual(_affine_flow()) < 1e-8


def test_localized_deformation_survives_affine_removal():
    flow = _affine_flow()
    flow[10:20, 10:20] += 5.0                    # a smear-like local deformation
    assert nonrigid_residual(flow) > 0.5


def test_quantile_buckets_cover_range():
    buckets = assign_quantile_buckets(list(range(100)), q=4)
    assert min(buckets) == 1 and max(buckets) == 4
    assert buckets[0] == 1 and buckets[-1] == 4
    assert assign_quantile_buckets([5.0, 5.0, 5.0], q=4) == [1, 1, 1]  # ties collapse low
