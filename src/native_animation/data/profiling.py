"""Motion profiling: flow energy (amplitude) and non-rigid residual (deformation).

The residual removes the best-fit global affine field — pans, zooms, and
rotations — so what remains is squash/stretch/smear-style deformation, the
quantity the v2 curriculum schedules on (spec §1/§2).
"""
from __future__ import annotations

import numpy as np


def flow_energy(flow: np.ndarray) -> float:
    """Mean L2 magnitude of an HxWx2 flow field."""
    return float(np.mean(np.linalg.norm(flow, axis=-1)))


def nonrigid_residual(flow: np.ndarray) -> float:
    """Mean magnitude of the flow after removing its least-squares affine fit.

    A pure camera move (pan/zoom/rotation) is an affine field over pixel
    coordinates and fits exactly; localized deformation survives.
    """
    h, w, _ = flow.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    ones = np.ones_like(xs)
    basis = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=1)      # N x 3
    targets = flow.reshape(-1, 2)                                          # N x 2
    coeffs, *_ = np.linalg.lstsq(basis, targets, rcond=None)               # 3 x 2
    residual = targets - basis @ coeffs
    return float(np.mean(np.linalg.norm(residual, axis=1)))


def assign_quantile_buckets(values: list[float], q: int) -> list[int]:
    """Map values to 1..q by corpus quantiles (ties collapse to the low bucket)."""
    arr = np.asarray(values, dtype=np.float64)
    edges = np.quantile(arr, [i / q for i in range(1, q)])
    return [int(np.searchsorted(edges, v, side="left")) + 1 for v in arr]
