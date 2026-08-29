"""Curriculum: a controlled migration from near-physical motion to full sakuga.

Difficulty = mean normalized (motion, deformation) quantile bucket; sampling
probability = sigmoid gate x long-tail rebalance weight (spec §2 Stage 2,
AniMatrix Eq. 16 + Eq. 2 adapted to our metadata columns).
"""
from __future__ import annotations

import math
import random
from collections import Counter

import torch


def difficulty(row: dict, bucket_cols=("q_motion", "q_deform"), q: int = 4) -> float:
    values = [(int(row[c]) - 1) / (q - 1) for c in bucket_cols]
    return sum(values) / len(values)


def curriculum_weight(d: float, tau: float, gamma: float = 8.0, beta: float = 0.25) -> float:
    """Sigmoid gate: easy samples open early, the hardest by tau ~ 1 - beta."""
    return 1.0 / (1.0 + math.exp(-gamma * (tau - d + beta)))


def rebalance_weights(rows, axis_cols=("series", "q_motion"), exponent: float = 0.7):
    """AniMatrix-style inverse-marginal-count weights, flattened by ``exponent``."""
    counts = {col: Counter(str(row[col]) for row in rows) for col in axis_cols}
    weights = []
    for row in rows:
        product = 1.0
        for col in axis_cols:
            product *= 1.0 / counts[col][str(row[col])]
        weights.append(product ** exponent)
    return weights


class CurriculumSampler:
    def __init__(self, rows, seed: int = 0, bucket_cols=("q_motion", "q_deform"),
                 axis_cols=("series", "q_motion"), gamma: float = 8.0, beta: float = 0.25):
        self.rows = rows
        self.rng = random.Random(seed)
        self.difficulties = [difficulty(r, bucket_cols) for r in rows]
        self.rebalance = rebalance_weights(rows, axis_cols)
        self.gamma, self.beta = gamma, beta
        self.refresh(tau=0.0)

    def refresh(self, tau: float) -> None:
        raw = [curriculum_weight(d, tau, self.gamma, self.beta) * w
               for d, w in zip(self.difficulties, self.rebalance)]
        total = sum(raw)
        self.cumulative = []
        acc = 0.0
        for value in raw:
            acc += value / total
            self.cumulative.append(acc)

    def sample_index(self) -> int:
        u = self.rng.random()
        lo, hi = 0, len(self.cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cumulative[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo


class CurriculumDataset(torch.utils.data.Dataset):
    """Wrap a base dataset: every __getitem__ draws by curriculum probability.

    DiffSynth's loader shuffles indices, but since every access resamples,
    the shuffle costs nothing and the effective distribution is the sampler's.
    """

    def __init__(self, base, sampler: CurriculumSampler):
        self.base, self.sampler = base, sampler

    def __len__(self):
        return len(self.base)

    def __getitem__(self, _index):
        return self.base[self.sampler.sample_index()]
