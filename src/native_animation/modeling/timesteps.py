"""Timestep density: keep training emphasis where line art and identity are
decided, instead of the heavy-noise regime that erases the drawing.

sigma ~ shift_map(LogitNormal(m, s)) with a small uniform tail on
[tail_lo, 1) for low-SNR coverage (spec §1.4). Sampled per-sample, replacing
the scheduler-grid timesteps of v1.
"""
from __future__ import annotations

import torch


class TimestepDensity:
    def __init__(self, m: float = 0.0, s: float = 1.0, shift: float = 3.0,
                 tail_p: float = 0.05, tail_lo: float = 0.95):
        self.m, self.s, self.shift = m, s, shift
        self.tail_p, self.tail_lo = tail_p, tail_lo

    def shift_map(self, u: torch.Tensor) -> torch.Tensor:
        return self.shift * u / (1 + (self.shift - 1) * u)

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        normal = torch.randn(n, generator=generator) * self.s + self.m
        u = torch.sigmoid(normal)                      # LogitNormal(m, s)
        sigma = self.shift_map(u)
        if self.tail_p > 0:
            tail = torch.rand(n, generator=generator) < self.tail_p
            uniform_hi = self.tail_lo + torch.rand(n, generator=generator) * (1 - self.tail_lo)
            sigma = torch.where(tail, uniform_hi, sigma)
        return sigma.clamp(1e-4, 1 - 1e-4)
