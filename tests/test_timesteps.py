"""Timestep density: logit-normal + shift map + low-SNR tail."""
import torch

from native_animation.modeling.timesteps import TimestepDensity


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_samples_live_in_open_unit_interval():
    sigma = TimestepDensity().sample(10_000, _gen())
    assert sigma.shape == (10_000,)
    assert float(sigma.min()) > 0.0 and float(sigma.max()) < 1.0


def test_shift_map_is_monotone_and_pins_endpoints():
    density = TimestepDensity(shift=3.0)
    u = torch.linspace(0.001, 0.999, 100)
    mapped = density.shift_map(u)
    assert torch.all(mapped[1:] > mapped[:-1])
    assert torch.allclose(density.shift_map(torch.tensor([0.0, 1.0])), torch.tensor([0.0, 1.0]))
    # shift>1 pushes mass toward high sigma: midpoint maps above 0.5
    assert float(density.shift_map(torch.tensor(0.5))) > 0.5


def test_tail_fraction_is_respected():
    sigma = TimestepDensity(tail_p=0.05, tail_lo=0.95).sample(100_000, _gen())
    frac_high = float((sigma >= 0.95).float().mean())
    assert 0.04 < frac_high < 0.10   # 5% forced tail + logit-normal mass that lands there


def test_mean_parameter_moves_the_median():
    low = TimestepDensity(m=-1.0, tail_p=0.0).sample(50_000, _gen(1)).median()
    high = TimestepDensity(m=+1.0, tail_p=0.0).sample(50_000, _gen(1)).median()
    assert float(high) > float(low)
