"""Runner v2 pure parts: LR schedule shape and EMA tracking."""
import torch

from native_animation.training.runner_v2 import EMA, cosine_warmup_lambda


def test_cosine_warmup_shape():
    lam = cosine_warmup_lambda(warmup_steps=10, total_steps=110)
    assert lam(0) == 0.0
    assert abs(lam(5) - 0.5) < 1e-9
    assert abs(lam(10) - 1.0) < 1e-9
    assert lam(60) < 1.0 and lam(60) > lam(110)
    assert abs(lam(110) - 0.1) < 0.02        # cosine floor


def test_ema_tracks_parameters():
    p = [torch.nn.Parameter(torch.zeros(3))]
    ema = EMA(p, decay=0.5)
    with torch.no_grad():
        p[0].add_(1.0)
    ema.update(p)
    assert torch.allclose(ema.shadow[0], torch.full((3,), 0.5))
    ema.update(p)
    assert torch.allclose(ema.shadow[0], torch.full((3,), 0.75))
    ema.copy_to(p)
    assert torch.allclose(p[0].data, torch.full((3,), 0.75))
    state = ema.state_dict()
    ema2 = EMA([torch.nn.Parameter(torch.zeros(3))], decay=0.5)
    ema2.load_state_dict(state)
    assert torch.allclose(ema2.shadow[0], ema.shadow[0])
