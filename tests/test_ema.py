import pytest
import torch

from mimir_decide.train import ema_decay_at, ema_scope, ema_update, swap_with_ema


def test_decay_warms_up_and_saturates():
    assert ema_decay_at(0, 0.999) == pytest.approx(0.1)
    assert ema_decay_at(10, 0.999) == pytest.approx(0.55)
    assert ema_decay_at(100000, 0.999) == 0.999


def test_ema_update_is_the_exponential_average():
    p = [torch.tensor([1.0, 2.0]), torch.tensor(3.0)]
    ema = [t.clone() for t in p]
    for t in p:
        t.add_(1.0)                                   # the parameters move by +1
    ema_update(ema, p, 0.9)
    assert torch.allclose(ema[0], torch.tensor([1.1, 2.1])) and torch.allclose(ema[1], torch.tensor(3.1))
    for _ in range(200):
        ema_update(ema, p, 0.9)                       # constant parameters: the average converges to them
    assert torch.allclose(ema[0], p[0], atol=1e-5)


def test_swap_is_an_involution_and_scope_restores_even_after_an_error():
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    ema = [torch.tensor([10.0, 20.0])]
    swap_with_ema([w], ema)
    assert torch.equal(w.data, torch.tensor([10.0, 20.0])) and torch.equal(ema[0], torch.tensor([1.0, 2.0]))
    swap_with_ema([w], ema)
    assert torch.equal(w.data, torch.tensor([1.0, 2.0]))
    with pytest.raises(RuntimeError):
        with ema_scope([w], ema):
            assert torch.equal(w.data, torch.tensor([10.0, 20.0]))
            raise RuntimeError("boom")
    assert torch.equal(w.data, torch.tensor([1.0, 2.0]))           # restored


def test_optimizer_state_survives_a_swap():
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = torch.optim.AdamW([w], lr=0.1)
    w.sum().backward(); opt.step()
    state_before = opt.state[w]["exp_avg"].clone()
    ema = [w.detach().clone() * 0]
    with ema_scope([w], ema):
        pass
    assert torch.equal(opt.state[w]["exp_avg"], state_before)
    w.grad = torch.ones_like(w); opt.step()                         # still trainable afterwards
