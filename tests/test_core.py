"""Unit tests for the measurement machinery. They use synthetic data and run in seconds.

Run inside a Slurm job:  source scripts/env.sh && $PY -m pytest -q tests
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from arorder.attacks import margin, pgd_linf
from arorder.data import make_splits, random_crop_flip
from arorder.investigate import batch_gradient, per_sample_gradient_products
from arorder.models import build_model, flat_params, set_flat_params
from arorder.robustness import ProbeEvaluator
from arorder.samplers import Band, PoolFilterSampler, build_scorer, directional_derivatives, score_loss

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def toy():
    torch.manual_seed(0)
    model = build_model("smallcnn", 10, width=8, norm="gn").to(DEV).eval()
    x = torch.rand(24, 3, 32, 32, device=DEV)
    y = torch.randint(0, 10, (24,), device=DEV)
    return model, x, y


def test_splits_are_disjoint_and_cover():
    s = make_splits(1000, n_probe=100, n_val=150, seed=3)
    allidx = np.concatenate([s.train, s.probe, s.val])
    assert len(allidx) == 1000 and len(np.unique(allidx)) == 1000
    s2 = make_splits(1000, n_probe=100, n_val=150, seed=3)
    assert np.array_equal(s.probe, s2.probe) and np.array_equal(s.val, s2.val)


def test_random_crop_flip_matches_reference():
    x = torch.rand(16, 3, 32, 32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(5)
    out = random_crop_flip(x, generator=g)
    # Re-draw the same random numbers and apply the transform naively.
    g = torch.Generator(device=DEV).manual_seed(5)
    i = torch.randint(0, 9, (16,), device=DEV, generator=g)
    j = torch.randint(0, 9, (16,), device=DEV, generator=g)
    flip = torch.rand(16, device=DEV, generator=g) < 0.5
    xp = F.pad(x, (4, 4, 4, 4))
    for b in range(16):
        ref = xp[b, :, i[b] : i[b] + 32, j[b] : j[b] + 32]
        if flip[b]:
            ref = ref.flip(-1)
        assert torch.equal(out[b], ref)


def test_margin_sign_matches_correctness(toy):
    model, x, y = toy
    logits = model(x)
    assert torch.equal(margin(logits, y) > 0, logits.argmax(1) == y)


def test_pgd_respects_constraints_and_is_deterministic(toy):
    model, x, y = toy
    eps = 4 / 255
    init = torch.rand_like(x) * 2 - 1
    xa = pgd_linf(model, x, y, eps, steps=5, init=init)
    assert (xa - x).abs().max() <= eps + 1e-6
    assert xa.min() >= 0 and xa.max() <= 1
    assert torch.equal(xa, pgd_linf(model, x, y, eps, steps=5, init=init))
    # The attack must not raise the margin relative to the clean input on average.
    assert margin(model(xa), y).mean() < margin(model(x), y).mean()
    assert torch.equal(pgd_linf(model, x, y, 0.0), x)


def test_pgd_keep_best_never_worse_than_start(toy):
    model, x, y = toy
    init = torch.rand_like(x) * 2 - 1
    eps = 2 / 255
    start = (x + eps * init).clamp(0, 1)
    xa = pgd_linf(model, x, y, eps, steps=3, init=init, keep_best=True)
    assert (margin(model(xa), y) <= margin(model(start), y) + 1e-5).all()


def test_flat_param_roundtrip(toy):
    model, x, _ = toy
    before = model(x).detach().clone()
    theta = flat_params(model)
    set_flat_params(model, theta + 1.0)
    assert not torch.allclose(model(x), before)
    set_flat_params(model, theta)
    assert torch.equal(model(x), before)


def test_per_sample_products_match_explicit_gradients(toy):
    model, x, y = toy
    P = flat_params(model).numel()
    V = torch.randn(3, P, device=DEV)
    dots, norms = per_sample_gradient_products(model, x, y, V, chunk=7)
    for i in (0, 5, 23):
        g = batch_gradient(model, x[i : i + 1], y[i : i + 1])
        assert torch.allclose(dots[i], V @ g, rtol=1e-3, atol=1e-4)
        assert torch.allclose(norms[i], g.norm(), rtol=1e-3)
    # The batch gradient is the mean of per-sample gradients.
    assert torch.allclose(dots.mean(0), V @ batch_gradient(model, x, y), rtol=1e-3, atol=1e-4)


def test_first_order_prediction_matches_small_real_step(toy):
    """A tiny real step changes the held-out clean loss by -eta * <grad Q, g>."""
    model, x, y = toy
    ev = ProbeEvaluator(x[:12], y[:12], [1 / 255], steps=3, seed=0)
    gq = ev.loss_gradient(model, 0.0)
    g = batch_gradient(model, x[12:], y[12:])
    theta, eta = flat_params(model), 1e-3
    before = ev.evaluate(model)["clean_loss"]
    set_flat_params(model, theta - eta * g)
    after = ev.evaluate(model)["clean_loss"]
    set_flat_params(model, theta)
    pred = -eta * (gq @ g).item()
    assert after - before == pytest.approx(pred, rel=0.15, abs=1e-6)


def test_probe_evaluator_is_paired_and_consistent(toy):
    model, x, y = toy
    ev = ProbeEvaluator(x, y, [1 / 255, 2 / 255], steps=3, seed=0)
    a, b = ev.evaluate(model), ev.evaluate(model)
    # Identical weights and identical starts give the same numbers up to GPU
    # floating-point non-determinism (cuDNN convolutions are not bit-reproducible),
    # which can occasionally flip one signed-gradient step of the attack.
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=2e-2 if "acc" in k or k == "auc" else 1e-2), k
    assert a["adv_acc@2"] <= a["adv_acc@1"] <= a["clean_acc"]
    assert a["adv_loss@2"] >= a["adv_loss@1"] >= a["clean_loss"]
    assert 0 <= a["auc"] <= 2


def test_directional_derivatives_match_per_sample_gradients(toy):
    model, x, y = toy
    V = torch.randn(2, flat_params(model).numel(), device=DEV)
    dots, _ = per_sample_gradient_products(model, x, y, V)
    assert torch.allclose(directional_derivatives(model, x, y, V), dots, rtol=1e-3, atol=1e-4)


def test_pool_filter_keeps_best_scoring_candidates(toy):
    model, x, y = toy

    class Data:  # minimal stand-in for GPUData
        device = DEV

        def __len__(self):
            return len(x)

        def batch(self, idx, augment=False, generator=None):
            return x[idx], y[idx]

    sampler = PoolFilterSampler(Data(), batch_size=6, pool_mult=2, scorer=build_scorer("loss:low"), seed=0)
    idx = sampler.next_batch(0, model)
    assert len(idx) == 6 and len(set(idx.tolist())) == 6
    # Reconstruct the pool the sampler drew and check the kept ones have the lowest loss.
    ref = PoolFilterSampler(Data(), batch_size=6, pool_mult=2, scorer=None, seed=0).base.next_batch(0)
    losses = score_loss(model, x[ref], y[ref], 0)
    assert set(idx.tolist()) == set(ref[losses.argsort()[:6]].tolist())
    assert sampler.stats()["sel/score_kept"] >= sampler.stats()["sel/score_pool"]


def test_band_scorer_prefers_middle_of_pool(toy):
    model, x, y = toy
    raw = score_loss(model, x, y, 0)
    kept = Band(score_loss, q=0.5)(model, x, y, 0).topk(8).indices
    ranks = raw.argsort().argsort()[kept]
    assert ranks.min() >= 7 and ranks.max() <= 16
