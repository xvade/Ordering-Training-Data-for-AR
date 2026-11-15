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


def test_probe_gradient_scorer_direction_and_scores(toy):
    from arorder.samplers import ProbeGradientScorer

    model, x, y = toy

    class Probe:  # minimal stand-in for GPUData
        device = DEV

        def __len__(self):
            return 12

        def batch(self, idx, augment=False, generator=None):
            return x[:12][idx], y[:12][idx]

    scorer = ProbeGradientScorer(Probe(), eps=2.0, pgd_steps=2, probe_batch=12, mode="rob")
    scores = scorer(model, x[12:], y[12:], step=0)
    # "rob" direction is orthogonal to the clean probe gradient ...
    g_clean = batch_gradient(model, x[:12], y[:12])
    cos = (scorer.direction @ g_clean) / (scorer.direction.norm() * g_clean.norm())
    assert abs(cos.item()) < 1e-3
    # ... and scores are the per-sample gradient products with it.
    dots, _ = per_sample_gradient_products(model, x[12:], y[12:], scorer.direction[None])
    assert torch.allclose(scores, dots[:, 0], rtol=1e-3, atol=1e-5)


def test_lin_radius_gradient_predicts_radius_change(toy):
    """A small step against the lin-radius gradient raises the mean linearised radius as predicted."""
    model, x, y = toy
    ev = ProbeEvaluator(x, y, [1 / 255], steps=2, seed=0)
    g = ev.lin_radius_gradient(model)
    theta, eta = flat_params(model), 1e-4 / g.norm().item()
    before = (ev.linearized_radius(model) * 255).clamp(-16, 16).mean().item()
    set_flat_params(model, theta - eta * g)
    after = (ev.linearized_radius(model) * 255).clamp(-16, 16).mean().item()
    set_flat_params(model, theta)
    assert after - before == pytest.approx(eta * (g @ g).item(), rel=0.2)


def test_smoothed_error_gradient_matches_finite_difference(toy):
    model, x, y = toy
    ev = ProbeEvaluator(x, y, [1 / 255], steps=2, seed=0)
    g = ev.loss_gradient(model, 0.0, "serr", temperature=0.5)
    theta, eta = flat_params(model), 1e-3 / g.norm().item()

    def serr():
        with torch.no_grad():
            from arorder.attacks import margin as mg
            return torch.sigmoid(-mg(model(x), y) / 0.5).mean().item()

    before = serr()
    set_flat_params(model, theta - eta * g)
    after = serr()
    set_flat_params(model, theta)
    assert after - before == pytest.approx(-eta * (g @ g).item(), rel=0.2)


def test_class_balanced_sampler_balances_every_batch_and_covers_data():
    from arorder.samplers import ClassBalancedSampler

    labels = torch.arange(10).repeat_interleave(50)  # 500 samples, 50 per class
    s = ClassBalancedSampler(labels, batch_size=40, seed=0)
    seen = []
    for step in range(12):  # 480 samples: less than one pass
        idx = s.next_batch(step)
        counts = torch.bincount(labels[idx], minlength=10)
        assert counts.min() >= 3 and counts.max() <= 5
        seen.append(idx)
    seen = torch.cat(seen)
    assert len(seen.unique()) == len(seen)  # no repeats within the first pass


def _tiny_cfg(tmp_path, **kw):
    from arorder.train import TrainConfig

    base = dict(run_dir=str(tmp_path), model="smallcnn", width=8, epochs=1, batch_size=32, eval_every=1000,
                eval_eps=[1.0, 2.0], eval_steps=2, ema_decay=0.9, stop_step=6)
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture
def tiny_data():
    """A 400-image stand-in for the CIFAR-10 bundle (random pixels, 10 classes)."""
    from arorder.data import DataBundle, GPUData, Splits

    rng = np.random.default_rng(0)

    def make(n):
        return GPUData(rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, n), DEV)

    return DataBundle(train=make(256), probe=make(64), val=make(64), test=make(64),
                      splits=Splits(np.arange(256), np.arange(64), np.arange(64)), num_classes=10)


def test_trainer_snapshot_restore_rewinds_the_run(tmp_path, tiny_data):
    from arorder.train import Trainer

    t = Trainer(_tiny_cfg(tmp_path), data=tiny_data)
    for _ in range(2):
        t.train_step()
    snap = t.snapshot()
    before = flat_params(t.model).clone()
    for _ in range(3):
        t.train_step()
    assert t.step == 5 and not torch.equal(flat_params(t.model), before)
    t.restore(snap)
    assert t.step == 2 and torch.equal(flat_params(t.model), before) and len(t.log.batch_idx) == 2


def test_lookahead_keeps_the_best_scoring_candidate(tmp_path, tiny_data):
    from arorder.train import Trainer

    cfg = _tiny_cfg(tmp_path, lookahead_k=3, lookahead_len=3, lookahead_until_epoch=1.0, lookahead_eps=[1.0, 2.0])
    t = Trainer(cfg, data=tiny_data)
    final = t.fit()
    assert t.step == 6 and "ema_clean_acc" in final and len(t.log.batch_idx) == 6
    for seg in t.lookahead_log:
        scores = [c["score"] for c in seg["candidates"]]
        assert len(scores) == 3 and seg["chosen"] == int(np.argmax(scores))
    # the final state is the chosen candidate's: its score is reproduced by scoring the current weights
    assert t.selection_score()[0] == pytest.approx(max(c["score"] for c in t.lookahead_log[-1]["candidates"]), abs=0.05)


def test_adversarial_training_step_runs_and_changes_weights(tmp_path, tiny_data):
    from arorder.train import Trainer

    t = Trainer(_tiny_cfg(tmp_path, train_eps=4.0, train_attack_steps=2), data=tiny_data)
    before = flat_params(t.model).clone()
    t.train_step()
    assert not torch.equal(flat_params(t.model), before)


def test_nonneg_least_squares_recovers_a_nonnegative_combination():
    from arorder.samplers import nonneg_least_squares

    torch.manual_seed(0)
    G = torch.randn(20, 200, device=DEV)
    w_true = torch.zeros(20, device=DEV)
    w_true[[1, 5, 9]] = torch.tensor([0.5, 1.0, 2.0], device=DEV)
    t = w_true @ G
    w = nonneg_least_squares(G @ G.T, G @ t, iters=2000)
    assert (w >= 0).all() and torch.allclose(w @ G, t, atol=1e-2 * t.norm().item())


def test_systematic_counts_sum_and_track_weights():
    from arorder.samplers import systematic_counts

    w = torch.tensor([0.0, 1.0, 3.0, 0.5, 0.0, 5.5])
    c = systematic_counts(w, 40, torch.Generator().manual_seed(0))
    assert c.sum() == 40 and (c[w == 0] == 0).all()
    assert ((c.float() - 40 * w / w.sum()).abs() < 1).all()


def test_gradient_match_batch_is_closer_to_target_than_uniform(tmp_path, tiny_data):
    from arorder.samplers import GradientMatchSampler
    from arorder.train import Trainer

    t = Trainer(_tiny_cfg(tmp_path), data=tiny_data)
    for _ in range(3):
        t.train_step()
    s = GradientMatchSampler(tiny_data.train, tiny_data.probe, batch_size=32, pool_mult=4, eps=8.0, pgd_steps=2,
                             target_n=64, seed=0)
    idx = s.next_batch(0, t.model)
    st = s.stats()
    assert len(idx) == 32
    # candidates and target samples are disjoint parts of one draw; the batch comes from the candidates
    draw = GradientMatchSampler(tiny_data.train, tiny_data.probe, 32, 4, target_n=64, seed=0).base.next_batch(0)
    assert len(draw) == 4 * 32 + 64 and set(idx.tolist()) <= set(draw[:128].tolist())
    # projecting onto the cone of candidate gradients cannot be further from the target than their plain mean
    assert st["sel/cos_fit"] >= st["sel/cos_uniform"] - 1e-3


def test_project_capped_simplex_satisfies_constraints():
    from arorder.samplers import project_capped_simplex

    v = torch.randn(50, generator=torch.Generator().manual_seed(0)) * 3
    w = project_capped_simplex(v, total=10.0, cap=1.0)
    assert w.min() >= 0 and w.max() <= 1 + 1e-6 and w.sum().item() == pytest.approx(10.0, abs=1e-3)
    # a point already in the set is left where it is
    inside = torch.full((50,), 0.2)
    assert torch.allclose(project_capped_simplex(inside, 10.0, 1.0), inside, atol=1e-4)


def test_capped_fit_improves_on_uniform_and_respects_cap():
    from arorder.samplers import capped_direction_fit, systematic_counts

    torch.manual_seed(0)
    G = torch.randn(64, 300, device=DEV)
    t = G[:8].sum(0) + 0.1 * torch.randn(300, device=DEV)  # a direction that 8 of the candidates point along
    K, b = G @ G.T, G @ t
    w = capped_direction_fit(K, b, t.norm().item(), total=16.0, cap=1.0)
    assert w.min() >= 0 and w.max() <= 1 + 1e-5 and w.sum().item() == pytest.approx(16.0, abs=1e-2)
    cos = lambda w_: (b @ w_ / ((w_ @ K @ w_).sqrt() * t.norm())).item()
    assert cos(w) > cos(torch.full_like(w, 16 / 64)) + 0.1
    counts = systematic_counts(w.cpu(), 16, torch.Generator().manual_seed(0))
    assert counts.sum() == 16 and counts.max() <= 1  # a batch of distinct samples


def test_gradient_match_with_coordinate_subset(tmp_path, tiny_data):
    from arorder.samplers import GradientMatchSampler, per_sample_gradient_matrix
    from arorder.train import Trainer

    t = Trainer(_tiny_cfg(tmp_path), data=tiny_data)
    model = t.model.eval()
    x, y = tiny_data.train.batch(torch.arange(16, device=DEV))
    full = per_sample_gradient_matrix(model, x, y)
    coords = torch.tensor([0, 5, 100, 2000], device=DEV)
    assert torch.allclose(per_sample_gradient_matrix(model, x, y, chunk=5, coords=coords), full[:, coords], atol=1e-6)
    s = GradientMatchSampler(tiny_data.train, tiny_data.probe, batch_size=32, pool_mult=4, eps=8.0, pgd_steps=2,
                             target_n=64, n_coords=512, max_repeat=1, seed=0)
    idx = s.next_batch(0, t.model)
    assert len(idx) == 32 and len(set(idx.tolist())) == 32 and len(s._coords) == 512


def test_project_capped_simplex_batched_matches_reference():
    """The vectorised projection agrees with a plain scalar bisection, row by row."""
    from arorder.samplers import project_capped_simplex

    v = torch.randn(4, 60, generator=torch.Generator().manual_seed(1)) * 2
    w = project_capped_simplex(v, total=12.0, cap=1.0)
    for row, out in zip(v, w):
        lo, hi = (row.min() - 1).item(), row.max().item()
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if (row - mid).clamp(0, 1).sum() > 12 else (lo, mid)
        assert torch.allclose(out, (row - lo).clamp(0, 1), atol=2e-3)
        assert out.sum().item() == pytest.approx(12.0, abs=2e-2)
