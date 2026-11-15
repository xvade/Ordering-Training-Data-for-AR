"""Counterfactual study of single updates at a saved checkpoint.

Question: starting from fixed weights ``theta``, which training samples, if
trained on next, raise or lower adversarial robustness on held-out data?

Three complementary measurements are made.

1. **First-order per-sample scores.** A pure gradient step on sample ``i`` with
   step size ``eta`` changes any differentiable held-out quantity ``Q`` by
   ``-eta * <grad Q(theta), grad loss_i(theta)>`` to first order. We take ``Q``
   to be the held-out clean loss and the held-out adversarial loss at several
   radii (its gradient is the loss gradient at the PGD points, by Danskin), and
   compute the inner products for *every* training sample with per-sample
   gradients (``torch.func``). The sign convention used throughout:

       ``gain[Q][i] = <grad Q, grad loss_i>``  > 0  means training on ``i``
       *lowers* ``Q`` (helps), < 0 means it raises ``Q`` (hurts).

2. **Real finite steps on random batches**, at three step sizes, to find out
   where first-order reasoning holds. We apply ``theta - eta * mean_grad(batch)``,
   re-run the attack at the new weights (same fixed starts, so the comparison
   is paired) on the ``val`` split, which played no part in computing the
   scores, and record the true change of every metric.

3. **Filter trials**: the lever a data-ordering rule actually has. Draw a
   candidate pool of ``2 * batch_size`` random samples, keep the half preferred
   by some rule, take a real step on it, and measure. Every rule is applied to
   the same pools, and ``random`` (keep an arbitrary half) is one of the rules,
   so each rule's effect is paired against random selection from the same pool.

Two confounds are handled explicitly (both were observed on 2026-11-15, see
JOURNAL.md):

* *Step size.* Training runs near the edge of stability: a pure step of the
  full learning rate is already outside the linear regime early in training,
  and the eventual momentum displacement ``lr / (1 - momentum)`` applied in one
  shot wrecks the model. The base step is therefore ``eta = eta_mult * lr`` with
  ``eta_mult = 0.25`` by default.
* *Gradient magnitude.* A batch of similar high-loss samples has a far larger
  mean gradient than a random batch, so it takes a far larger step; raw
  comparisons then measure step length, not direction. Since a learning rate
  can always absorb magnitude, every filter trial is run twice: ``raw``
  (``eta * mean_grad``) and ``norm`` (the step rescaled to the median length of
  a random-batch step, isolating the *direction* the rule picks).

All steps are *pure gradient steps* (no momentum history, no weight decay):
they isolate the contribution of the batch itself, which is what a
data-selection rule controls.

Run as a module::

    python -m arorder.investigate --run-dir RUN --step 351 3510
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .attacks import pgd_linf
from .data import load_cifar10
from .models import build_model, flat_params, set_flat_params
from .robustness import ProbeEvaluator
from .train import TrainConfig, environment_info, lr_at


def per_sample_gradient_products(model, x, y, vectors: torch.Tensor, chunk: int = 250, other=None):
    """Inner products of every sample's loss gradient with given directions.

    Args:
        vectors: ``(K, P)`` directions in flat-parameter space.
        other: optional second input tensor shaped like ``x`` (for example the
            adversarial version of each sample). If given, the cosine between
            each sample's gradient at ``x`` and at ``other`` is also returned.

    Returns:
        ``dots`` ``(N, K)`` with ``dots[i, k] = <grad loss_i, vectors[k]>``,
        ``norms`` ``(N,)`` the per-sample gradient norms, and (if ``other`` is
        given) ``cos_other`` ``(N,)`` and ``norms_other`` ``(N,)``.

    The model must be in eval mode and free of batch-coupled layers in training
    mode (per-sample gradients of a BatchNorm network in train mode are not
    well defined).
    """
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b for n, b in model.named_buffers()}

    def loss_one(p, xi, yi):
        return F.cross_entropy(functional_call(model, (p, buffers), (xi[None],)), yi[None])

    per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))

    def flat_grads(xs, ys):
        g = per_sample(params, xs, ys)
        return torch.cat([g[n].flatten(1) for n in names], dim=1)

    dots, norms, cos_o, norms_o = [], [], [], []
    for s in range(0, x.shape[0], chunk):
        flat = flat_grads(x[s : s + chunk], y[s : s + chunk])
        dots.append(flat @ vectors.T)
        norms.append(flat.norm(dim=1))
        if other is not None:
            fo = flat_grads(other[s : s + chunk], y[s : s + chunk])
            norms_o.append(fo.norm(dim=1))
            cos_o.append((flat * fo).sum(1) / (norms[-1] * norms_o[-1]).clamp_min(1e-30))
    if other is None:
        return torch.cat(dots), torch.cat(norms)
    return torch.cat(dots), torch.cat(norms), torch.cat(cos_o), torch.cat(norms_o)


def batch_gradient(model, x, y) -> torch.Tensor:
    """Flat gradient of the mean cross-entropy over a batch (eval mode, no augmentation)."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(F.cross_entropy(model(x), y), params)
    return torch.cat([g.reshape(-1) for g in grads])


def sample_features(model, x, y, eps_list, steps: int, batch_size: int = 1000) -> dict:
    """Intrinsic per-sample quantities at the current weights (no held-out data involved).

    Returns arrays over samples: ``loss``, ``margin``, ``p_true``, ``entropy``,
    ``lin_radius`` (first-order L-inf boundary distance, 1/255 units),
    ``input_grad_l1`` (L1 norm of the loss gradient w.r.t. the pixels) and
    ``adv_margin@e`` (the sample's own margin under PGD at each radius).
    """
    ev = ProbeEvaluator(x, y, eps_list, steps=steps, seed=99, batch_size=batch_size)
    res = ev.evaluate(model, per_example=True)["per_example"]
    out = {k: v for k, v in res.items()}
    loss, p_true, ent, gl1 = [], [], [], []
    for s in range(0, x.shape[0], batch_size):
        xb = x[s : s + batch_size].clone().requires_grad_(True)
        yb = y[s : s + batch_size]
        logits = model(xb)
        l = F.cross_entropy(logits, yb, reduction="none")
        (g,) = torch.autograd.grad(l.sum(), xb)
        logp = logits.detach().log_softmax(1)
        loss.append(l.detach())
        p_true.append(logp.gather(1, yb[:, None]).squeeze(1).exp())
        ent.append(-(logp.exp() * logp).sum(1))
        gl1.append(g.abs().flatten(1).sum(1))
    out.update(loss=torch.cat(loss), p_true=torch.cat(p_true), entropy=torch.cat(ent),
               input_grad_l1=torch.cat(gl1), margin=out.pop("clean_margin"))
    return out


def filter_rules(gains: dict, feats: dict, vec_gram: np.ndarray, names: list, mid: float) -> dict:
    """Per-sample scores for every filter rule; *higher score = kept*.

    Rules that use held-out gradients (first-order gains):

    * ``gain_adv`` / ``gain_clean``: lowers held-out adversarial / clean loss.
    * ``gain_gap``: lowers (adversarial - clean) loss.
    * ``gain_rob``: lowers adversarial loss along the part of the adversarial
      gradient orthogonal to the clean gradient, i.e. robustness at (first
      order) no change in clean loss.
    * ``cos_*``: the same, divided by the sample's gradient norm, so the rule
      prefers a *direction* rather than a large gradient.

    Rules that use only the sample itself (cheap, need no held-out set):
    ``low_loss``, ``high_loss``, ``low_grad_norm``, ``high_lin_radius``,
    ``high_adv_margin``, ``low_adv_margin``, ``high_self_cos`` (the sample's
    clean and adversarial gradients agree) and ``low_self_cos``.
    """
    adv, clean = gains[f"probe_adv@{mid:g}"], gains["probe_clean"]
    ia, ic = names.index(f"probe_adv@{mid:g}"), names.index("probe_clean")
    beta = vec_gram[ia, ic] / vec_gram[ic, ic]  # projection coefficient of g_adv on g_clean
    norm = np.maximum(feats["grad_norm"], 1e-30)
    rules = {
        "gain_adv": adv, "gain_clean": clean, "gain_gap": adv - clean, "gain_rob": adv - beta * clean,
        "low_loss": -feats["loss"], "high_loss": feats["loss"], "low_grad_norm": -feats["grad_norm"],
        "high_lin_radius": feats["lin_radius"],
        "high_adv_margin": feats[f"adv_margin@{mid:g}"], "low_adv_margin": -feats[f"adv_margin@{mid:g}"],
        "high_self_cos": feats["self_cos"], "low_self_cos": -feats["self_cos"],
    }
    for k in ("adv", "clean", "gap", "rob"):
        rules[f"cos_{k}"] = rules[f"gain_{k}"] / norm
    return rules


def investigate_checkpoint(
    run_dir: Path,
    step: int,
    out_path: Path,
    score_eps=(1.0, 2.0, 4.0),
    eval_eps=(1.0, 2.0),
    pgd_steps: int = 10,
    n_random: int = 100,
    n_pools: int = 30,
    batch_size: int = 128,
    seed: int = 0,
    eta_mult: float = 0.25,
) -> None:
    """Run the full counterfactual study for one checkpoint and save one ``.npz``.

    Saved arrays (N = number of train samples):

    * ``feat/<name>`` (N,): intrinsic sample features, see :func:`sample_features`;
      also ``feat/grad_norm``, ``feat/label``, ``feat/self_cos`` (cosine between
      the sample's clean and adversarial loss gradients at the middle radius)
      and ``feat/adv_grad_norm``.
    * ``gain/<Q>`` (N,): first-order gains ``<grad Q, grad loss_i>`` for ``Q`` in
      ``probe_clean``, ``probe_adv@e`` (cross-entropy), ``probe_advm@e``
      (clipped negative margin) and the same on ``val`` for clean and the
      middle radius. Positive = training on the sample lowers ``Q``.
    * ``vec_names`` and ``vec_gram``: names and Gram matrix of the held-out
      gradient directions (their norms and mutual alignment).
    * ``trial/*``: one row per real finite step.
      ``trial/kind`` is ``random`` (random batch, base step), ``scale`` (the
      first 40 random batches at 0.25x and 4x the base step) or ``filter``;
      ``trial/rule`` (filter rule or empty), ``trial/pool`` (pool id or -1),
      ``trial/norm`` (1 if the step was rescaled to the reference length),
      ``trial/step_scale`` (the multiplier actually applied to the mean
      gradient), ``trial/grad_norm`` (norm of the batch mean gradient),
      ``trial/idx`` (the batch), ``trial/pred/<Q>`` (first-order prediction of
      the change of ``Q``) and ``trial/d/<metric>`` (measured change of each
      ``val`` metric).
    * ``base/<metric>``: ``val`` metrics at the unperturbed weights.
    * ``noise/d/<metric>`` (seeds, batches): the same 12 random steps measured
      under 5 different PGD start seeds (the measurement noise floor).
    * ``meta``: JSON string with step, eta, lr, reference step length, host, timing.
    """
    dev = torch.device("cuda")
    cfg = TrainConfig(**json.loads((Path(run_dir) / "config.json").read_text()))
    data = load_cifar10(dev)
    model = build_model(cfg.model, data.num_classes, width=cfg.width, norm=cfg.norm).to(dev)
    ckpt_name = "ckpt_final.pt" if step < 0 else f"ckpt_{step:06d}.pt"
    ckpt = torch.load(Path(run_dir) / ckpt_name, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt["step"]
    steps_per_epoch = len(data.train) // cfg.batch_size
    total_steps = cfg.epochs * steps_per_epoch
    lr = lr_at(min(step, total_steps - 1), cfg, steps_per_epoch, total_steps)
    eta = eta_mult * max(lr, 0.02 * cfg.lr)  # floor: the schedule ends at lr = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    out: dict[str, np.ndarray] = {}

    tx, ty = data.train.batch(torch.arange(len(data.train), device=dev))
    px, py = data.probe.batch(torch.arange(len(data.probe), device=dev))
    vx, vy = data.val.batch(torch.arange(len(data.val), device=dev))
    score_eps_px = [e / 255 for e in score_eps]
    probe_ev = ProbeEvaluator(px, py, score_eps_px, steps=pgd_steps, seed=1234)
    val_score_ev = ProbeEvaluator(vx, vy, score_eps_px, steps=pgd_steps, seed=4321)
    val_ev = ProbeEvaluator(vx, vy, [e / 255 for e in eval_eps], steps=pgd_steps, seed=4321)

    # 1. held-out gradient directions
    mid = score_eps[len(score_eps) // 2]
    vec = {"probe_clean": probe_ev.loss_gradient(model, 0.0)}
    for e in score_eps:
        vec[f"probe_adv@{e:g}"] = probe_ev.loss_gradient(model, e / 255, "ce")
        vec[f"probe_advm@{e:g}"] = probe_ev.loss_gradient(model, e / 255, "neg_margin")
    vec["val_clean"] = val_score_ev.loss_gradient(model, 0.0)
    vec[f"val_adv@{mid:g}"] = val_score_ev.loss_gradient(model, mid / 255, "ce")
    names = list(vec)
    V = torch.stack([vec[k] for k in names])
    out["vec_names"] = np.array(names)
    out["vec_gram"] = (V @ V.T).cpu().numpy()

    # 2. per-sample first-order gains and intrinsic features, all train samples
    g_init = torch.Generator(device=dev).manual_seed(99)
    tx_adv = torch.cat([
        pgd_linf(model, tx[s : s + 1000], ty[s : s + 1000], mid / 255, steps=pgd_steps,
                 init=torch.rand(tx[s : s + 1000].shape, device=dev, generator=g_init) * 2 - 1)
        for s in range(0, len(tx), 1000)
    ])
    dots, norms, self_cos, adv_norms = per_sample_gradient_products(model, tx, ty, V, other=tx_adv)
    del tx_adv
    gains = {name: dots[:, k].cpu().numpy() for k, name in enumerate(names)}
    feats = {"grad_norm": norms.cpu().numpy(), "self_cos": self_cos.cpu().numpy(),
             "adv_grad_norm": adv_norms.cpu().numpy(), "label": ty.cpu().numpy()}
    feats.update({k: v.cpu().numpy() for k, v in sample_features(model, tx, ty, score_eps_px, pgd_steps).items()})
    for k, v in gains.items():
        out[f"gain/{k}"] = v
    for k, v in feats.items():
        out[f"feat/{k}"] = v
    t_scores = time.time() - t0

    # 3. real finite steps, measured on val
    theta0 = flat_params(model)
    base = val_ev.evaluate(model)
    for k, v in base.items():
        out[f"base/{k}"] = np.float32(v)
    trials = []

    def run_trial(kind, idx, step_scale=None, target_len=None, rule="", pool=-1, evaluator=val_ev, base_=base,
                  record=True):
        """One real step on ``idx``. Give ``step_scale`` (multiplier on the mean
        gradient) or ``target_len`` (desired step length). Returns measured changes."""
        it = torch.as_tensor(idx, device=dev)
        g = batch_gradient(model, tx[it], ty[it])
        gnorm = g.norm().item()
        scale = step_scale if target_len is None else target_len / max(gnorm, 1e-30)
        set_flat_params(model, theta0 - scale * g)
        after = evaluator.evaluate(model)
        set_flat_params(model, theta0)
        d = {k: after[k] - base_[k] for k in base_}
        if record:
            trials.append(dict(kind=kind, rule=rule, pool=pool, norm=int(target_len is not None), step_scale=scale,
                               grad_norm=gnorm, idx=np.asarray(idx, np.int32), d=d,
                               pred={name: (-scale * (V[k] @ g)).item() for k, name in enumerate(names)}))
        return d

    n = len(data.train)
    random_batches = [rng.choice(n, batch_size, replace=False) for _ in range(n_random)]
    for idx in random_batches:
        run_trial("random", idx, step_scale=eta)
    for mult in (0.25, 4.0):
        for idx in random_batches[:40]:
            run_trial("scale", idx, step_scale=eta * mult)
    ref_len = float(np.median([t["step_scale"] * t["grad_norm"] for t in trials if t["kind"] == "random"]))

    rules = filter_rules(gains, feats, out["vec_gram"], names, mid)
    out["rule_names"] = np.array(["random"] + list(rules))
    for p in range(n_pools):
        pool = rng.choice(n, 2 * batch_size, replace=False)
        picks = {"random": pool[:batch_size]}
        for rname, score in rules.items():
            picks[rname] = pool[np.argsort(-score[pool])[:batch_size]]
        for rname, idx in picks.items():
            run_trial("filter", idx, step_scale=eta, rule=rname, pool=p)
            run_trial("filter", idx, target_len=ref_len, rule=rname, pool=p)

    out["trial/kind"] = np.array([t["kind"] for t in trials])
    out["trial/rule"] = np.array([t["rule"] for t in trials])
    out["trial/pool"] = np.array([t["pool"] for t in trials])
    out["trial/norm"] = np.array([t["norm"] for t in trials])
    out["trial/step_scale"] = np.array([t["step_scale"] for t in trials], np.float32)
    out["trial/grad_norm"] = np.array([t["grad_norm"] for t in trials], np.float32)
    out["trial/idx"] = np.stack([t["idx"] for t in trials])
    for name in names:
        out[f"trial/pred/{name}"] = np.array([t["pred"][name] for t in trials], np.float32)
    for k in base:
        out[f"trial/d/{k}"] = np.array([t["d"][k] for t in trials], np.float32)

    # 4. noise floor: the same 12 random steps measured under 5 different PGD start seeds
    seeds_d = []
    for s in range(5):
        ev_s = ProbeEvaluator(vx, vy, [e / 255 for e in eval_eps], steps=pgd_steps, seed=7000 + s)
        base_s = ev_s.evaluate(model)
        seeds_d.append([run_trial("noise", idx, step_scale=eta, evaluator=ev_s, base_=base_s, record=False)
                        for idx in random_batches[:12]])
    for k in base:
        out[f"noise/d/{k}"] = np.array([[r[k] for r in row] for row in seeds_d], np.float32)

    out["meta"] = np.array(json.dumps(dict(
        step=int(step), epoch=step / steps_per_epoch, lr=float(lr), eta=float(eta), eta_mult=eta_mult,
        ref_step_len=ref_len, run_dir=str(run_dir), score_eps=list(score_eps), eval_eps=list(eval_eps),
        pgd_steps=pgd_steps, batch_size=batch_size, n_trials=len(trials),
        seconds_scores=t_scores, seconds_total=time.time() - t0, env=environment_info())))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"step {step}: wrote {out_path} in {time.time() - t0:.0f}s (scores {t_scores:.0f}s, "
          f"{len(trials)} real steps)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--step", type=int, nargs="+", required=True, help="checkpoint step(s); -1 = final")
    p.add_argument("--out-dir", type=Path, default=None, help="default: RUN_DIR/investigate")
    p.add_argument("--n-random", type=int, default=100)
    p.add_argument("--n-pools", type=int, default=30)
    p.add_argument("--eta-mult", type=float, default=0.25, help="base step size as a multiple of the learning rate")
    a = p.parse_args()
    for st in a.step:
        tag = "final" if st < 0 else f"{st:06d}"
        investigate_checkpoint(a.run_dir, st, (a.out_dir or a.run_dir / "investigate") / f"step_{tag}.npz",
                               n_random=a.n_random, n_pools=a.n_pools, eta_mult=a.eta_mult)
