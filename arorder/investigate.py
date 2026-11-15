"""Counterfactual study of single updates at a saved checkpoint.

Question: starting from fixed weights ``theta``, which training samples, if
trained on next, raise or lower adversarial robustness on held-out data?

Two complementary measurements are made.

1. **First-order per-sample scores.** A pure gradient step on sample ``i`` with
   step size ``eta`` changes any differentiable held-out quantity ``Q`` by
   ``-eta * <grad Q(theta), grad loss_i(theta)>`` to first order. We take ``Q``
   to be the held-out clean loss and the held-out adversarial loss at several
   radii (its gradient is the loss gradient at the PGD points, by Danskin), and
   compute the inner products for *every* training sample with per-sample
   gradients (``torch.func``). The sign convention used throughout:

       ``gain[Q][i] = <grad Q, grad loss_i>``  > 0  means training on ``i``
       *lowers* ``Q`` (helps), < 0 means it raises ``Q`` (hurts).

2. **Real finite steps.** First-order scores are only trustworthy if real steps
   agree with them. So we actually apply ``theta - eta * mean_grad(batch)`` for
   random batches and for batches built from score quantiles, re-run the
   attack at the new weights (same fixed starts, so the comparison is paired)
   on the ``val`` split, which played no part in computing the scores, and
   record the true change of every metric.

The step is a *pure gradient step* (no momentum history, no weight decay): it
isolates the contribution of the batch itself, which is what a data-selection
rule controls. ``eta`` defaults to the learning rate at the checkpoint's step,
i.e. the displacement one gradient causes in the step it is applied. (The total
displacement momentum eventually assigns to a gradient, ``lr / (1 - momentum)``,
is 10x larger; applied in one shot it is far outside the linear regime and
simply wrecks the model - measured on 2026-11-15, see JOURNAL.md - so it says
nothing about what the batch does in real training.) Steps at 0.25x and 4x
``eta`` are included to check linearity.

Run as a module::

    python -m arorder.investigate --run-dir RUN --step 3510 --out OUT.npz
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

from .data import load_cifar10
from .models import build_model, flat_params, set_flat_params
from .robustness import ProbeEvaluator
from .train import TrainConfig, environment_info, lr_at


def per_sample_gradient_products(model, x, y, vectors: torch.Tensor, chunk: int = 250):
    """Inner products of every sample's loss gradient with given directions.

    Args:
        vectors: ``(K, P)`` directions in flat-parameter space.

    Returns:
        ``dots`` ``(N, K)`` with ``dots[i, k] = <grad loss_i, vectors[k]>`` and
        ``norms`` ``(N,)``, the per-sample gradient norms.

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
    dots, norms = [], []
    for s in range(0, x.shape[0], chunk):
        g = per_sample(params, x[s : s + chunk], y[s : s + chunk])
        flat = torch.cat([g[n].flatten(1) for n in names], dim=1)
        dots.append(flat @ vectors.T)
        norms.append(flat.norm(dim=1))
    return torch.cat(dots), torch.cat(norms)


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


def investigate_checkpoint(
    run_dir: Path,
    step: int,
    out_path: Path,
    score_eps=(1.0, 2.0, 4.0),
    eval_eps=(1.0, 2.0),
    pgd_steps: int = 10,
    n_random: int = 200,
    n_bins: int = 20,
    batches_per_bin: int = 4,
    batch_size: int = 128,
    seed: int = 0,
    eta_mult: float = 1.0,
) -> None:
    """Run the full counterfactual study for one checkpoint and save one ``.npz``.

    Saved arrays (N = number of train samples):

    * ``feat/<name>`` (N,): intrinsic sample features, see :func:`sample_features`;
      also ``feat/grad_norm`` and ``feat/label``.
    * ``gain/<Q>`` (N,): first-order gains ``<grad Q, grad loss_i>`` for ``Q`` in
      ``probe_clean``, ``probe_adv@e`` (cross-entropy), ``probe_advm@e``
      (clipped negative margin) and the same on ``val`` for clean and the
      middle radius. Positive = training on the sample lowers ``Q``.
    * ``vec_names`` and ``vec_gram``: names and Gram matrix of the held-out
      gradient directions (their norms and mutual alignment).
    * ``trial/*``: one row per real finite step. ``trial/kind`` is ``random``,
      ``bin:<score>`` (batch drawn from one quantile bin of a score) or
      ``scale`` (a random batch at a different ``eta``); ``trial/bin``,
      ``trial/eta``, ``trial/idx`` (the batch), ``trial/pred/<Q>`` (first-order prediction of the change of
      ``Q``), and ``trial/d/<metric>`` (measured change of each ``val`` metric).
    * ``base/<metric>``: ``val`` metrics at the unperturbed weights.
    * ``noise/*``: spread of measured changes across PGD start seeds.
    * ``meta``: JSON string with step, eta, lr, host, timing.
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
    dots, norms = per_sample_gradient_products(model, tx, ty, V)
    for k, name in enumerate(names):
        out[f"gain/{name}"] = dots[:, k].cpu().numpy()
    out["feat/grad_norm"] = norms.cpu().numpy()
    out["feat/label"] = ty.cpu().numpy()
    for k, v in sample_features(model, tx, ty, score_eps_px, pgd_steps).items():
        out[f"feat/{k}"] = v.cpu().numpy()
    t_scores = time.time() - t0

    # 3. real finite steps, measured on val
    theta0 = flat_params(model)
    base = val_ev.evaluate(model)
    for k, v in base.items():
        out[f"base/{k}"] = np.float32(v)

    def measure(idx: np.ndarray, eta_: float, evaluator=val_ev) -> tuple[dict, dict]:
        """Apply one pure gradient step on ``idx``; return (predicted, measured) changes."""
        it = torch.as_tensor(idx, device=dev)
        g = batch_gradient(model, tx[it], ty[it])
        pred = {name: (-eta_ * (V[k] @ g)).item() for k, name in enumerate(names)}
        set_flat_params(model, theta0 - eta_ * g)
        after = evaluator.evaluate(model)
        set_flat_params(model, theta0)
        return pred, after

    trials = []

    def run_trial(kind, bin_, idx, eta_):
        pred, after = measure(idx, eta_)
        trials.append(dict(kind=kind, bin=bin_, eta=eta_, pred=pred, idx=np.asarray(idx, np.int32),
                           d={k: after[k] - base[k] for k in base}))

    n = len(data.train)
    random_batches = [rng.choice(n, batch_size, replace=False) for _ in range(n_random)]
    for idx in random_batches:
        run_trial("random", -1, idx, eta)
    for scale in (0.25, 4.0):  # linearity check on the first 40 random batches
        for idx in random_batches[:40]:
            run_trial("scale", -1, idx, eta * scale)
    gains = {name: dots[:, k].cpu().numpy() for k, name in enumerate(names)}
    scores = {
        "adv": gains[f"probe_adv@{mid:g}"],                      # effect on held-out adversarial loss
        "clean": gains["probe_clean"],                           # effect on held-out clean loss
        "gap": gains[f"probe_adv@{mid:g}"] - gains["probe_clean"],  # effect on (adv - clean) loss
    }
    for sname, s in scores.items():
        order = np.argsort(s)
        for b, members in enumerate(np.array_split(order, n_bins)):
            for _ in range(batches_per_bin):
                run_trial(f"bin:{sname}", b, rng.choice(members, batch_size, replace=False), eta)

    out["trial/kind"] = np.array([t["kind"] for t in trials])
    out["trial/bin"] = np.array([t["bin"] for t in trials])
    out["trial/eta"] = np.array([t["eta"] for t in trials], np.float32)
    out["trial/idx"] = np.stack([t["idx"] for t in trials])  # the batches, as positions in the train split
    for name in names:
        out[f"trial/pred/{name}"] = np.array([t["pred"][name] for t in trials], np.float32)
    for k in base:
        out[f"trial/d/{k}"] = np.array([t["d"][k] for t in trials], np.float32)

    # 4. noise floor: the same 20 random steps measured under 5 different PGD start seeds
    seeds_d = []
    for s in range(5):
        ev_s = ProbeEvaluator(vx, vy, [e / 255 for e in eval_eps], steps=pgd_steps, seed=7000 + s)
        base_s = ev_s.evaluate(model)
        row = []
        for idx in random_batches[:20]:
            _, after = measure(idx, eta, evaluator=ev_s)
            row.append({k: after[k] - base_s[k] for k in base_s})
        seeds_d.append(row)
    for k in base:
        arr = np.array([[r[k] for r in row] for row in seeds_d], np.float32)  # (seeds, batches)
        out[f"noise/d/{k}"] = arr

    out["meta"] = np.array(json.dumps(dict(
        step=int(step), epoch=step / steps_per_epoch, lr=float(lr), eta=float(eta), run_dir=str(run_dir),
        score_eps=list(score_eps), eval_eps=list(eval_eps), pgd_steps=pgd_steps, batch_size=batch_size,
        seconds_scores=t_scores, seconds_total=time.time() - t0, env=environment_info())))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"step {step}: wrote {out_path} in {time.time() - t0:.0f}s (scores {t_scores:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--step", type=int, nargs="+", required=True, help="checkpoint step(s); -1 = final")
    p.add_argument("--out-dir", type=Path, default=None, help="default: RUN_DIR/investigate")
    p.add_argument("--n-random", type=int, default=200)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--batches-per-bin", type=int, default=4)
    p.add_argument("--eta-mult", type=float, default=1.0, help="step size as a multiple of the learning rate")
    a = p.parse_args()
    for st in a.step:
        tag = "final" if st < 0 else f"{st:06d}"
        investigate_checkpoint(a.run_dir, st, (a.out_dir or a.run_dir / "investigate") / f"step_{tag}.npz",
                               n_random=a.n_random, n_bins=a.n_bins, batches_per_bin=a.batches_per_bin,
                               eta_mult=a.eta_mult)
