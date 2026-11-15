"""Ad-hoc test (2026-11-15): can a non-negative combination of *clean* per-sample
gradients reproduce the adversarial-training gradient?

If it can, a sampler could emulate adversarial training by choosing (weighting)
clean samples; if it cannot, "selection towards the adversarial gradient" has no
target it can reach, whatever the selection rule.

For a checkpoint, with weights theta (raw or averaged):

* target ``t``: gradient of the adversarial loss (PGD at 2/255 and 8/255) on the
  held-out probe points, and on the candidate pool itself;
* candidates: per-sample clean gradients ``g_i`` of a random pool of training
  samples (several pool sizes);
* fit: ``min_{w >= 0} || sum_i w_i g_i - t ||^2`` by projected gradient on the
  Gram matrix, with the pool split in two halves: weights are fitted against a
  target measured on one half of the probe set and scored against the target
  from the other half, so fitting probe noise is not rewarded.

Reported: cosine between the uniform pool mean and the target (what plain
training gets), cosine of the fitted combination with the fit target and with
the held-out target, and the effective number of samples carrying the weight.

Usage: python scripts/adhoc/gradient_matching.py RUN_DIR STEP [raw|ema]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from arorder.attacks import pgd_linf
from arorder.data import load_cifar10
from arorder.models import build_model
from arorder.train import TrainConfig, load_checkpoint


def flat_grad(model, x, y):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(F.cross_entropy(model(x), y), params)
    return torch.cat([g.reshape(-1) for g in grads])


def per_sample_grads(model, x, y, chunk=128):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    f = vmap(grad(lambda p, xi, yi: F.cross_entropy(functional_call(model, (p, buffers), (xi[None],)), yi[None])),
             in_dims=(None, 0, 0))
    out = []
    for s in range(0, len(x), chunk):
        g = f(params, x[s : s + chunk], y[s : s + chunk])
        out.append(torch.cat([g[n].flatten(1) for n in names], 1))
    return torch.cat(out)


def nnls(K, b, iters=3000):
    """min_{w>=0} w^T K w - 2 b^T w  by projected gradient (K = G G^T, b = G t)."""
    w = torch.full_like(b, 1.0 / len(b))
    step = 1.0 / torch.linalg.eigvalsh(K)[-1].item()
    for _ in range(iters):
        w = (w - step * (K @ w - b)).clamp_min(0)
    return w


def cos(a, b):
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def main(run_dir, step, weights):
    dev = torch.device("cuda")
    cfg = TrainConfig(**json.loads((Path(run_dir) / "config.json").read_text()))
    data = load_cifar10(dev)
    model = build_model(cfg.model, data.num_classes, width=cfg.width, norm=cfg.norm).to(dev)
    model.load_state_dict(load_checkpoint(Path(run_dir) / f"ckpt_{step:06d}.pt", map_location=dev)["ema" if weights == "ema" else "model"])
    model.eval()
    px, py = data.probe.batch(torch.arange(len(data.probe), device=dev))
    half = len(px) // 2
    gen = torch.Generator(device="cpu").manual_seed(0)
    print(f"{run_dir} step {step} weights {weights}")
    for eps in (2.0, 8.0):
        xa = torch.cat([pgd_linf(model, px[s : s + 500], py[s : s + 500], eps / 255, steps=10) for s in range(0, len(px), 500)])
        t_fit, t_test = flat_grad(model, xa[:half], py[:half]), flat_grad(model, xa[half:], py[half:])
        g_clean = flat_grad(model, px, py)
        print(f" eps {eps:g}: cos(clean probe grad, adv probe grad) = {cos(g_clean, t_fit + t_test):.3f}; "
              f"cos(adv grad half A, half B) = {cos(t_fit, t_test):.3f}  (ceiling for any fit)")
        for pool_size in (128, 512, 2048):
            rows = []
            for rep in range(3):
                idx = torch.randperm(len(data.train), generator=gen)[:pool_size].to(dev)
                x, y = data.train.batch(idx)
                G = per_sample_grads(model, x, y)
                K, mean = G @ G.T, G.mean(0)
                w = nnls(K, G @ t_fit)
                fit = w @ G
                # own-pool target: the adversarial gradient of the same pool samples
                xa_pool = pgd_linf(model, x, y, eps / 255, steps=10)
                t_pool = flat_grad(model, xa_pool, y)
                w2 = nnls(K, G @ t_pool)
                rows.append((cos(mean, t_test), cos(fit, t_fit), cos(fit, t_test), (w.sum() ** 2 / (w ** 2).sum()).item(),
                             (w > 1e-8).float().mean().item(), cos(mean, t_pool), cos(w2 @ G, t_pool), cos(w2 @ G, t_test)))
                del G, K
            r = np.mean(rows, 0)
            print(f"   pool {pool_size:5d}: uniform mean vs held-out target {r[0]:.3f} | fitted: vs fit target {r[1]:.3f}, "
                  f"vs held-out target {r[2]:.3f} (effective n {r[3]:.0f}, {100 * r[4]:.0f}% nonzero) | "
                  f"own-pool target: uniform {r[5]:.3f}, fitted {r[6]:.3f}, fitted vs held-out probe target {r[7]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "ema")
