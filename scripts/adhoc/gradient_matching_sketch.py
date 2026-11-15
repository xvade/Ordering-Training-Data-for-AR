"""Ad-hoc test (2026-11-15): does gradient matching still work when the fit only
sees a small part of each gradient?

Materialising per-sample gradients costs (pool size) x (number of parameters)
floats, which does not scale to large models. Here the non-negative
least-squares weights are fitted using only (a) a random subset of m gradient
coordinates or (b) the final linear layer, and the resulting combination is
scored in the *full* gradient space against a held-out target. If the cosine
is close to that of the full fit, a large model can be handled by keeping only
m coordinates of each per-sample gradient.

Usage: python scripts/adhoc/gradient_matching_sketch.py RUN_DIR STEP [raw|ema]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from arorder.attacks import pgd_linf
from arorder.data import load_cifar10
from arorder.models import build_model
from arorder.samplers import nonneg_least_squares, per_sample_gradient_matrix
from arorder.train import TrainConfig, load_checkpoint


def flat_grad(model, x, y):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(torch.nn.functional.cross_entropy(model(x), y), params)
    return torch.cat([g.reshape(-1) for g in grads])


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
    n_params = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    print(f"{run_dir} step {step} weights {weights}; {n_params} parameters, head = last {head}")
    xa = torch.cat([pgd_linf(model, px[s : s + 500], py[s : s + 500], 8 / 255, steps=10) for s in range(0, len(px), 500)])
    t_fit, t_test = flat_grad(model, xa[:half], py[:half]), flat_grad(model, xa[half:], py[half:])
    rows = {}
    for rep in range(3):
        idx = torch.randperm(len(data.train), generator=gen)[:512].to(dev)
        G = per_sample_gradient_matrix(model, *data.train.batch(idx))
        subsets = {"all": torch.arange(n_params, device=dev), "head only": torch.arange(n_params - head, n_params, device=dev)}
        for m in (512, 2048, 8192, 32768):
            subsets[f"random {m}"] = torch.randperm(n_params, generator=gen)[:m].to(dev)
        rows.setdefault("uniform batch of 128", []).append(cos(G[:128].mean(0), t_test))
        for name, coords in subsets.items():
            Gs, ts = G[:, coords], t_fit[coords]
            w = nonneg_least_squares(Gs @ Gs.T, Gs @ ts)
            rows.setdefault(f"fit on {name}", []).append(cos(w @ G, t_test))
    for name, v in rows.items():
        print(f"   {name:24s} cos(full-space combination, held-out target) = {np.mean(v):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "ema")
