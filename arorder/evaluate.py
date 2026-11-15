"""Final-quality robustness evaluation of a trained checkpoint.

The attack used here is deliberately stronger than (and different from) the
PGD used inside training-time measurements, so that a method cannot look good
merely by confusing the attack it was tuned against (gradient masking):

* several random restarts,
* two objectives (cross-entropy and direct margin minimisation),
* many steps,
* and the per-example *worst case* over all of them.

If the ``autoattack`` package is installed, ``--autoattack`` additionally runs
the standard AutoAttack ensemble, the community reference.

Run as a module::

    python -m arorder.evaluate --run-dir RUN --split test --eps 1 2 4 8

Writes ``RUN/eval_<split>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .attacks import margin, pgd_linf
from .data import load_cifar10
from .models import build_model
from .train import TrainConfig, environment_info


@torch.no_grad()
def _margins(model, x, y, bs):
    return torch.cat([margin(model(x[s : s + bs]), y[s : s + bs]) for s in range(0, len(x), bs)])


def worst_case_margins(model, x, y, eps: float, steps: int = 50, restarts: int = 3,
                       batch_size: int = 1000, seed: int = 0) -> torch.Tensor:
    """Per-example minimum margin over restarts x {cross-entropy, margin} PGD."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    worst = _margins(model, x, y, batch_size)
    for r in range(restarts):
        for loss in ("ce", "margin"):
            for s in range(0, len(x), batch_size):
                xb, yb = x[s : s + batch_size], y[s : s + batch_size]
                init = torch.rand(xb.shape, device=x.device, generator=g) * 2 - 1 if r > 0 else None
                xa = pgd_linf(model, xb, yb, eps, steps=steps, init=init, loss=loss)
                with torch.no_grad():
                    worst[s : s + batch_size] = torch.minimum(worst[s : s + batch_size], margin(model(xa), yb))
    return worst


def evaluate_checkpoint(run_dir: Path, split: str = "val", eps_list=(1.0, 2.0, 4.0, 8.0), steps: int = 50,
                        restarts: int = 3, autoattack: bool = False, ckpt: str = "ckpt_final.pt") -> dict:
    """Evaluate ``run_dir/ckpt`` on ``split`` (``val`` or ``test``); radii in 1/255."""
    dev = torch.device("cuda")
    cfg = TrainConfig(**json.loads((Path(run_dir) / "config.json").read_text()))
    data = load_cifar10(dev)
    ds = getattr(data, split)
    x, y = ds.batch(torch.arange(len(ds), device=dev))
    model = build_model(cfg.model, data.num_classes, width=cfg.width, norm=cfg.norm).to(dev)
    model.load_state_dict(torch.load(Path(run_dir) / ckpt, map_location=dev)["model"])
    model.eval()
    res = {"split": split, "n": len(ds), "clean_acc": (_margins(model, x, y, 1000) > 0).float().mean().item(),
           "attack": f"worst case over {restarts} restarts x (CE, margin) PGD-{steps}", "env": environment_info()}
    for e in eps_list:
        m = worst_case_margins(model, x, y, e / 255, steps=steps, restarts=restarts)
        res[f"robust_acc@{e:g}"] = (m > 0).float().mean().item()
    radii = [0.0] + list(eps_list)
    accs = [res["clean_acc"]] + [res[f"robust_acc@{e:g}"] for e in eps_list]
    res["auc"] = sum((radii[i + 1] - radii[i]) * (accs[i + 1] + accs[i]) / 2 for i in range(len(eps_list)))
    if autoattack:
        from autoattack import AutoAttack  # optional dependency
        for e in eps_list:
            adversary = AutoAttack(model, norm="Linf", eps=e / 255, version="standard", verbose=False)
            xa = adversary.run_standard_evaluation(x, y, bs=1000)
            res[f"autoattack_acc@{e:g}"] = (_margins(model, xa, y, 1000) > 0).float().mean().item()
    (Path(run_dir) / f"eval_{split}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test", "probe"])
    p.add_argument("--eps", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--autoattack", action="store_true")
    a = p.parse_args()
    out = evaluate_checkpoint(a.run_dir, a.split, a.eps, a.steps, a.restarts, a.autoattack)
    print(json.dumps({k: v for k, v in out.items() if k != "env"}, indent=1))
