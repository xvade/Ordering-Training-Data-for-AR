"""Weight-space reference: interpolate an early averaged model with a late one and evaluate.

    w(alpha) = (1 - alpha) * average(epoch a) + alpha * average(epoch b)

The planning agent found (one run) that blending the averaged weights from the
robustness peak with the final weights lands above the run's own
accuracy-robustness path. A sampler has to be compared with what uniform
sampling achieves *with* such weight-space operations, so this script produces
that reference for any run trained with ``ema_decay > 0`` and ``ckpt_every > 0``.

Usage::

    python scripts/interp_eval.py --out OUT.csv [--split val] [--early 3 4 5 6] [--late 20] \
        [--alphas 0.25 0.5 0.75] LABEL=VARIANT_DIR [...]

Evaluates with the same PGD-10 protocol as the trainer's periodic evaluation
(fixed starts), on ``val`` by default. One row per (run, early epoch, alpha).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from arorder.data import load_dataset
from arorder.models import build_model
from arorder.robustness import ProbeEvaluator
from arorder.train import TrainConfig, load_checkpoint

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("runs", nargs="+")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--split", default="val", choices=["val", "test"])
p.add_argument("--early", type=int, nargs="+", default=[3, 4, 5, 6])
p.add_argument("--late", type=int, default=0, help="late epoch (0 = final checkpoint)")
p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
a = p.parse_args()

dev = torch.device("cuda")
rows, bundles = [], {}
for spec in a.runs:
    label, d = spec.split("=", 1)
    for run in sorted(Path(d).glob("seed*")):
        cfg = TrainConfig(**json.loads((run / "config.json").read_text()))
        if cfg.dataset not in bundles:
            bundles[cfg.dataset] = load_dataset(cfg.dataset, dev)
        data = bundles[cfg.dataset]
        ds = getattr(data, a.split)
        x, y = ds.batch(torch.arange(len(ds), device=dev))
        ev = ProbeEvaluator(x, y, [e / 255 for e in cfg.eval_eps], steps=cfg.eval_steps, seed=4321)
        spe = len(data.train) // cfg.batch_size
        model = build_model(cfg.model, data.num_classes, width=cfg.width, norm=cfg.norm).to(dev).eval()
        late_name = f"ckpt_{a.late * spe:06d}.pt" if a.late else "ckpt_final.pt"
        late = load_checkpoint(run / late_name, map_location=dev)["ema"]
        for ep in a.early:
            early = load_checkpoint(run / f"ckpt_{ep * spe:06d}.pt", map_location=dev)["ema"]
            for alpha in [0.0] + a.alphas + [1.0]:
                model.load_state_dict({k: (1 - alpha) * early[k] + alpha * late[k] if early[k].is_floating_point() else late[k]
                                       for k in late})
                m = ev.evaluate(model)
                rows.append(dict(label=label, seed=int(run.name[4:]), early=ep, late=a.late or cfg.epochs, alpha=alpha,
                                 **{k: v for k, v in m.items() if not k.startswith(("adv_loss", "adv_margin", "clean_margin"))}))
        print(f"done {run}", flush=True)
df = pd.DataFrame(rows)
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
pd.set_option("display.width", 220, "display.max_rows", 200, "display.float_format", lambda v: f"{v:.3f}")
cols = [c for c in ("clean_acc", "adv_acc@2", "adv_acc@4", "auc") if c in df]
print(df.groupby(["label", "early", "alpha"])[cols].agg(["mean", "std"]).to_string())
