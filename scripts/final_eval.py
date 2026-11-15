"""Strong-attack evaluation of several runs, with a summary table.

Usage::

    python scripts/final_eval.py --split test [--autoattack] [--ckpt ckpt_003510.pt --weights ema] \
        --out results/final/test_eval.csv  LABEL=VARIANT_DIR [LABEL=VARIANT_DIR ...]

Each VARIANT_DIR holds ``seed*/`` run directories. Every run is evaluated with
:func:`arorder.evaluate.evaluate_checkpoint` (worst case over restarts and two
objectives of PGD-50, plus AutoAttack if asked); runs already evaluated with the
same settings are read from their JSON. Prints mean and standard deviation over
seeds and writes one row per run to the CSV.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from arorder.evaluate import evaluate_checkpoint

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("runs", nargs="+")
p.add_argument("--split", default="test", choices=["val", "test"])
p.add_argument("--eps", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
p.add_argument("--autoattack", action="store_true")
p.add_argument("--ckpt", default="ckpt_final.pt")
p.add_argument("--weights", default="raw", choices=["raw", "ema"])
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

rows = []
for spec in a.runs:
    label, d = spec.split("=", 1)
    for run in sorted(Path(d).glob("seed*")):
        if not (run / a.ckpt).exists():
            print(f"skip {run}: no {a.ckpt}")
            continue
        default = a.ckpt == "ckpt_final.pt" and a.weights == "raw"
        f = run / (f"eval_{a.split}.json" if default else f"eval_{a.split}_{Path(a.ckpt).stem}_{a.weights}.json")
        res = json.loads(f.read_text()) if f.exists() else None
        if res is None or (a.autoattack and not any(k.startswith("autoattack") for k in res)) \
                or any(f"robust_acc@{e:g}" not in res for e in a.eps):
            res = evaluate_checkpoint(run, a.split, a.eps, autoattack=a.autoattack, ckpt=a.ckpt, weights=a.weights)
        rows.append(dict(label=label, run=str(run), **{k: v for k, v in res.items() if isinstance(v, (int, float))}))
        print(f"{label} {run.name}: " + " ".join(f"{k}={v:.3f}" for k, v in rows[-1].items() if isinstance(v, float)), flush=True)
df = pd.DataFrame(rows)
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
num = [c for c in df.columns if c not in ("label", "run", "n")]
summary = df.groupby("label", sort=False)[num].agg(lambda v: f"{v.mean():.3f}±{v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}")
summary.insert(0, "seeds", df.groupby("label", sort=False).size())
pd.set_option("display.width", 250, "display.max_columns", 40)
print(summary.to_string())
