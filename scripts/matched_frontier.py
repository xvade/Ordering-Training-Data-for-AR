"""Compare families of runs at matched clean accuracy, the fair way.

A training recipe does not give one model but a set of *operating points*: the
weight-averaged model at every epoch of every variant in the family (and,
optionally, weight interpolations between an early and a late averaged model).
Robustness always trades against clean accuracy along such a set, so two
families are compared by their frontiers:

    frontier(c) = the best robustness among operating points whose clean accuracy is at least c.

An operating point is a (variant, epoch) pair averaged over seeds, so the
selection is made on seed means of ``val`` metrics and is the same for every
seed; the spread over seeds at the selected point is reported with it. The
selected points are written to a JSON file so that exactly those checkpoints
can then be evaluated once on the test split (``scripts/final_eval.py --ckpt``).

Usage::

    python scripts/matched_frontier.py --out OUT_PREFIX --metric adv_acc@2 --levels 0.6 0.65 0.7 0.75 \
        --family "uniform=DIR1,DIR2,..." [--interp "uniform=interp.csv"] --family "grad-match=DIR,..."

Each DIR is a variant directory with ``seed*/evals.json``. Light numpy/pandas work.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def points_of_variant(d: Path, metric: str, min_epoch: float):
    """Seed-averaged averaged-weight operating points of one variant: list of dicts."""
    runs = {int(f.parent.name[4:]): json.loads(f.read_text()) for f in sorted(d.glob("seed*/evals.json"))}
    if not runs:
        return []
    n = min(len(r) for r in runs.values())
    pre = "ema_" if all("ema_clean_acc" in e for r in runs.values() for e in r[:n]) else ""
    out = []
    for i in range(n):
        evs = [r[i] for r in runs.values()]
        if evs[0]["epoch"] < min_epoch:
            continue
        clean = np.array([e[pre + "clean_acc"] for e in evs])
        rob = np.array([e[pre + metric] for e in evs])
        out.append(dict(variant=str(d), kind="epoch", epoch=float(evs[0]["epoch"]), step=int(evs[0]["step"]),
                        seeds=len(evs), clean=clean.mean(), clean_sd=clean.std(), rob=rob.mean(), rob_sd=rob.std(),
                        weights="ema" if pre else "raw"))
    return out


def points_of_interp(csv: Path, metric: str):
    """Seed-averaged operating points from ``scripts/interp_eval.py`` output."""
    df = pd.read_csv(csv)
    out = []
    for (label, early, alpha), g in df.groupby(["label", "early", "alpha"]):
        if alpha in (0.0, 1.0):
            continue  # the end points are ordinary epoch points
        out.append(dict(variant=f"interp[{label}]", kind="interp", epoch=float(early), alpha=float(alpha), seeds=len(g),
                        clean=g["clean_acc"].mean(), clean_sd=g["clean_acc"].std(ddof=0),
                        rob=g[metric].mean(), rob_sd=g[metric].std(ddof=0), weights="ema"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="prefix for OUT.csv and OUT.json")
    p.add_argument("--metric", default="adv_acc@2")
    p.add_argument("--levels", type=float, nargs="+", default=[0.55, 0.60, 0.65, 0.70, 0.75])
    p.add_argument("--family", action="append", required=True, help="NAME=DIR,DIR,...")
    p.add_argument("--interp", action="append", default=[], help="NAME=CSV from scripts/interp_eval.py")
    p.add_argument("--min-epoch", type=float, default=2.0)
    p.add_argument("--min-seeds", type=int, default=2)
    a = p.parse_args()

    families = {}
    for spec in a.family:
        name, dirs = spec.split("=", 1)
        for d in dirs.split(","):
            families.setdefault(name, []).extend(points_of_variant(Path(d), a.metric, a.min_epoch))
    for spec in a.interp:
        name, csv = spec.split("=", 1)
        families.setdefault(name, []).extend(points_of_interp(Path(csv), a.metric))

    rows, selected = [], {}
    for name, pts in families.items():
        pts = [q for q in pts if q["seeds"] >= a.min_seeds]
        for level in a.levels:
            ok = [q for q in pts if q["clean"] >= level]
            if not ok:
                rows.append(dict(family=name, level=level))
                continue
            best = max(ok, key=lambda q: q["rob"])
            rows.append(dict(family=name, level=level, **{k: best[k] for k in ("rob", "rob_sd", "clean", "clean_sd", "seeds", "kind", "epoch")},
                             alpha=best.get("alpha", np.nan), variant=Path(best["variant"]).name))
            selected[f"{name}@{level:g}"] = best
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out.with_suffix(".csv"), index=False)
    a.out.with_suffix(".json").write_text(json.dumps(selected, indent=1, default=float))

    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.3f}")
    print(f"frontier of {a.metric}: best seed-mean value among operating points with seed-mean clean accuracy >= level\n")
    print(df.pivot(index="level", columns="family", values="rob").to_string())
    print("\nselected operating points:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
