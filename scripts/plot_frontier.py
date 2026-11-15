"""Accuracy-robustness plot for a set of runs, against the uniform-sampling reference curves.

Usage::

    python scripts/plot_frontier.py OUT.png [--weights ema|raw] [--metrics adv_acc@2 adv_acc@8 auc] \
        --ref "uniform 5/10/20/40 ep=ROOT/base_e05,ROOT/base_e10,ROOT/base_e20,ROOT/base_e40" \
        --path "uniform (mid-run)=ROOT/base_e20" \
        --run "grad-match 2/255=ROOT2/gm_e2" --run "label=DIR" ...

* ``--ref LABEL=DIR,DIR,...``: each DIR is a variant directory holding ``seed*/evals.json``;
  the *final* evaluation of each (mean over seeds) is plotted as a point and the points are joined:
  the curve of annealed end-of-training models.
* ``--path LABEL=DIR``: every periodic evaluation of the variant (mean over seeds), joined in
  training order: the curve traced by stopping that run at any epoch.
* ``--run LABEL=DIR``: like ``--path`` but drawn as a method (coloured line, large final marker
  with error bars = standard deviation over seeds).

With ``--weights ema`` (default) the weight-averaged model is plotted when a run has one.
A table of the final points is printed. Light, single-threaded work.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_variant(d: Path, weights: str):
    """Return (steps, {metric: array[seeds, evals]}) for a variant directory (finished and unfinished runs)."""
    runs = []
    for f in sorted(Path(d).glob("seed*/evals.json")):
        runs.append(json.loads(f.read_text()))
    if not runs:
        raise SystemExit(f"no runs under {d}")
    n = min(len(r) for r in runs)
    pre = "ema_" if weights == "ema" and all("ema_clean_acc" in e for r in runs for e in r[:n]) else ""
    keys = [k for k in runs[0][0] if not isinstance(runs[0][0][k], list) and not k.startswith("ema_")]
    out = {k: np.array([[e.get(pre + k, e[k]) if k not in ("step", "epoch", "elapsed") else e[k] for e in r[:n]] for r in runs])
           for k in keys}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--metrics", nargs="+", default=["adv_acc@2", "adv_acc@8", "auc"])
    p.add_argument("--ref", action="append", default=[])
    p.add_argument("--path", action="append", default=[])
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--min-epoch", type=float, default=1.0, help="drop evaluations before this epoch from paths")
    a = p.parse_args()

    fig, axes = plt.subplots(1, len(a.metrics), figsize=(6.2 * len(a.metrics), 5.6), squeeze=False)
    rows = []
    for spec in a.ref:
        label, dirs = spec.split("=", 1)
        pts = []
        for d in dirs.split(","):
            v = load_variant(d, a.weights)
            pts.append({m: v[m][:, -1].mean() for m in ["clean_acc"] + a.metrics})
            rows.append((f"{label} [{Path(d).name}]", v, len(v["clean_acc"])))
        for ax, m in zip(axes[0], a.metrics):
            ax.plot([q["clean_acc"] for q in pts], [q[m] for q in pts], "k--o", lw=1.2, ms=6, label=label)
    for spec in a.path:
        label, d = spec.split("=", 1)
        v = load_variant(d, a.weights)
        keep = v["epoch"][0] >= a.min_epoch
        for ax, m in zip(axes[0], a.metrics):
            ax.plot(v["clean_acc"].mean(0)[keep], v[m].mean(0)[keep], "-", color="0.45", lw=1.5, label=label)
    for i, spec in enumerate(a.run):
        label, d = spec.split("=", 1)
        v = load_variant(d, a.weights)
        keep = v["epoch"][0] >= a.min_epoch
        rows.append((label, v, len(v["clean_acc"])))
        for ax, m in zip(axes[0], a.metrics):
            c = f"C{i % 10}"
            ax.plot(v["clean_acc"].mean(0)[keep], v[m].mean(0)[keep], "-", color=c, lw=1.2, alpha=0.8)
            ax.errorbar(v["clean_acc"][:, -1].mean(), v[m][:, -1].mean(), xerr=v["clean_acc"][:, -1].std(),
                        yerr=v[m][:, -1].std(), fmt="os^Dv<>"[i // 10 % 7], color=c, ms=9, mec="k", capsize=3, label=label)
    for ax, m in zip(axes[0], a.metrics):
        ax.set(xlabel="clean accuracy (val)", ylabel=f"{m} (val)", title=m)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Accuracy vs robustness, {'weight-averaged' if a.weights == 'ema' else 'raw'} models; "
                 "lines: training path, markers: end of training (bars: sd over seeds)", fontsize=10)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)

    print(f"{'run':44s} seeds  " + "  ".join(f"{m:>15s}" for m in ["clean_acc"] + a.metrics))
    for label, v, n in rows:
        print(f"{label:44s} {n:5d}  " + "  ".join(f"{v[m][:, -1].mean():7.3f}±{v[m][:, -1].std():5.3f}  " for m in ["clean_acc"] + a.metrics))


if __name__ == "__main__":
    main()
