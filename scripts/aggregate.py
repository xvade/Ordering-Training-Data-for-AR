"""Aggregate a sweep: mean and spread over seeds of final (and per-epoch) val metrics.

Usage: python scripts/aggregate.py ROOT [OUT_DIR]

Reads ``ROOT/<variant>/seed*/evals.json`` and writes ``summary.csv`` (final
metrics per variant, mean and std over seeds), ``curves.csv`` (every periodic
evaluation of every run) and ``pareto.png`` (clean accuracy against robustness
for every evaluation point, which shows whether a variant is better than simply
stopping the baseline earlier). Light, single-threaded work.
"""

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(root.glob("*/seed*/evals.json")):
        if not (f.parent / "meta.json").exists():
            continue  # unfinished run
        for ev in json.loads(f.read_text()):
            rows.append(dict(variant=f.parent.parent.name, seed=int(f.parent.name[4:]), **ev))
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "curves.csv", index=False)
    final = curves.sort_values("step").groupby(["variant", "seed"]).tail(1)
    metrics = [c for c in final.columns if c not in ("variant", "seed", "step", "epoch", "elapsed")
               and not c.startswith(("adv_loss", "adv_margin", "clean_margin"))]
    agg = final.groupby("variant")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "seeds", final.groupby("variant").size())
    agg.to_csv(out / "summary.csv")

    pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", lambda v: f"{v:.3f}")
    show = [c for c in ("clean_acc", "adv_acc@2", "adv_acc@4", "auc", "lin_radius",
                        "ema_clean_acc", "ema_adv_acc@2", "ema_adv_acc@4", "ema_auc", "ema_lin_radius") if c in final]
    text = final.groupby("variant")[show].agg(lambda v: f"{v.mean():.3f}±{v.std():.3f}")
    text.insert(0, "seeds", final.groupby("variant").size())
    print(text.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pre = "ema_" if "ema_auc" in curves and curves["ema_auc"].notna().all() else ""  # prefer averaged weights
    for ax, ycol in zip(axes, (pre + "auc", pre + "adv_acc@2")):
        if ycol not in curves:
            continue
        xcol = pre + "clean_acc"
        for i, (variant, g) in enumerate(curves.groupby("variant")):
            mean = g.groupby("step")[[xcol, ycol]].mean()
            ax.plot(mean[xcol], mean[ycol], "-", color=f"C{i % 10}", alpha=0.6, lw=1)
            ax.scatter(mean[xcol].iloc[-1], mean[ycol].iloc[-1], color=f"C{i % 10}", s=60, label=variant,
                       marker="osD^v<>"[i // 10 % 7], edgecolor="k")
        ax.set(xlabel=f"val {xcol}", ylabel=f"val {ycol}", title=f"{ycol} vs clean accuracy (line: training path, marker: final)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=110)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    main(root, Path(sys.argv[2]) if len(sys.argv) > 2 else root / "_summary")
