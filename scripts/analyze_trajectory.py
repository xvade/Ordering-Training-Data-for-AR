"""Summarise the per-step probe log of an instrumented training run.

Usage: python scripts/analyze_trajectory.py RUN_DIR OUT_DIR

Row ``t`` of the probe log describes the weights after ``t`` updates, so the
effect of update ``t`` is ``log[t + 1] - log[t]``. Writes into OUT_DIR:

* ``trajectory_epochs.csv`` - per-epoch means of every probe metric and statistics of
  the per-update changes (how often an update lowers clean loss but raises
  adversarial loss, how correlated the two changes are, and so on);
* ``trajectory.png``        - the curves;
* ``worst_steps.csv``       - the updates with the largest increase of the
  adversarial-minus-clean loss gap.

Light, single-threaded numpy/matplotlib work.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text())
    with np.load(run_dir / "steps.npz") as z:
        d = {k: z[k] for k in z.files}
    probe = {k[len("probe/"):]: v for k, v in d.items() if k.startswith("probe/")}
    steps = d["probe_step"]
    assert (np.diff(steps) == 1).all(), "analysis assumes a probe measurement before every step"
    n_updates = len(steps) - 1
    spe = 45000 // cfg["batch_size"]
    eps = f"{cfg['probe_eps'][-1]:g}"  # the largest per-step radius
    gap = probe[f"adv_loss@{eps}"] - probe["clean_loss"]
    probe["gap"] = gap
    delta = {k: np.diff(v) for k, v in probe.items()}

    rows = []
    for ep in range(n_updates // spe):
        s = slice(ep * spe, (ep + 1) * spe)
        dc, da, dg = delta["clean_loss"][s], delta[f"adv_loss@{eps}"][s], delta["gap"][s]
        rows.append(dict(
            epoch=ep + 1, lr=float(d["lr"][s].mean()), train_loss=float(d["train_loss"][s].mean()),
            train_acc=float(d["train_acc"][s].mean()),
            **{k: float(v[s].mean()) for k, v in probe.items()},
            # statistics of single-update changes within the epoch
            net_d_clean_loss=float(dc.sum()), net_d_adv_loss=float(da.sum()), net_d_gap=float(dg.sum()),
            std_d_clean_loss=float(dc.std()), std_d_adv_loss=float(da.std()), std_d_gap=float(dg.std()),
            corr_d_clean_d_adv=float(np.corrcoef(dc, da)[0, 1]),
            frac_clean_down_adv_up=float(((dc < 0) & (da > 0)).mean()),
            frac_gap_up=float((dg > 0).mean()),
            # how much of the epoch's gap increase comes from the worst 10% of updates
            gap_up_share_top10pct=float(np.sort(np.maximum(dg, 0))[-spe // 10:].sum() / max(np.maximum(dg, 0).sum(), 1e-30)),
            lag1_autocorr_d_gap=float(np.corrcoef(dg[:-1], dg[1:])[0, 1]),
        ))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "trajectory_epochs.csv", index=False)

    worst = np.argsort(-delta["gap"])[:50]
    pd.DataFrame(dict(step=worst, epoch=worst / spe, **{f"d_{k}": v[worst] for k, v in delta.items()})) \
        .to_csv(out_dir / "worst_steps.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    x = steps / spe
    smooth = lambda v, w=25: np.convolve(v, np.ones(w) / w, mode="same")
    ax = axes[0, 0]
    ax.plot(x, probe["clean_acc"], label="clean")
    for e in cfg["probe_eps"]:
        ax.plot(x, probe[f"adv_acc@{e:g}"], label=f"PGD {e:g}/255")
    ax.set(title="Probe accuracy", xlabel="epoch"); ax.legend()
    ax = axes[0, 1]
    for e in cfg["probe_eps"]:
        ax.plot(x, probe[f"robust_given_correct@{e:g}"], label=f"{e:g}/255")
    ax.set(title="Robust accuracy / clean accuracy", xlabel="epoch"); ax.legend()
    ax = axes[0, 2]
    ax.plot(x, probe["lin_radius"])
    ax.set(title="Linearised boundary distance (1/255)", xlabel="epoch")
    ax = axes[1, 0]
    ax.plot(x, probe["clean_loss"], label="clean loss")
    ax.plot(x, probe[f"adv_loss@{eps}"], label=f"adv loss @{eps}")
    ax.plot(x, gap, label="gap")
    ax.set(title="Probe losses", xlabel="epoch", yscale="log"); ax.legend()
    ax = axes[1, 1]
    ax.plot(x[:-1], smooth(delta["clean_loss"]), label="d clean loss")
    ax.plot(x[:-1], smooth(delta["gap"]), label="d gap")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title="Per-update change (25-step mean)", xlabel="epoch", ylim=(-0.01, 0.01)); ax.legend()
    ax = axes[1, 2]
    sc = ax.scatter(delta["clean_loss"], delta[f"adv_loss@{eps}"], c=x[:-1], s=2, cmap="viridis")
    lim = np.percentile(np.abs(delta["clean_loss"]), 99)
    ax.plot([-lim, lim], [-lim, lim], "k", lw=0.5)
    ax.set(title="Per-update: d clean loss vs d adv loss", xlabel="d clean loss", ylabel=f"d adv loss @{eps}",
           xlim=(-lim, lim), ylim=(-2 * lim, 2 * lim))
    fig.colorbar(sc, ax=ax, label="epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=110)

    pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", lambda v: f"{v:.4g}")
    print(table.T.to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
