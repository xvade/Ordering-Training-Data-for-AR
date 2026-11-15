"""Describe what a selecting sampler actually trained on, from a run's step log.

Usage: python scripts/analyze_selection.py RUN_DIR [RUN_DIR ...]

For each run prints, per epoch: the sampler's own statistics (``sel/*`` columns
of ``steps.npz``), the number of distinct training samples used and the share
of updates going to the most-used 10% of them, the class shares of the trained
samples, and the val metrics at the end of the epoch. Light numpy work.
"""

import json
import sys
from pathlib import Path

import numpy as np

from arorder.data import DEFAULT_DATA_DIR, make_splits

np.set_printoptions(linewidth=220, precision=3, suppress=True)
_labels = {}


def train_labels(dataset: str) -> np.ndarray:
    """Labels of the train split of ``dataset``, by position."""
    if dataset not in _labels:
        y = np.load(DEFAULT_DATA_DIR / f"{dataset}.npz")["y_train"]
        _labels[dataset] = y[make_splits(len(y)).train]
    return _labels[dataset]


for run in sys.argv[1:]:
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    labels = train_labels(cfg.get("dataset", "cifar10"))
    n_classes = int(labels.max()) + 1
    z = np.load(run / "steps.npz")
    evals = {int(e["step"]): e for e in json.loads((run / "evals.json").read_text())}
    idx = z["batch_idx"]
    spe = len(labels) // cfg["batch_size"]
    sel = sorted(k for k in z.files if k.startswith("sel/"))
    print(f"\n=== {run}  sampler={cfg['sampler']} args={cfg['sampler_args']} pool_mult={cfg['pool_mult']}")
    print("epoch | " + " ".join(f"{k[4:]:>14s}" for k in sel) + " | distinct  top10%share | class shares (1.0 = uniform; min/median/max if > 10 classes) | val clean @2 @8 (ema)")
    for ep in range(len(idx) // spe):
        s = slice(ep * spe, (ep + 1) * spe)
        flat = idx[s].ravel()
        counts = np.bincount(flat, minlength=len(labels))
        used = np.sort(counts[counts > 0])[::-1]
        top = used[: max(1, len(labels) // 10)].sum() / used.sum()
        cls = np.bincount(labels[flat], minlength=n_classes) / len(flat) * n_classes
        if n_classes > 10:  # too many to print: most and least trained-on class shares
            cls = np.array([cls.min(), np.median(cls), cls.max()])
        ev = evals.get((ep + 1) * spe, {})
        pre = "ema_" if "ema_clean_acc" in ev else ""
        stats = " ".join(f"{np.nanmean(z[k][s]):14.3f}" if np.isfinite(z[k][s]).any() else f"{'-':>14s}" for k in sel)
        print(f"{ep + 1:5d} | {stats} | {len(used):8d}  {top:10.3f} | {cls} | "
              + (f"{ev[pre + 'clean_acc']:.3f} {ev[pre + 'adv_acc@2']:.3f} {ev[pre + 'adv_acc@8']:.3f}" if ev else ""))
