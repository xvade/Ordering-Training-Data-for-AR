"""Print the mean of every sampler statistic (``sel/*``) in a run's step log, by quarter of the run.

Usage: python scripts/adhoc/sel_stats.py RUN_DIR
"""
import sys

import numpy as np

z = np.load(sys.argv[1] + "/steps.npz")
n = len(z["lr"])
print("quarters of the run:", [f"{a}-{b}" for a, b in zip(range(0, n, max(n // 4, 1)), range(max(n // 4, 1), n + 1, max(n // 4, 1)))])
for k in sorted(z.files):
    if k.startswith("sel/") or k in ("train_loss", "train_acc"):
        print(f"{k:24s}", [round(float(np.nanmean(q)), 3) if np.isfinite(q).any() else None for q in np.array_split(z[k], 4)])
