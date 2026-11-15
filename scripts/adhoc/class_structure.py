"""Ad-hoc check (2026-11-15): how much of the per-sample first-order gain is explained by the class label?

Usage: python scripts/adhoc/class_structure.py INVESTIGATE_DIR
Prints, per checkpoint, the per-class mean of clean gain, the share of gain variance
explained by class (eta^2), and the same for the robustness-only gain.
"""
import json
import sys
from pathlib import Path

import numpy as np

np.set_printoptions(linewidth=220, precision=2, suppress=True)
for f in sorted(Path(sys.argv[1]).glob("step_*.npz")):
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    m = json.loads(str(d["meta"]))
    names = list(d["vec_names"]); gram = d["vec_gram"]
    ia, ic = names.index("probe_adv@2"), names.index("probe_clean")
    clean, adv = d["gain/probe_clean"], d["gain/probe_adv@2"]
    rob = adv - gram[ia, ic] / gram[ic, ic] * clean
    y = d["feat/label"]

    def eta2(v):
        means = np.array([v[y == c].mean() for c in range(10)])
        return float(((means[y] - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum()), means

    e_c, mc = eta2(clean); e_r, mr = eta2(rob); e_n, _ = eta2(clean / np.maximum(d["feat/grad_norm"], 1e-30))
    # within-class: correlation of gain with loss after removing class means
    res = clean - mc[y]; loss = d["feat/loss"]; lres = loss - np.array([loss[y == c].mean() for c in range(10)])[y]
    print(f"step {m['step']:5d} ep {m['epoch']:5.2f} | eta2(class): clean gain {e_c:.2f}, cos-clean {e_n:.2f}, rob gain {e_r:.2f} "
          f"| corr(within-class gain, loss) {np.corrcoef(res, lres)[0,1]:.2f} | frac clean gain<0: {(clean<0).mean():.2f}")
    print("   class mean clean gain / overall mean:", mc / clean.mean())
    print("   class mean rob gain / std:           ", mr / rob.std())
