"""Summarise the per-checkpoint counterfactual study written by ``arorder.investigate``.

Usage: python scripts/analyze_investigation.py INVESTIGATE_DIR OUT_DIR

Reads every ``step_*.npz`` and writes, into OUT_DIR:

* ``summary.md``   - human-readable tables (also printed);
* ``summary.json`` - the same numbers, machine-readable;
* ``bins.csv``     - measured effect of real steps on batches drawn from each score-quantile bin;
* ``features.csv`` - rank correlation of every intrinsic sample feature with every first-order gain;
* ``deciles.csv``  - mean first-order gains per decile of selected features.

This is light, single-threaded numpy work on a few small files.

Sign convention: ``gain`` > 0 means training on the sample lowers the held-out
loss in question (helps); ``d/<metric>`` is the measured change after one real step.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def r2(pred, meas) -> float:
    """Squared Pearson correlation between predicted and measured changes."""
    if np.std(pred) == 0 or np.std(meas) == 0:
        return float("nan")
    return float(np.corrcoef(pred, meas)[0, 1] ** 2)


def main(inv_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(inv_dir.glob("step_*.npz"), key=lambda p: load(p)["meta"]["step"])
    rows, bin_rows, feat_rows, dec_rows = [], [], [], []
    for f in files:
        d = load(f)
        m = d["meta"]
        mid = m["score_eps"][len(m["score_eps"]) // 2]
        adv, clean = f"probe_adv@{mid:g}", "probe_clean"
        names = list(d["vec_names"])
        gram = d["vec_gram"]
        cos = lambda a, b: float(gram[names.index(a), names.index(b)]
                                 / np.sqrt(gram[names.index(a), names.index(a)] * gram[names.index(b), names.index(b)]))
        g_adv, g_clean = d[f"gain/{adv}"], d[f"gain/{clean}"]
        g_gap = g_adv - g_clean
        kind = d["trial/kind"]
        rnd = kind == "random"
        e_tag = f"{m['eval_eps'][-1]:g}"
        noise = d[f"noise/d/adv_loss@{e_tag}"]  # (seeds, batches)
        row = dict(
            step=m["step"], epoch=round(m["epoch"], 2), eta=m["eta"],
            val_clean_acc=float(d["base/clean_acc"]), **{f"val_adv_acc@{e:g}": float(d[f"base/adv_acc@{e:g}"]) for e in m["eval_eps"]},
            val_lin_radius=float(d["base/lin_radius"]),
            cos_clean_adv=cos(clean, adv),
            cos_probe_val_adv=cos(adv, f"val_adv@{mid:g}"),
            cos_probe_val_clean=cos(clean, "val_clean"),
            # population of single-sample effects
            frac_hurt_adv=float((g_adv < 0).mean()),
            frac_hurt_clean=float((g_clean < 0).mean()),
            frac_help_clean_hurt_adv=float(((g_clean > 0) & (g_adv < 0)).mean()),
            frac_widen_gap=float((g_gap < 0).mean()),
            spearman_gain_adv_clean=float(spearmanr(g_adv, g_clean).statistic),
            # how concentrated is the harm: share of total negative adv gain carried by the worst 5% of samples
            harm_share_top5pct=float(np.sort(np.minimum(g_adv, 0))[: len(g_adv) // 20].sum() / min(np.minimum(g_adv, 0).sum(), -1e-30)),
            # does first order predict real steps? (random batches)
            r2_adv_loss_random=r2(d[f"trial/pred/val_adv@{mid:g}"][rnd], d[f"trial/d/adv_loss@{mid:g}"][rnd]) if f"trial/d/adv_loss@{mid:g}" in d else float("nan"),
            r2_clean_loss_random=r2(d["trial/pred/val_clean"][rnd], d["trial/d/clean_loss"][rnd]),
            r2_adv_loss_probe_to_val=r2(d[f"trial/pred/{adv}"][rnd], d[f"trial/d/adv_loss@{mid:g}"][rnd]) if f"trial/d/adv_loss@{mid:g}" in d else float("nan"),
            # random-batch effects on val: mean and spread
            **{f"rand_mean_d_{k}": float(d[f"trial/d/{k}"][rnd].mean()) for k in ("clean_loss", f"adv_loss@{e_tag}", "clean_acc", f"adv_acc@{e_tag}", "lin_radius")},
            **{f"rand_std_d_{k}": float(d[f"trial/d/{k}"][rnd].std()) for k in ("clean_loss", f"adv_loss@{e_tag}", "clean_acc", f"adv_acc@{e_tag}", "lin_radius")},
            # noise floor: spread across PGD seeds of the same step's measured effect, vs spread across batches
            noise_seed_std_adv_loss=float(noise.std(axis=0).mean()),
            noise_batch_std_adv_loss=float(noise.mean(axis=0).std()),
            noise_seed_std_adv_acc=float(d[f"noise/d/adv_acc@{e_tag}"].std(axis=0).mean()),
            noise_batch_std_adv_acc=float(d[f"noise/d/adv_acc@{e_tag}"].mean(axis=0).std()),
        )
        rows.append(row)

        # real steps on quantile bins
        metrics = [k[len("trial/d/"):] for k in d if k.startswith("trial/d/")]
        for score in ("adv", "clean", "gap"):
            sel = kind == f"bin:{score}"
            for b in np.unique(d["trial/bin"][sel]):
                s = sel & (d["trial/bin"] == b)
                bin_rows.append(dict(step=m["step"], score=score, bin=int(b), n=int(s.sum()),
                                     **{f"d_{k}": float(d[f"trial/d/{k}"][s].mean()) for k in metrics}))
        bin_rows.append(dict(step=m["step"], score="random", bin=-1, n=int(rnd.sum()),
                             **{f"d_{k}": float(d[f"trial/d/{k}"][rnd].mean()) for k in metrics}))

        # intrinsic features vs gains
        feats = {k[len("feat/"):]: d[k] for k in d if k.startswith("feat/") and k != "feat/label"}
        targets = {"gain_adv": g_adv, "gain_clean": g_clean, "gain_gap": g_gap}
        for fname, fv in feats.items():
            feat_rows.append(dict(step=m["step"], feature=fname,
                                  **{t: float(spearmanr(fv, tv).statistic) for t, tv in targets.items()}))
        for fname in ("loss", "margin", "grad_norm", "lin_radius", f"adv_margin@{mid:g}"):
            order = np.argsort(feats[fname])
            for dec, members in enumerate(np.array_split(order, 10)):
                dec_rows.append(dict(step=m["step"], feature=fname, decile=dec,
                                     feat_mean=float(feats[fname][members].mean()),
                                     gain_adv=float(g_adv[members].mean()), gain_clean=float(g_clean[members].mean()),
                                     gain_gap=float(g_gap[members].mean()),
                                     frac_hurt_adv=float((g_adv[members] < 0).mean())))

    summary = pd.DataFrame(rows)
    bins = pd.DataFrame(bin_rows)
    feats = pd.DataFrame(feat_rows)
    decs = pd.DataFrame(dec_rows)
    summary.to_json(out_dir / "summary.json", orient="records", indent=1)
    bins.to_csv(out_dir / "bins.csv", index=False)
    feats.to_csv(out_dir / "features.csv", index=False)
    decs.to_csv(out_dir / "deciles.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", lambda v: f"{v:.4g}")
    text = ["# Per-checkpoint summary\n", summary.T.to_string(), "\n\n# Spearman(feature, gain) per step\n"]
    text.append(feats.pivot(index="feature", columns="step", values="gain_adv").to_string())
    text.append("\n\n(above: gain_adv; below: gain_gap)\n")
    text.append(feats.pivot(index="feature", columns="step", values="gain_gap").to_string())
    (out_dir / "summary.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
