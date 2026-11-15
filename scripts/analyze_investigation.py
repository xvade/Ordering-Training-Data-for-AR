"""Summarise the per-checkpoint counterfactual study written by ``arorder.investigate``.

Usage: python scripts/analyze_investigation.py INVESTIGATE_DIR OUT_DIR

Reads every ``step_*.npz`` and writes, into OUT_DIR:

* ``summary.md``     - human-readable tables (also printed);
* ``checkpoints.csv``- one row per checkpoint: state of the model, alignment of
  held-out gradient directions, population statistics of per-sample first-order
  gains, linearity of real steps, measurement noise floor;
* ``filters.csv``    - for every filter rule: mean paired difference against
  random filtering of the same candidate pool, with standard error over pools;
* ``features.csv``   - rank correlation of every intrinsic sample feature with
  the first-order gains;
* ``deciles.csv``    - mean first-order gains per decile of selected features;
* ``stability.csv``  - rank correlation of per-sample gains between checkpoints.

Light, single-threaded numpy work on a few small files.

Sign conventions: ``gain`` > 0 means training on the sample lowers the held-out
loss in question (helps); ``d_<metric>`` is a measured change after one real
step; ``gap`` is adversarial loss minus clean loss.
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
    if len(pred) < 3 or np.std(pred) == 0 or np.std(meas) == 0:
        return float("nan")
    return float(np.corrcoef(pred, meas)[0, 1] ** 2)


def slope(pred, meas) -> float:
    """Least-squares slope of measured on predicted through the origin (1 = perfectly linear)."""
    return float(np.dot(pred, meas) / max(np.dot(pred, pred), 1e-30))


def derived_gains(d: dict, mid: float) -> dict:
    """First-order gains used throughout the analysis (all 'higher = helps')."""
    names = list(d["vec_names"])
    gram = d["vec_gram"]
    ia, ic = names.index(f"probe_adv@{mid:g}"), names.index("probe_clean")
    adv, clean = d[f"gain/probe_adv@{mid:g}"], d["gain/probe_clean"]
    norm = np.maximum(d["feat/grad_norm"], 1e-30)
    g = {"adv": adv, "clean": clean, "gap": adv - clean, "rob": adv - gram[ia, ic] / gram[ic, ic] * clean}
    g.update({f"cos_{k}": v / norm for k, v in list(g.items())})
    return g


def main(inv_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = sorted((load(p) for p in inv_dir.glob("step_*.npz")), key=lambda d: d["meta"]["step"])
    ck_rows, filt_rows, feat_rows, dec_rows, gain_by_step = [], [], [], [], {}
    for d in data:
        m = d["meta"]
        step = m["step"]
        mid = m["score_eps"][len(m["score_eps"]) // 2]
        adv_name, clean_name = f"probe_adv@{mid:g}", "probe_clean"
        names = list(d["vec_names"])
        gram = d["vec_gram"]

        def cos(a, b):
            i, j = names.index(a), names.index(b)
            return float(gram[i, j] / np.sqrt(gram[i, i] * gram[j, j]))

        G = derived_gains(d, mid)
        gain_by_step[step] = G
        kind, scale = d["trial/kind"], d["trial/step_scale"]
        rnd = kind == "random"
        e_tag = f"{m['eval_eps'][-1]:g}"
        dm = {k[len("trial/d/"):]: v for k, v in d.items() if k.startswith("trial/d/")}
        dm["gap"] = dm[f"adv_loss@{e_tag}"] - dm["clean_loss"]
        pred_adv, pred_clean = d[f"trial/pred/val_adv@{mid:g}"], d["trial/pred/val_clean"]
        meas_adv = dm[f"adv_loss@{mid:g}"] if f"adv_loss@{mid:g}" in dm else dm[f"adv_loss@{e_tag}"]

        def at_scale(mult):
            if mult == 1.0:
                return np.flatnonzero(rnd)[:40]
            return np.flatnonzero((kind == "scale") & np.isclose(scale, mult * m["eta"], rtol=1e-3))

        noise = d[f"noise/d/adv_loss@{e_tag}"]
        ck_rows.append(dict(
            step=step, epoch=round(m["epoch"], 2), lr=m["lr"], eta=m["eta"], ref_step_len=m["ref_step_len"],
            val_clean_acc=float(d["base/clean_acc"]),
            **{f"val_adv_acc@{e:g}": float(d[f"base/adv_acc@{e:g}"]) for e in m["eval_eps"]},
            **{f"val_rgc@{e:g}": float(d[f"base/robust_given_correct@{e:g}"]) for e in m["eval_eps"]},
            val_lin_radius=float(d["base/lin_radius"]),
            val_gap=float(d[f"base/adv_loss@{e_tag}"] - d["base/clean_loss"]),
            # geometry of held-out gradients
            norm_g_clean=float(np.sqrt(gram[names.index(clean_name)] [names.index(clean_name)])),
            norm_g_adv=float(np.sqrt(gram[names.index(adv_name)][names.index(adv_name)])),
            cos_clean_adv=cos(clean_name, adv_name),
            cos_probe_val_adv=cos(adv_name, f"val_adv@{mid:g}"),
            cos_probe_val_clean=cos(clean_name, "val_clean"),
            # population of single-sample first-order effects
            frac_hurt_adv=float((G["adv"] < 0).mean()),
            frac_hurt_clean=float((G["clean"] < 0).mean()),
            frac_help_clean_hurt_adv=float(((G["clean"] > 0) & (G["adv"] < 0)).mean()),
            frac_widen_gap=float((G["gap"] < 0).mean()),
            spearman_gain_adv_clean=float(spearmanr(G["adv"], G["clean"]).statistic),
            mean_gain_adv=float(G["adv"].mean()), mean_gain_clean=float(G["clean"].mean()),
            mean_gain_gap=float(G["gap"].mean()),
            # share of the total adversarial harm (sum of negative gains) carried by the worst 5% of samples
            harm_share_top5pct=float(np.sort(np.minimum(G["adv"], 0))[: len(G["adv"]) // 20].sum()
                                     / min(np.minimum(G["adv"], 0).sum(), -1e-30)),
            gap_harm_share_top5pct=float(np.sort(np.minimum(G["gap"], 0))[: len(G["gap"]) // 20].sum()
                                         / min(np.minimum(G["gap"], 0).sum(), -1e-30)),
            # linearity of real steps on random batches at 0.25x, 1x, 4x the base step
            **{f"r2_adv_x{mu:g}": r2(pred_adv[at_scale(mu)], meas_adv[at_scale(mu)]) for mu in (0.25, 1.0, 4.0)},
            **{f"slope_adv_x{mu:g}": slope(pred_adv[at_scale(mu)], meas_adv[at_scale(mu)]) for mu in (0.25, 1.0, 4.0)},
            r2_clean_x1=r2(pred_clean[rnd], dm["clean_loss"][rnd]),
            # does a score computed on probe predict the effect on val?
            r2_adv_probe_to_val=r2(d[f"trial/pred/{adv_name}"][rnd], meas_adv[rnd]),
            # what a random batch does (mean, spread over batches)
            **{f"rand_mean_d_{k}": float(dm[k][rnd].mean()) for k in ("clean_loss", f"adv_loss@{e_tag}", "gap", "clean_acc", f"adv_acc@{e_tag}", "lin_radius")},
            **{f"rand_std_d_{k}": float(dm[k][rnd].std()) for k in ("clean_loss", f"adv_loss@{e_tag}", "gap", "clean_acc", f"adv_acc@{e_tag}", "lin_radius")},
            # noise floor: spread over PGD start seeds of one step's measured effect, vs spread over batches
            noise_seed_std_adv_loss=float(noise.std(axis=0).mean()),
            noise_batch_std_adv_loss=float(noise.mean(axis=0).std()),
            noise_seed_std_adv_acc=float(d[f"noise/d/adv_acc@{e_tag}"].std(axis=0).mean()),
        ))

        # filter trials: paired difference against random filtering of the same pool
        fsel = kind == "filter"
        rule, pool, nrm = d["trial/rule"], d["trial/pool"], d["trial/norm"]
        metrics = ["clean_loss", f"adv_loss@{e_tag}", "gap", "clean_acc", f"adv_acc@{e_tag}",
                   f"robust_given_correct@{e_tag}", "lin_radius"]
        for norm_flag in (0, 1):
            ref = {int(pool[i]): i for i in np.flatnonzero(fsel & (rule == "random") & (nrm == norm_flag))}
            for rname in np.unique(rule[fsel]):
                rows = np.flatnonzero(fsel & (rule == rname) & (nrm == norm_flag))
                ref_rows = np.array([ref[int(pool[i])] for i in rows])
                rec = dict(step=step, epoch=round(m["epoch"], 2), rule=rname, norm=norm_flag, n=len(rows),
                           step_len_ratio=float((scale[rows] * d["trial/grad_norm"][rows]).mean() / m["ref_step_len"]))
                for k in metrics:
                    diff = dm[k][rows] - dm[k][ref_rows] if rname != "random" else dm[k][rows]
                    rec[f"{k}"] = float(diff.mean())
                    rec[f"{k}_se"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                filt_rows.append(rec)

        # intrinsic features vs first-order gains
        feats = {k[len("feat/"):]: v for k, v in d.items() if k.startswith("feat/") and k != "feat/label"}
        for fname, fv in feats.items():
            feat_rows.append(dict(step=step, feature=fname,
                                  **{t: float(spearmanr(fv, G[t]).statistic) for t in ("adv", "clean", "gap", "rob", "cos_gap", "cos_rob")}))
        for fname in ("loss", "margin", "grad_norm", "lin_radius", f"adv_margin@{mid:g}", "self_cos"):
            order = np.argsort(feats[fname])
            for dec, members in enumerate(np.array_split(order, 10)):
                dec_rows.append(dict(step=step, feature=fname, decile=dec, feat_mean=float(feats[fname][members].mean()),
                                     **{f"gain_{t}": float(G[t][members].mean()) for t in ("adv", "clean", "gap", "rob")},
                                     frac_hurt_adv=float((G["adv"][members] < 0).mean()),
                                     frac_widen_gap=float((G["gap"][members] < 0).mean())))

    steps = sorted(gain_by_step)
    stab_rows = [dict(step_a=a, step_b=b, **{t: float(spearmanr(gain_by_step[a][t], gain_by_step[b][t]).statistic)
                                             for t in ("adv", "clean", "gap", "rob", "cos_gap", "cos_rob")})
                 for i, a in enumerate(steps) for b in steps[i + 1:]]

    ck, filt, feat, dec, stab = map(pd.DataFrame, (ck_rows, filt_rows, feat_rows, dec_rows, stab_rows))
    ck.to_csv(out_dir / "checkpoints.csv", index=False)
    filt.to_csv(out_dir / "filters.csv", index=False)
    feat.to_csv(out_dir / "features.csv", index=False)
    dec.to_csv(out_dir / "deciles.csv", index=False)
    stab.to_csv(out_dir / "stability.csv", index=False)

    pd.set_option("display.width", 260, "display.max_columns", 80, "display.max_rows", 500,
                  "display.float_format", lambda v: f"{v:.4g}")
    text = ["# Per-checkpoint summary\n", ck.set_index("step").T.to_string()]
    text.append("\n\n# Filter trials: mean paired difference vs random filtering of the same pool\n"
                "(`random` rows show the absolute effect of a random batch. z = mean / standard error.)\n")
    key = filt.columns[filt.columns.str.startswith("adv_loss@") & ~filt.columns.str.endswith("_se")][0]
    for norm_flag, label in ((1, "step length equalised (direction only)"), (0, "raw step (eta * mean gradient)")):
        for metric in ("gap", key, "clean_loss", "lin_radius"):
            sub = filt[filt.norm == norm_flag]
            z = sub.assign(z=sub[metric] / sub[f"{metric}_se"].replace(0, np.nan))
            text.append(f"\n## {label}: change in `{metric}` (rows: rule, columns: epoch) and z-scores\n")
            text.append(z.pivot(index="rule", columns="epoch", values=metric).to_string())
            text.append("\nz:\n" + z.pivot(index="rule", columns="epoch", values="z").round(1).to_string())
    text.append("\n\n# Spearman(feature, first-order gain), rows: feature, columns: step\n")
    for t in ("adv", "gap", "rob", "cos_rob"):
        text.append(f"\n## target: gain `{t}`\n" + feat.pivot(index="feature", columns="step", values=t).to_string())
    if len(stab):
        text.append("\n\n# Rank stability of per-sample gains between checkpoints\n" + stab.to_string(index=False))
    (out_dir / "summary.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
