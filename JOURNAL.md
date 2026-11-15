# Experiment journal

Newest entries at the bottom. Every entry records when it ran and on what
hardware. Bulky run directories live in `/gscratch/scrubbed/sgvtc/ordering-ar-runs/`
(purged after 21 days without access); the numbers that matter are copied here
or into `results/`.

## 2026-11-15 — Setup

- **Cluster**: klone. Notes in `docs/KLONE.md`.
- **Environment**: uv virtualenv at `/gscratch/amath/sgvtc/ordering-ar/venv`
  (Python 3.11.15, torch 2.6.0+cu124, torchvision 0.21.0; ~28k files, 8 GB).
  Built in Slurm job 43269972 on g3026 (Quadro RTX 6000). Exact versions in
  `requirements.lock.txt`. It lives in group storage, not scrubbed, because
  scrubbed deletes untouched files one by one and would corrupt it.
- **Data**: CIFAR-10 python archive from cs.toronto.edu, converted to a single
  `cifar10.npz` in `/gscratch/scrubbed/sgvtc/ordering-ar-data/` (reproducible,
  so scrubbed is fine). Not available under `/mmfs1/data`.
- **Splits** (fixed, `make_splits(seed=0)`): 45,000 train / 2,000 probe /
  3,000 val out of the official 50,000; the official 10,000 test images are
  reserved for final reporting and are not used by any selection or tuning code.
- **Unit tests**: 9 passed (job 43270087, g-node RTX 6000, 10:28 PST).

### Measurement design decisions (and why)

1. **Robustness is measured at small radii and with continuous quantities.**
   A standard-trained net has ~0% accuracy under the usual 8/255 attack, so that
   number cannot reveal decay. We track PGD accuracy/loss/margin at 0.5-8/255,
   the area under the robust-accuracy curve, and a first-order boundary distance.
2. **Paired evaluation.** PGD random starts are fixed per evaluator, so two sets
   of weights are attacked from identical starts. The residual noise (PGD path
   sensitivity) is measured by repeating with different start seeds.
3. **Attribution by counterfactual replay.** From a saved checkpoint we apply a
   pure gradient step for a chosen batch and re-measure. This removes momentum
   history and weight decay, which are not something a data-selection rule
   controls. The baseline model uses GroupNorm, so a step has no hidden effect
   through running statistics. BatchNorm and momentum are checked later.
4. **Robustness is always reported next to clean accuracy**, and
   "robust given correct" is tracked, so a model that just learns less cannot
   pass as robust.
5. **Probe vs val.** Scores that drive selection are computed on `probe`;
   their effect is measured on `val`.
