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

## 2026-11-15 10:35-11:10 PST — Benchmarks and two measurement pitfalls

Hardware: g3030, Quadro RTX 6000 (24 GB), 4 CPUs, jobs 43270183, 43270397, 43270605.
Model: SmallCNN width 32, GroupNorm (0.28 M parameters), SGD lr 0.05, momentum 0.9,
1 warm-up epoch then cosine, batch 128. Checkpoint studied: end of epoch 1
(`bench_probe`, step 351, val clean accuracy 32.9%).

**Speed.** One epoch without per-step probing: ~8 s. With a probe measurement
before every update (1000 points, PGD-5 at 1/255 and 2/255): ~135 s per epoch.
One real replayed step plus a PGD-10 evaluation on the 3000 val points: ~1.6 s.
Per-sample gradient products for all 45,000 train samples: ~50 s.

**Noise floor.** With fixed PGD starts, the measured effect of one step on val
adversarial loss changes by ~6e-5 (std) across PGD start seeds, against ~1e-2
spread between random batches. Single-update effects are measurable with a
signal-to-noise ratio above 100.

**Pitfall 1: step size.** First-order predictions of the change in val
adversarial loss against the measured change, random batches, at step sizes
relative to the learning rate (0.05):

| step size | R^2 | slope (1 = linear) |
|---|---|---|
| 0.0625 x lr | 0.998 | 0.97 |
| 0.25 x lr | 0.93 | 0.84 |
| 1 x lr | 0.28 | 0.34 |
| 10 x lr (= lr / (1 - momentum), one shot) | 0.005 | val loss rises by 2.3, accuracy falls 19 points |

Training sits near the edge of stability, so a full-lr pure gradient step from a
trajectory point is already nonlinear. The replay study uses 0.25 x lr.

**Pitfall 2: gradient magnitude.** Batches drawn from the extreme quantiles of a
first-order score (either end) are made of similar high-loss samples; their
mean gradient is many times longer than a random batch's, and the step just
overshoots (val loss +1.2 to +2.5 at step size lr). Raw comparisons between
selected batches therefore measure step length. The replay study now runs every
selection twice: raw, and with the step rescaled to the median random-batch
step length.

**First look at filtering (epoch 1 only, 3 pools, not yet conclusive).** With
step length equalised, *every* rule that keeps a preferred half of a 256-sample
pool reduces val adversarial loss *less* than keeping a random half. A random
batch is an unbiased estimate of the population gradient, which at this point
is 0.97 cosine-aligned with the held-out adversarial gradient; any selection
bends the direction away from it. Rules aimed at the adversarial-minus-clean
gap do shrink the gap, but by making less clean progress. So "gap" on its own
is confounded with progress; the analysis needs a measure of robustness change
at matched clean progress.

**The phenomenon is present in the baseline.** In `base_gn_s0` (running), val
accuracy under PGD at 2/255 is 22.0% after epoch 1 and 20.1% after epoch 11,
while clean accuracy goes from 31% to 73%.
