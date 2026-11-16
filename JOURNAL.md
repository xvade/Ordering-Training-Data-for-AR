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

## 2026-11-15 10:43-11:27 PST — Instrumented baseline `base_gn_s0`

Hardware: g3030, Quadro RTX 6000, 4 CPUs, job 43270336, 44 min (probe before every update).
Config: SmallCNN w32 GroupNorm, 20 epochs, batch 128, SGD lr 0.05 momentum 0.9 wd 5e-4,
1 warm-up epoch + cosine, standard augmentation, seed 0. Run dir
`/gscratch/scrubbed/sgvtc/ordering-ar-runs/base_gn_s0` (60 checkpoints, every 117 steps).
Summary tables and figure: `results/phase1/trajectory_epochs.csv`, `trajectory.png`, `worst_steps.csv`.

Val split (3000 points, PGD-10), by epoch:

| epoch | clean | @1/255 | @2/255 | @4/255 | AUC(0-8) | lin. radius |
|---|---|---|---|---|---|---|
| 1 | 0.313 | 0.263 | 0.220 | 0.153 | 1.34 | 1.54 |
| 2 | 0.387 | 0.327 | 0.273 | 0.171 | 1.56 | 2.01 |
| 5 | 0.580 | 0.446 | 0.315 | 0.138 | 1.66 | 1.88 |
| 8 | 0.680 | 0.452 | 0.257 | 0.057 | 1.35 | 1.33 |
| 11 | 0.731 | 0.459 | 0.201 | 0.025 | 1.20 | 1.21 |
| 14 | 0.762 | 0.456 | 0.197 | 0.022 | 1.20 | 1.20 |
| 17 | 0.794 | 0.451 | 0.172 | 0.014 | 1.15 | 1.14 |
| 20 | 0.799 | 0.456 | 0.172 | 0.013 | 1.15 | 1.14 |

What the per-update log shows (probe split, 1000 points, PGD-5):

1. **Robustness rises, peaks early, then decays while clean accuracy keeps
   rising.** Accuracy at 2/255 peaks at ~0.31 in epoch 4 and ends at 0.185; at
   4/255 it peaks in epoch 2 (0.17) and ends at 0.013. The linearised boundary
   distance peaks in epoch 3 (2.0/255) and ends at 1.14/255. Robust-given-correct
   at 2/255 falls monotonically from 0.73 to 0.23.
2. **The decay is a slow drift, not a few bad updates.** The adversarial-minus-clean
   loss gap rises by ~0.15 per epoch, i.e. ~4e-4 per update, while single updates
   move it by +-0.04 (std). Half of all updates raise the gap and half lower it,
   in every epoch. The worst 10% of updates account for about half of the upward
   movement, but that is what a symmetric noisy walk looks like; it does not
   identify a small set of culprit batches.
3. **Clean and adversarial loss move together update by update** (correlation
   0.99 in epoch 1, 0.9 mid-training, falling to 0.4 and below only when the
   learning rate is nearly zero), with the adversarial loss moving more. The
   fraction of updates that lower clean loss but raise adversarial loss grows
   from 2% (epoch 1) to 10% (epoch 10) to 29% (epoch 20).
4. **Loss-based robustness numbers are confounded by logit scale.** The mean
   clean margin grows from -0.5 to 2.3 and the adversarial margin at 2/255 falls
   from -0.7 to -2.9: the network becomes more confident, including when wrong.
   Accuracy under attack and the linearised radius are scale-free and are the
   quantities to trust.

Consequence for the method: because harm is a small systematic bias under large
symmetric noise, a selection rule has to shift the *average direction* of
updates; removing rare outlier batches cannot be the mechanism.

## 2026-11-15 11:27-12:10 PST — Single-update study at raw iterates (epochs 0.33-6), first sweeps, weight averaging

Hardware: g3026/g3030-g3033, Quadro RTX 6000, 4 CPUs per job. Jobs: replay study
43270749-50 and 43271337-39 (`arorder.investigate` on `base_gn_s0`); sweeps 43271007
(A), 43271008 (B), 43271300 (C) into `ordering-ar-runs/p2a`; averaged baseline 43272742
(`base_ema_s0`). Tables: `results/phase1/investigate/`, `results/p2a/`.

### A. One-step replay at raw training iterates (checkpoints at epochs 0.33, 1, 2, 4, 6)

Setup: pure gradient steps of 0.25 x lr from the checkpoint; effect measured on val
(3000 points, PGD-10, fixed starts). Per-sample first-order gains for all 45,000
train samples.

1. **First-order reasoning is reliable at this step size.** Predicted vs measured
   change of val adversarial loss on random batches: R^2 0.95-0.99 at 0.25 x lr,
   0.98-1.00 at 0.0625 x lr, 0.17-0.79 at 1 x lr. Scores computed on `probe`
   predict the effect on `val` (R^2 0.95-0.99); even the "robustness-only"
   direction (adversarial gradient with the clean gradient projected out) agrees
   between probe and val (cosine 0.97-0.99).
2. **There is no separate population of robustness-harming samples.** A sample's
   effect on held-out adversarial loss and on held-out clean loss are almost the
   same ranking (Spearman 0.97-0.999). Only 0.6-4% of samples lower clean loss
   while raising adversarial loss. 23-45% of samples raise *both*.
3. **One-step effects are dominated by transient class imbalance.** The class
   label alone explains 76% of the variance of the clean gain at epoch 0.33, 40%
   at epoch 1 and 19-29% at epochs 2-6 (40-80% for the direction, i.e. gain
   divided by gradient norm), and 11-44% of the robustness-only gain. Which
   classes help flips between checkpoints (rank correlation of per-sample clean
   gains between epoch 0.33 and epoch 1 is -0.70). At a given moment the iterate
   over-predicts some classes; training on those classes hurts, training on the
   others helps, and the roles rotate.
4. **The robustness-only gain is not a stable property of a sample.** No
   intrinsic feature (loss, margin, gradient norm, boundary distance, own
   adversarial margin) predicts it (|Spearman| < 0.25), and its rank correlation
   between checkpoints ranges from -0.28 to 0.64.
5. **A pure gradient step from a trajectory point looks far better than what
   training achieves.** At epoch 6 a random-batch step of 0.25 x lr lowers val
   clean loss by 0.20 and adversarial loss by 0.34; a whole epoch of training
   moves clean loss by about 0.1. The iterate sits on the wall of a sharp valley
   and a small step slides it back towards the floor. One-step attribution at a
   raw iterate therefore mostly measures this fast transverse motion, not the
   slow drift along the valley that changes robustness over epochs.
6. **Loss-based robustness targets do not move robust accuracy consistently.**
   With step length equalised, keeping the half of a pool preferred by the
   robustness-only gain lowers "adversarial loss at matched clean loss" at every
   checkpoint (z about -20), but its effect on accuracy under attack at matched
   clean accuracy changes sign from checkpoint to checkpoint (+0.002, -0.005,
   +0.005, -0.003) and on the linearised radius likewise.

### B. First full-training sweeps (3 seeds each, 20 epochs unless noted, val split, raw weights)

| variant | clean | @2/255 | @4/255 | AUC |
|---|---|---|---|---|
| base | 0.795 +- 0.004 | 0.167 +- 0.019 | 0.014 | 1.14 +- 0.06 |
| base, 10 epochs | 0.704 +- 0.007 | 0.218 +- 0.032 | 0.034 | 1.22 +- 0.11 |
| base, 5 epochs | 0.593 +- 0.003 | 0.314 +- 0.012 | 0.113 | 1.58 +- 0.05 |
| keep lowest-loss half of each 256 pool | 0.522 +- 0.031 | 0.251 +- 0.013 | 0.075 | 1.26 +- 0.08 |
| keep highest-loss half | 0.815 +- 0.006 | 0.109 +- 0.012 | 0.006 | 0.99 +- 0.04 |
| keep largest boundary-distance half | 0.525 +- 0.032 | 0.262 +- 0.018 | 0.078 | 1.30 +- 0.07 |
| keep smallest boundary-distance half | 0.822 +- 0.005 | 0.112 +- 0.012 | 0.005 | 1.00 +- 0.03 |
| keep middle half by boundary distance | 0.727 +- 0.004 | 0.193 +- 0.021 | 0.022 | 1.15 +- 0.06 |
| keep best half by robustness-only gain (probe) | 0.475 +- 0.037 | 0.283 +- 0.009 | 0.143 | 1.52 +- 0.03 |

* Easy-sample rules give a more robust but much less accurate model; hard-sample
  rules the reverse. This is the same axis as training shorter or longer.
* **Easy-sample selection is dominated by simply training the baseline for 5
  epochs** (0.593 clean / 1.58 AUC against 0.52 / 1.26-1.30). Any claim for a
  selection rule has to be made against the baseline's whole
  accuracy-robustness curve, not against the 20-epoch baseline alone.
* The robustness-only rule collapsed the model to chance within one epoch of
  being switched on (a constant predictor is trivially "robust") before partly
  recovering. Selection has ample leverage over the update direction; used
  without a constraint on clean progress it is destructive.

### C. Weight averaging (`base_ema_s0`, EMA decay 0.99, same seed as base)

| epoch | raw clean | raw @2 | raw AUC | EMA clean | EMA @2 | EMA AUC |
|---|---|---|---|---|---|---|
| 2 | 0.402 | 0.269 | 1.54 | 0.468 | 0.316 | 1.76 |
| 4 | 0.547 | 0.305 | 1.59 | 0.571 | 0.349 | 1.86 |
| 6 | 0.558 | 0.254 | 1.30 | 0.653 | 0.320 | 1.60 |
| 10 | 0.680 | 0.218 | 1.21 | 0.731 | 0.245 | 1.32 |
| 14 | 0.770 | 0.198 | 1.20 | 0.780 | 0.219 | 1.26 |
| 20 | 0.798 | 0.176 | 1.16 | 0.799 | 0.176 | 1.16 |

* The averaged weights are both more accurate and more robust than the raw
  iterate while the learning rate is high: the oscillation costs both.
* The rise-then-decay of robustness is still there in the averaged weights
  (accuracy at 2/255 peaks at 0.349 in epoch 4 and ends at 0.176), so the decay
  is a real drift of the solution and not an artefact of the oscillation.
* Decision: study single updates at the averaged weights (no transverse
  component), and judge selection rules on averaged weights as well as raw.

### Bugs found and fixed

* Checkpoints written after warm-up contained a numpy scalar (the learning rate)
  and could not be loaded with `torch.load(weights_only=True)`; three replay jobs
  failed after 24 s. Fixed at the source (`lr_at` returns a Python float) and
  `load_checkpoint` falls back for the legacy files.
* `scripts/env.sh` did not export `PY`, so `bash -c '$PY ...'` ran an empty
  command. Exported.
* A unit test demanded bit-identical repeated PGD evaluations; cuDNN is not
  bit-reproducible, so it failed once in a while. Now uses a tolerance.

## 2026-11-15 12:10-13:15 PST — Replay at averaged weights, more sweeps, planning agent

Hardware: g3026/g3030-g3033 (Quadro RTX 6000, 4 CPUs per job); planner on n3432
(cpu-g2, 4 CPUs, no GPU). Jobs: replay at averaged weights 43272835-37
(`base_ema_s0/investigate`, 9 checkpoints, 60 random batches + 15 pools x 13 rules
each); sweeps D 43272897 (`p2b`), E 43273186 (`p2c_branch7`), F 43273312-13
(`p2d_linr`), G 43273677-78 (`p3_offset`, running); planner 43272890.
Tables: `results/phase1/investigate_ema/`, `results/p2b`, `results/p2c_branch7`,
`results/p2d_linr`.

### A. What the averaged weights show over training (val, `base_ema_s0`)

| epoch | clean | @2/255 | lin. radius | cos(clean grad, adv grad) | ‖adv grad‖ / ‖clean grad‖ | samples that help clean but hurt adv |
|---|---|---|---|---|---|---|
| 1 | 0.365 | 0.265 | 2.01 | 0.975 | 1.2 | 1.6% |
| 3 | 0.524 | 0.347 | 2.16 | 0.925 | 1.5 | 2.2% |
| 4 | 0.571 | 0.349 | 2.19 | 0.879 | 1.7 | 4.7% |
| 6 | 0.653 | 0.320 | 1.72 | 0.768 | 2.2 | 8.6% |
| 8 | 0.695 | 0.276 | 1.41 | 0.721 | 2.7 | 6.6% |
| 11 | 0.745 | 0.231 | 1.25 | 0.539 | 3.9 | 10.2% |
| 14 | 0.780 | 0.219 | 1.21 | 0.520 | 4.0 | 5.9% |
| 17 | 0.794 | 0.192 | 1.15 | 0.376 | 4.8 | 9.9% |
| 20 | 0.799 | 0.176 | 1.12 | 0.286 | 4.8 | 12.0% |

(gradients are of the held-out loss, adversarial at 2/255, measured on `probe`)

* **The direction that lowers clean loss and the direction that lowers
  adversarial loss start out identical and separate steadily**: cosine 0.975
  at epoch 1, 0.29 at epoch 20, while the adversarial gradient grows 2.2x and
  the clean gradient shrinks. This is the cleanest single description of the
  decay we have: late in training, making clean progress and making robust
  progress are nearly unrelated directions, and training follows the first.
* At the averaged weights a random step has the trajectory's signs from epoch 8
  on (clean accuracy up, accuracy at 2/255 and boundary distance down), which it
  does not at raw iterates. First-order predictions hold (R^2 0.93-0.99 at the
  full base step) until the learning rate is nearly zero.
* Per-sample gains are stable between checkpoints at the averaged weights (rank
  correlation 0.5-0.85) but about a third of their variance is class
  membership, and roughly half of all samples raise held-out clean loss.

### B. Filter trials at averaged weights

* Selecting by first-order benefit to the held-out **linearised radius**
  (`gain_linr`, a scale-free target) raises the radius relative to random steps
  at every checkpoint (equal step length: +0.009, +0.004, +0.003 at epochs 4, 8,
  14; z = 5.5-12.5) with clean-accuracy progress equal to random at epochs 4 and
  8. Easy-sample rules raise the radius as much but cancel clean progress.
* The selected batches are not neutral in size: gradient-guided halves take
  steps 1.4-2.7x longer than a random batch, easy halves 0.13-0.5x.

### C. Full-training tests (3 seeds unless noted; val; raw / averaged weights agree at the end of a run)

| variant | clean | @2/255 | @4/255 | AUC |
|---|---|---|---|---|
| uniform, 5 epochs | 0.597 | 0.312 | 0.114 | 1.58 |
| uniform, 10 epochs | 0.702 | 0.221 | 0.035 | 1.23 |
| uniform, 20 epochs | 0.797 | 0.171 | 0.012 | 1.14 |
| uniform, 40 epochs | 0.841 | 0.106 | 0.004 | 1.00 |
| class-balanced batches, 5 epochs (2 seeds) | 0.642 | 0.282 | 0.070 | 1.42 |
| class-balanced batches, 10 epochs | 0.734 | 0.199 | 0.023 | 1.18 |
| class-balanced batches, 20 epochs | 0.809 | 0.150 | 0.008 | 1.09 |
| from epoch 7: keep easiest half | 0.594 | 0.263 | 0.069 | 1.33 |
| from epoch 7: drop hardest quarter | 0.740 | 0.228 | 0.028 | 1.27 |
| from epoch 7: drop the quarter nearest the boundary (2 seeds) | 0.736 | 0.220 | 0.024 | 1.24 |
| keep best half by radius gain, scored on averaged weights | 0.436 | 0.231 | 0.103 | 1.24 |
| keep best half by radius gain, scored on raw weights (2 seeds) | 0.358 | 0.161 | 0.075 | 0.92 |

* **Class-balanced batches train faster and land on the same curve**, slightly
  further along it (more accurate, less robust).
* **Hard-sample rules are matched by training longer** (40 epochs: 0.841 / 0.106).
* **Dropping the hardest quarter from epoch 7** ends at 0.740 / 0.228 / 1.27.
  The uniform annealed endpoints interpolate to about 0.20 / 1.20 at that
  accuracy, but the uniform run's own averaged weights mid-run reach about
  0.24 there (planner, table D). So it beats "stop the cosine schedule earlier"
  and not "take the averaged weights earlier".
* **The radius-gain rule fails in closed loop** (0.44 clean after 20 epochs),
  even though it was the best rule in one-step replay. Applied at every step
  with a step ~1.5-2x longer and momentum on top, the same bias accumulates
  coherently while the random part averages out. One-step replay, at raw or
  averaged weights, is not a safe guide to what a rule does over thousands of
  steps.

### D. Planning agent (Fable, max effort) — `docs/plans/2026-11-15_phase2_plan.md`

Ran inside its own Slurm allocation, re-analysed the raw files and ran two CPU
pilots. Its conclusions that change the plan:

1. Vulnerability is a **global loss of input-space margin**: between epochs 2
   and 14 the margin of correct training points grows 2.8x while its
   input-gradient norm grows 8x. Samples correct at every checkpoint lose
   robustness too (accuracy at 2/255: 0.80 to 0.49).
2. It is **class-structured**: 94% of the 2/255 accuracy lost between epochs 4
   and 14 is lost in the seven easier classes while bird/cat/deer are being
   learned; the selection rules tried so far are class reweightings in disguise
   (the high-loss half of a pool is 18% cats, 4.5% cars).
3. **Weight-space operations move off the curve; no data rule has**: averaged
   weights taken mid-run, and interpolation between the epoch-4 average and the
   final weights (0.728 clean / 0.315 at 2/255 against 0.262 on the averaged
   path; one run).
4. The final learning-rate anneal is the most expensive phase: 1.5-3.0 points of
   2/255 accuracy lost per point of clean accuracy gained, against 0.5-1.1 before.
5. **A run's offset from the curve is set early and persists** (mid-run offset
   predicts late offset, r = 0.91 over ten runs; three uniform branches from one
   epoch-7 checkpoint end within 0.006 of each other), and in its pilot **the
   data order alone, from identical initial weights, moves that offset** (three
   orders span 0.038 in 2/255 accuracy).
6. Recommendation: (a) lookahead order selection in the early window, (b) class
   exposure as the second lever, (c) a fair reference curve built from
   averaging, early stopping and interpolation, (d) adversarial training as a
   separate track, since standard training cannot produce a model with
   non-trivial accuracy at 8/255 whatever the sampler does.

Decisions taken: implement lookahead order selection (done: `Trainer.lookahead_segment`),
run the order-vs-initialisation experiment (sweep G), and open the
adversarial-training track with gate runs (`p5_at`).

## 2026-11-15 13:00-13:40 PST — Lookahead (null), adversarial-training gate, gradient matching (works)

Hardware: g3026-g3032, Quadro RTX 6000, 4 CPUs per job. Jobs: lookahead pilot
43273793 (`p4_lookahead`), AT gate 43273794-95 (`p5_at`), gradient-matching
feasibility 43273910 (`scripts/adhoc/gradient_matching.py` on `base_ema_s0`), smoke
run 43274171 (`gm_smoke`).

### A. Lookahead order selection: no gain

Best of 4 data orders per epoch for the first 6 epochs, scored on `probe` by
(mean robust accuracy over 0-4/255) + clean accuracy of the averaged weights
(`la_k4_e6`, seed 0): final 0.788 clean / 0.172 at 2/255 / AUC 1.13, the same as
uniform (0.797 / 0.171 / 1.14). Within one epoch the four candidates' robustness
terms differ by about 0.003; the score differences are clean-accuracy noise. The
order-driven offset the planner measured needs several epochs to develop, so a
greedy one-epoch lookahead has nothing to select. Not pursued further.

### B. Adversarial-training gate (PGD-3 on every batch, 30 epochs, one seed)

SmallCNN, train radius 4/255: clean 0.66, 0.56 at 2/255, 0.45 at 4/255, 0.23 at
8/255 at epoch 26, still improving; no decay of robustness (the small model
underfits adversarial training), as the planner predicted. ResNet-20 at 8/255:
0.41 clean / 0.28 at 8/255 at epoch 8 (running). These are the "robust model"
reference numbers for this model size.

### C. Gradient matching: clean samples can stand in for adversarial examples

**Feasibility** (`scripts/adhoc/gradient_matching.py`, averaged weights of
`base_ema_s0`). Target: gradient of the adversarial loss on held-out probe
points. Candidates: clean per-sample gradients of a random pool of training
samples. Fit: non-negative least squares, fitted against the target from one
half of the probe set and scored against the other half.

| epoch | radius | cos(uniform pool mean, target): pool 128 / 512 / 2048 | cos(fitted combination, held-out target): pool 128 / 512 / 2048 | ceiling (target half A vs half B) |
|---|---|---|---|---|
| 4 | 2/255 | 0.69 / 0.77 / 0.83 | 0.90 / 0.92 / 0.94 | 0.94 |
| 4 | 8/255 | 0.20 / 0.37 / 0.46 | 0.80 / 0.89 / 0.94 | 0.98 |
| 11 | 2/255 | 0.23 / 0.35 / 0.52 | 0.64 / 0.72 / 0.80 | 0.89 |
| 11 | 8/255 | 0.15 / 0.24 / 0.28 | 0.58 / 0.70 / 0.79 | 0.98 |
| 17 | 2/255 | 0.17 / 0.21 / 0.41 | 0.46 / 0.54 / 0.66 | 0.82 |
| 17 | 8/255 | 0.12 / 0.10 / 0.15 | 0.43 / 0.55 / 0.67 | 0.96 |

A non-negative combination of a few hundred clean-sample gradients reproduces
most of the adversarial-training direction, far beyond what a uniform batch
does, with the weight spread over roughly a quarter of the pool. So although no
individual sample is "robustness-promoting" in a stable way, at every step there
are *combinations* of clean samples whose joint update is.

**Why this differs from the rules that failed.** Keeping the top half of a pool
by an inner product rewards large gradients and whatever class the iterate
currently under-predicts; the batch's mean gradient is then not the target
direction, and the bias compounds. A least-squares fit matches the direction of
the *sum*, so it is not rewarded for size and balances the classes itself.

**Closed loop** (`gm_smoke`: SmallCNN, uniform for epoch 1, then
`GradientMatchSampler` with 512 candidates, target = adversarial gradient at
8/255 of 256 other training samples, batch 128; val, PGD-10):

| epoch | clean | @2/255 | @4/255 | @8/255 | AUC |
|---|---|---|---|---|---|
| 1.00 (switch on) | 0.340 | 0.238 | 0.161 | 0.060 | 1.42 |
| 1.33 | 0.337 | 0.285 | 0.232 | 0.146 | 1.90 |
| 1.67 | 0.359 | 0.314 | 0.264 | 0.183 | 2.15 |
| 2.00 | 0.373 | 0.315 | 0.267 | 0.174 | 2.15 |

Sampler statistics over the second epoch: cosine of the returned batch's mean
gradient with the target 0.92-0.94 (a plain batch of the same size: 0.31-0.42);
its norm 1.4-1.5x a plain batch's; effective number of distinct samples in the
batch 53-62 of 128; accuracy of the model on the selected samples 0.18-0.23
(against 0.34 overall), i.e. it mostly picks samples the model currently gets
wrong. Cost: 0.225 s per step against 0.022 s (uniform) and about 0.05 s
(PGD-3 adversarial training).

For scale: standard training never exceeds 0.09 at 8/255 and ends at 0.000;
adversarial training at 4/255 has 0.156 at 8/255 after two epochs.

A first version put the target samples among the candidates. A sample's clean
gradient is trivially close to its own adversarial gradient, so the fit could
score well by matching the target's sampling noise; the unit test caught it
(the batch looked worse than a plain batch made of the target samples). Target
samples are now always disjoint from candidates.

Still to establish: that it holds to the end of training and over seeds (runs
`p7_gm` queued), that the robustness survives stronger attacks (AutoAttack on
`gm_smoke` queued), its accuracy-robustness curve against real adversarial
training, cost reduction, other architectures and datasets.

## 2026-11-15 13:40-14:30 PST — Gradient matching over full training (v1, uncapped)

Hardware: g3026-g3037, Quadro RTX 6000, 4 CPUs per job. Jobs 43274256-65 (`p7_gm`:
one 20-epoch run each, SmallCNN, uniform for epoch 1 then gradient matching, 512
candidates + 256 target samples per step, batch 128), AT reference sweep 43273911
(`p6_at_rules/at_uniform`, 3 seeds), feasibility on the AT model 43274870, coordinate
subsets 43274281, AutoAttack on the smoke model 43274279. Figure:
`results/p7_gm/frontier_v1.png`.

### A. End-of-training results (val, PGD-10, averaged weights = raw at the end)

| run | seeds | clean | @2/255 | @8/255 | AUC |
|---|---|---|---|---|---|
| uniform, 5 epochs | 3 | 0.597 | 0.313 | 0.009 | 1.59 |
| uniform, 10 epochs | 3 | 0.702 | 0.222 | 0.000 | 1.23 |
| uniform, 20 epochs | 3 | 0.797 | 0.171 | 0.000 | 1.14 |
| gradient matching, target 2/255 | 1 | 0.731 | 0.291 | 0.001 | 1.51 |
| gradient matching, target 4/255 | 1 | 0.581 | 0.257 | 0.004 | 1.31 |
| gradient matching, target 8/255 | 3 | 0.384 +- 0.037 | 0.214 +- 0.015 | 0.022 +- 0.003 | 1.16 +- 0.05 |
| real adversarial training at 8/255 (PGD-3) | 3 | 0.510 +- 0.004 | 0.458 +- 0.005 | 0.295 +- 0.001 | 3.22 +- 0.03 |

* **Target 2/255 is the first data-only rule above the uniform curves at the end
  of training**: 0.291 at 2/255 with 0.731 clean, where the uniform end points
  interpolate to about 0.206 and the uniform run's averaged weights mid-run to
  about 0.26. Mid-run it is further above (epoch 10: 0.655 clean / 0.387; uniform
  at that accuracy: about 0.32).
* **Larger targets do better early and then collapse.** Target 8/255 reaches
  0.17-0.19 at 8/255 by epoch 2 and holds it to about epoch 5 with clean accuracy
  ~0.40, then loses it while clean accuracy does not improve; target 4/255 the
  same, later. All three seeds of the 8/255 run show it.

### B. Why it collapses (from the step log, `scripts/analyze_selection.py`; 8/255 run, seed 0)

| epoch | cos(batch, target) | cos(uniform batch, target) | effective distinct samples per batch of 128 | distinct samples used in the epoch | share of updates going to the most-used 10% of data |
|---|---|---|---|---|---|
| 2 | 0.93 | 0.38 | 60 | 16,061 | 0.59 |
| 5 | 0.83 | 0.05 | 42 | 12,290 | 0.70 |
| 8 | 0.67 | -0.29 | 27 | 9,003 | 0.83 |
| 12 | 0.53 | -0.44 | 21 | 7,226 | 0.90 |
| 20 | 0.31 | -0.45 | 18 | 4,832 | 0.99 |

The unconstrained fit puts its weight on fewer and fewer samples, and the same
ones every time they are drawn (cats twice their share by the end). The rest of
the data stops being trained on, the model drifts to states where the clean
gradient of a random batch points *against* the adversarial gradient, and the
fit can no longer express the target. With the 2/255 target the same trend is
much slower (15,410 distinct samples and cosine 0.57 in epoch 20).

### C. It is the selection that drifts, not the idea that runs out

The same fit on a real adversarially-trained model (`p5_at/cnn_at8`, averaged
weights, 512 candidates, target at 8/255 on held-out probe points): cosine of the
fitted combination with the held-out target 0.86 / 0.89 / 0.87 at epochs 4 / 11 /
20 (uniform pool mean 0.69 / 0.51 / 0.31), effective number of samples 139 / 90 /
81. Along a genuine adversarial-training trajectory the adversarial gradient stays
expressible by clean samples to the end. So a version that stays near that
trajectory should not hit the wall; the obvious constraint is on how concentrated
the weights may be (next entry).

### D. Robustness is real under stronger attacks (so far)

`gm_smoke` (2 epochs), val, 3000 points: PGD-10 gives 0.315 / 0.267 / 0.174 at
2 / 4 / 8 per 255; worst case over 3 restarts x (cross-entropy, margin) PGD-50
gives 0.299 / 0.244 / 0.134; standard AutoAttack 0.299 / 0.242 / 0.126. No sign
of gradient masking, but PGD-10 overstates accuracy at 8/255 by about a third,
so headline numbers must come from the strong attack.

### E. The fit does not need whole gradients

Fitting on a random subset of gradient coordinates and scoring the combination in
the full space against a held-out target (512 candidates, 8/255, epoch 4 / 11 of
`base_ema_s0`): all 279,466 coordinates 0.88 / 0.71; 32,768 random coordinates
0.88 / 0.69; 8,192: 0.85 / 0.66; 2,048: 0.83 / 0.62; final layer only (1,290):
0.59 / 0.28; uniform batch 0.41 / 0.20. A few thousand random coordinates per
sample are enough, the last layer is not. `GradientMatchSampler(n_coords=...)`
implements this for models too large to hold full per-sample gradients.

## 2026-11-15 14:30-16:40 PST — Gradient matching: seeds, variants, controls, fair frontier, generalisation, test

Hardware: g3026-g3037, Quadro RTX 6000, 4 CPUs per job (analysis on cpu-g2). Run
roots: `p7_gm` (gradient-matching variants), `p8_general` (ResNet-20, CIFAR-100,
BatchNorm), `p9_ref` (uniform references with per-epoch checkpoints: default,
lr 0.025 / 0.1, weight decay 0 / 2e-3, batch 64; adversarial training at 2/255).
Tables: `results/final/`, `results/p7_gm/`, `results/p8_general/`.

Provenance note: from 14:55 every run records the code revision in `meta.json`
(`start.git`). Runs started before that do not. The sampler was edited while
jobs were queued (capped fit 13:58, coordinate subsets 14:05, adaptive radius
14:38, uniform fraction 14:50); all edits kept the default path
(`max_repeat=0`, single `eps`, `uniform_frac=0`) unchanged, and the unit tests
passed after each (22, 25, 26, 27, 28 tests), but the exact file a given
`p7_gm`/`p8_general` run imported is only known from its start time in
`meta.json`.

### A. Variants (val, end of training, 20 epochs; averaged = raw at the end)

| variant | seeds | clean | 2/255 | AUC | note |
|---|---|---|---|---|---|
| target 2/255 (default) | 3 | 0.730 +- 0.009 | 0.289 +- 0.002 | 1.50 +- 0.01 | very small spread over seeds |
| target 1/255 | 1 | 0.766 | 0.291 | 1.52 | higher accuracy, same robustness |
| target 2/255, 1024 candidates | 1 | 0.725 | 0.288 | 1.52 | a larger pool does not help |
| target 2/255, fit on 8,192 gradient coordinates | 1 | 0.729 | 0.276 | 1.45 | works in closed loop, slightly weaker |
| target 2/255, no repeats in a batch | 1 | 0.739 | 0.286 | 1.48 | no better |
| target 4/255 | 1 | 0.581 | 0.257 | 1.31 | late collapse |
| target 4/255, no repeats | 1 | 0.672 | 0.203 | 1.16 | cap weakens the match |
| target 4/255, half of each batch uniform | 1 | 0.761 | 0.252 | 1.37 | no collapse |
| target 8/255 | 3 | 0.385 +- 0.045 | 0.214 +- 0.018 | 1.16 | late collapse |
| target 8/255: 256 / 1024 candidates, probe target, from step 0, mixed target | 1 each | 0.35-0.57 | 0.21-0.25 | 1.11-1.33 | all collapse |
| target 8/255, no repeats | 1 | 0.641 | 0.159 | 0.99 | cap weakens the match |
| target 8/255, half of each batch uniform | 1 | 0.749 | 0.236 | 1.32 | no collapse, robustness at 8/255 not kept |
| **control: same sampler, clean target** | 2 | on/below the uniform curve (table B) | | | the gain needs the adversarial direction |

Nothing tried keeps robustness at 8/255 to the end of training. Stopped looking;
it is reported as a limit.

### B. The fair comparison (val, PGD-10, averaged weights, 3 seeds unless noted)

Best accuracy at 2/255 among operating points (variant x epoch, seed means) with
clean accuracy at least the level (`scripts/matched_frontier.py`,
`results/final/frontier_val_adv2.csv`):

| clean at least | uniform, epochs only | uniform tuned (lr, weight decay, batch) | uniform tuned + weight interpolation | gradient matching 2/255 | clean-target control (2 seeds) | adversarial training 2/255 |
|---|---|---|---|---|---|---|
| 0.55 | 0.342 | 0.348 | 0.348 | **0.402** | 0.338 | 0.541 |
| 0.60 | 0.325 | 0.338 | 0.338 | **0.402** | 0.323 | 0.541 |
| 0.65 | 0.308 | 0.308 | 0.317 | **0.384** | 0.286 | 0.541 |
| 0.70 | 0.260 | 0.263 | 0.297 | **0.351** | 0.245 | 0.540 |
| 0.75 | 0.229 | 0.236 | 0.245 | (not reached with >= 2 seeds) | 0.203 | - |

* Gradient matching is 0.054-0.067 above the best uniform recipe including
  learning-rate, weight-decay and batch-size variants and interpolation of an
  early with the final averaged model. Seed sd at the selected points: gradient
  matching 0.002-0.007, uniform 0.006-0.028.
* Tuning the uniform recipe moves its frontier by at most 0.013; interpolation
  adds up to 0.034 at clean 0.70. The planner's interpolation finding
  reproduces on three seeds.
* The identical sampler with a clean target lands on or below the uniform curve:
  the effect is the adversarial direction, not the concentration of the fit or
  its longer steps.
* Real adversarial training at 2/255 is 0.14-0.19 above gradient matching at
  matched accuracy: choosing clean samples recovers roughly a third of what
  training on attacked inputs gives.

### C. Test split, strong attack (worst case over 3 restarts x 2 losses of PGD-50), averaged weights, 3 seeds

Operating points were selected on val (table B) and evaluated once on test.

| model | clean | 1/255 | 2/255 | 4/255 |
|---|---|---|---|---|
| gradient matching 2/255, epoch 6 | 0.609 +- 0.009 | 0.489 +- 0.005 | **0.375 +- 0.007** | 0.195 +- 0.005 |
| uniform lr 0.025, epoch 5 (best uniform point at clean >= 0.60) | 0.610 +- 0.008 | 0.457 +- 0.008 | 0.314 +- 0.007 | 0.123 +- 0.010 |
| gradient matching 2/255, epoch 9 | 0.656 +- 0.007 | 0.501 +- 0.004 | **0.354 +- 0.006** | 0.138 +- 0.010 |
| uniform, epoch 6 | 0.653 +- 0.006 | 0.455 +- 0.021 | 0.280 +- 0.028 | 0.076 +- 0.023 |
| gradient matching 2/255, epoch 13 | 0.703 +- 0.009 | 0.501 +- 0.001 | **0.320 +- 0.002** | 0.094 +- 0.005 |

The remaining reference points (uniform lr 0.025 at epoch 10, the two
interpolated models, final checkpoints, AutoAttack) are in the next entry.

### D. Other architecture, other dataset (val, PGD-10, averaged weights, 2 seeds, uniform varied by epochs only)

| setting | clean at least | uniform | gradient matching 2/255 |
|---|---|---|---|
| ResNet-20 (GroupNorm), CIFAR-10 | 0.60 | 0.263 | 0.384 |
| | 0.70 | 0.178 | 0.374 |
| | 0.75 | 0.169 | 0.335 |
| SmallCNN, CIFAR-100 | 0.30 | 0.145 | 0.170 |
| | 0.35 | 0.136 | 0.161 |

End of training: ResNet-20 uniform 0.856 clean / 0.080 at 2/255; gradient
matching 0.745 / 0.301 (seed 0). CIFAR-100 uniform 0.422 / 0.091; gradient
matching 0.369 / 0.153. A ResNet-20 gradient-matching run takes 81 minutes
(per-sample gradients of 512 candidates through 20 layers).

### E. Reference: adversarial training at 2/255 (PGD-3), SmallCNN, 3 seeds

0.699 +- 0.005 clean, 0.541 +- 0.007 at 2/255, about 0.37 at 4/255, 0.12 at 8/255.

### F. Bug found: weight average and BatchNorm

The averaged model copied parameters only, so a BatchNorm network's average
kept its initial running statistics and scored chance (first BatchNorm pair,
`p8_general/bn_*`: averaged clean 0.097-0.100 while raw was 0.82-0.85). Fixed
(`update_ema` now averages floating-point buffers; unit test added) and the
pair is being rerun as `bn2_*`. Raw weights of the first pair: uniform 0.853 /
0.094, gradient matching 0.816 / 0.125 at the end of training.

## 2026-11-15 16:40-17:15 PST — Extra seeds, remaining test references, AutoAttack, BatchNorm rerun, archive

Hardware: g3026-g3037, Quadro RTX 6000, 4 CPUs per job (analysis and archiving
on cpu-g2, 1 CPU). All runs here record their code revision in `meta.json`.

**Seeds 3 and 4** for gradient matching 2/255, uniform default and uniform
lr 0.025; seeds 1 and 2 for gradient matching 1/255. Five-seed validation
frontier (`results/final/frontier5_val_*.csv`, accuracy at 2/255, PGD-10,
averaged weights): clean >= 0.60: gradient matching 0.403 (sd 0.005) against
uniform 0.331, with all tuning and interpolation 0.334; >= 0.65: 0.389 against
0.317; >= 0.70: 0.340 (1/255 target: 0.343) against 0.297; >= 0.75: 1/255
target 0.297 against 0.245. End of training, 5 seeds: 0.726 +- 0.009 clean,
0.290 +- 0.004 at 2/255. Same ordering at 4/255 and for the area under the
robust-accuracy curve. Figure: `results/final/frontier.png`.

**Test split, strong attack, remaining references** (3 seeds unless noted):
uniform lr 0.025 epoch 10: 0.710 clean / 0.242 at 2/255; uniform epoch-4 average
interpolated half-way to the final weights: 0.653 / 0.295; uniform epoch-3
average interpolated 0.75 of the way (5 seeds, the two new ones included):
0.705 / 0.276. Against gradient matching at epochs 9 and 13: 0.656 / 0.354 and
0.703 / 0.320. End-of-schedule models (`results/final/test_strong.csv`):
uniform 20 epochs 0.795 / 0.146; gradient matching 2/255 0.728 / 0.261;
adversarial training 8/255 0.519 / 0.449 and 0.269 at 8/255.

**AutoAttack** (standard, first 2,000 test points, seed 0;
`results/final/test_autoattack_*.csv`), accuracy at 2/255 and 4/255, with the
PGD-50 worst case on the same points in brackets: gradient matching epoch 9
averaged 0.359 (0.365), 0.138 (0.158); uniform epoch 6 averaged 0.310 (0.319),
0.097 (0.110); gradient matching end 0.255 (0.267); uniform 20 epochs end 0.149
(0.166); adversarial training 2/255 end 0.525 (0.529), 0.343 (0.356). No sign of
gradient masking. The first test job (`finalA`, 43275689) also ran AutoAttack at
2 and 8/255 on end-of-schedule models and agrees where they overlap; it was
left to run into its 2 h limit.

**BatchNorm, rerun with the fixed weight average** (`p8_general/bn2_*`, 2
seeds, validation): accuracy at 2/255 at clean >= 0.60 / 0.70 / 0.75 / 0.80:
uniform 0.348 / 0.297 / 0.246 / 0.224; gradient matching 0.417 / 0.362 / 0.353
/ 0.231. End of training: uniform 0.858 / 0.088; gradient matching 0.803 /
0.134. The gain survives eval-mode per-sample gradients; it is gone at 0.80.

**Mistake caught**: `interp_eval.py` takes the run list before list-valued
options (`--alphas 0.25 0.5 0.75 RUN` swallowed the run); first submission
failed at argument parsing, usage text corrected.

**Archive**: `scripts/archive_models.sh` wrote ten tar files (418 MB, one per
family because the group fileset is at 95% of its file-count limit) to
`/gscratch/amath/sgvtc/ordering-ar/models/`: configs, metadata, validation
curves, test evaluations and the checkpoints at the selected epochs for
gradient matching (2/255, 1/255), the uniform references, adversarial training
(2/255, 8/255), ResNet-20 and CIFAR-100.
