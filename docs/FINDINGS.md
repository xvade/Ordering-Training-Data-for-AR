# Findings: what erodes adversarial robustness during training, and what data ordering can do about it

Status: 2026-11-15. Everything here is from small models (a 0.28 M-parameter CNN
unless stated) on CIFAR-10, validation split, L-infinity attacks with radii in
units of 1/255. Numbers marked **[strong]** come from the strong evaluation
(worst case over restarts of PGD-50 with two objectives, and AutoAttack); the
rest use PGD-10 with fixed starts, which overstates accuracy at 8/255 by about a
third. Run-by-run detail, hardware and job numbers are in `JOURNAL.md`; the
planning agent's independent analysis is in
`docs/plans/2026-11-15_phase2_plan.md`.

## Summary

1. Under ordinary training, robustness at small radii rises for a few epochs and
   then decays for the rest of training while clean accuracy keeps rising. The
   decay is a slow drift of the solution; it is not caused by identifiable bad
   updates or bad samples.
2. The cleanest description of the drift: **the direction that lowers clean loss
   and the direction that lowers adversarial loss are the same at the start of
   training and separate steadily** (cosine 0.98 after epoch 1, 0.29 at the end).
   Training follows the first. In input space, margins grow ~3x while the
   margins' input gradients grow ~8x, for every sample, including those that
   were always classified correctly.
3. Rules that pick samples by a per-sample score (easy, hard, near the boundary,
   class-balanced, first-order effect on held-out robustness) do not help: they
   move the model along the same accuracy-robustness curve that training longer
   or shorter traces, or they break training.
4. What does help is choosing, at every step, a *combination* of clean samples
   whose joint gradient points along the adversarial-training direction
   ("gradient matching"). It is the only data-only rule we found that ends
   training above the uniform-sampling curves, and it does so consistently over
   seeds. Its robustness holds under AutoAttack.
5. Its limits are real: with a large target radius it gains robustness at 8/255
   early and then loses it; data ordering alone does not produce a model that is
   robust at 8/255 the way adversarial training does.

## 1. The phenomenon

Baseline: uniform shuffling, SGD with momentum, 20 epochs, cosine schedule.

| epoch | clean | 1/255 | 2/255 | 4/255 | boundary distance (linearised, /255) |
|---|---|---|---|---|---|
| 1 | 0.313 | 0.263 | 0.220 | 0.153 | 1.54 |
| 2 | 0.387 | 0.327 | 0.273 | 0.171 | 2.01 |
| 5 | 0.580 | 0.446 | 0.315 | 0.138 | 1.88 |
| 8 | 0.680 | 0.452 | 0.257 | 0.057 | 1.33 |
| 14 | 0.762 | 0.456 | 0.197 | 0.022 | 1.20 |
| 20 | 0.799 | 0.456 | 0.172 | 0.013 | 1.14 |

Robustness peaks at epochs 2-5 depending on the radius, then decays. End points
of uniform runs of different lengths trace the trade-off: 5 epochs 0.597 clean /
0.313 at 2/255; 10 epochs 0.702 / 0.222; 20 epochs 0.797 / 0.171; 40 epochs
0.841 / 0.106 (3 seeds each).

## 2. What the decay is

**A slow drift under large symmetric noise.** With robustness measured before
every one of 7,020 updates: the adversarial-minus-clean loss gap rises by about
4e-4 per update while single updates move it by +-0.04. Half of all updates
raise it and half lower it, in every epoch. No small set of updates carries the
decay.

**Not an artefact of the iterate's oscillation.** An exponential moving average
of the weights is both more accurate and more robust than the raw iterate
mid-training (epoch 6: 0.653 clean / 0.320 at 2/255 against 0.558 / 0.254) and
shows the same rise and decay (peak 0.349 at epoch 4, 0.176 at the end).

**A separation of two directions.** At the averaged weights, with gradients of
the held-out loss (adversarial at 2/255):

| epoch | cos(clean gradient, adversarial gradient) | adversarial / clean gradient norm | samples that lower clean loss but raise adversarial loss |
|---|---|---|---|
| 1 | 0.975 | 1.2 | 1.6% |
| 4 | 0.879 | 1.7 | 4.7% |
| 8 | 0.721 | 2.7 | 6.6% |
| 14 | 0.520 | 4.0 | 5.9% |
| 20 | 0.286 | 4.8 | 12.0% |

**A global loss of input-space margin** (planning agent, from the per-sample
data). Between epochs 2 and 14 the margin of correctly classified training
points grows 2.8x while its input-gradient norm grows 8x. The 6,195 samples that
are correct at every checkpoint lose robustness like the rest (accuracy at 2/255
0.80 to 0.49). Samples that become correct late are non-robust on arrival.

**Class-structured.** 94% of the 2/255 accuracy lost between epochs 4 and 14 is
lost in the seven easier classes while bird, cat and deer are being learned.

## 3. Why per-sample attribution does not find culprits

We replayed single updates from saved checkpoints (pure gradient step on a chosen
batch, robustness re-measured with identical attack starts, effect read on
held-out data that played no part in choosing the batch) and computed, for all
45,000 training samples, the first-order effect of training on that sample.

* The measurement itself is sound: noise from the attack is ~100x below the
  spread between batches, first-order predictions match real steps (R^2 0.95-0.99
  at a quarter of the learning rate), and scores from one held-out set predict
  effects on another.
* But a sample's effect on held-out adversarial loss and on held-out clean loss
  are nearly the same ranking early on (Spearman 0.97-0.999), and what a single
  step from a training iterate measures is mostly transient: 20-75% of the
  variance of per-sample effects is the class label, and which classes help
  flips between checkpoints, because the iterate over-predicts different classes
  at different moments.
* The "robustness-only" part of a sample's effect is not predicted by any
  property of the sample (loss, margin, gradient norm, boundary distance, own
  adversarial margin: |Spearman| < 0.25).
* The rule that looked best in one-step replay, selecting by first-order benefit
  to the held-out boundary distance, stalled training when applied at every step
  (0.44 clean after 20 epochs). A bias that is harmless in one step compounds
  over thousands, with momentum on top.

Two measurement pitfalls cost time and are worth knowing: a pure step of the full
learning rate from a training iterate is already outside the linear regime, and
batches selected by any inner-product score have gradients 1.4-2.7x longer (or,
for easy samples, 2-8x shorter) than a random batch, so naive comparisons measure
step length.

## 4. Selection rules that only move along the curve (or break training)

All at equal update budget, 3 seeds, 20 epochs unless stated.

| rule | clean | 2/255 | verdict |
|---|---|---|---|
| keep easiest half of each pool (by loss, boundary distance, or own adversarial margin) | 0.52-0.56 | 0.25-0.26 | worse than uniform for 5 epochs (0.597 / 0.313) |
| keep hardest half | 0.815-0.822 | 0.100-0.112 | matched by uniform for 40 epochs (0.841 / 0.106) |
| class-balanced batches | 0.809 | 0.150 | on the uniform curve, slightly further along |
| from epoch 7: drop the hardest quarter | 0.740 | 0.228 | above annealed uniform end points (~0.20), not above uniform averaged weights mid-run (~0.24) |
| keep best half by held-out "robustness-only" loss gradient | 0.475 | 0.283 | collapsed to chance for an epoch, on the curve after recovery |
| keep best half by held-out adversarial / clean loss gradient | 0.30-0.32 | 0.17-0.20 | training stalls |
| keep best half by boundary-distance gain | 0.436 | 0.231 | training stalls |
| best of 4 data orders per epoch, first 6 epochs (lookahead) | 0.788 | 0.172 | equal to uniform (0.797 / 0.171) |

Weight-space operations do move off the curve (averaged weights mid-run;
interpolating an early averaged model with the final one), which sets a fair
reference for any sampler: the uniform run's *averaged-weight path*, not just
its end point.

## 5. Gradient matching

### Idea

Adversarial training updates along `t`, the loss gradient at attacked inputs. A
clean batch's gradient is increasingly unrelated to `t` as training proceeds
(section 2). But the clean per-sample gradients `g_i` of a few hundred candidate
samples span a cone that contains a good approximation of `t`. So at each step:
draw candidates, measure `t` on *other* samples, solve
`min_{w >= 0} || sum_i w_i g_i - t ||`, and train on a batch in which sample `i`
appears in proportion to `w_i`. Every update is an ordinary step on clean
training samples; only the choice of samples used the attack.

Implemented as `arorder.samplers.GradientMatchSampler` (`sampler="grad_match"`).

### Feasibility (fit quality measured against a held-out half of the target)

Cosine with the adversarial gradient at 8/255, 512 candidates: uniform pool mean
0.37 / 0.24 / 0.10 at epochs 4 / 11 / 17 of a standard run; fitted combination
0.89 / 0.70 / 0.55. On an adversarially trained model the fit stays at 0.86-0.89
to the end. About a quarter of the candidates carry the weight. A random 3% of
the gradient coordinates is enough for the fit (0.85 against 0.88 with all of
them); the last layer alone is not (0.59).

### Results

End of training, 20 epochs, uniform for the first epoch then gradient matching
with 512 candidates and 256 target samples per step:

| run | seeds | clean | 2/255 | 4/255 | 8/255 |
|---|---|---|---|---|---|
| uniform | 3 | 0.797 | 0.171 | 0.012 | 0.000 |
| uniform, 10 epochs | 3 | 0.702 | 0.222 | 0.035 | 0.000 |
| **gradient matching, target 2/255** | see `JOURNAL.md` | 0.72-0.73 | 0.29 | 0.07 | 0.00 |
| gradient matching, target 4/255 | 1 | 0.581 | 0.257 | 0.075 | 0.004 |
| gradient matching, target 8/255 | 3 | 0.384 | 0.214 | 0.102 | 0.022 |
| real adversarial training at 8/255 | 3 | 0.510 | 0.458 | 0.400 | 0.295 |

* With the 2/255 target the whole training path lies above both uniform
  reference curves (`results/p7_gm/frontier_v1.png`). At the end: 0.29 at 2/255
  with 0.72-0.73 clean, where uniform end points interpolate to about 0.21 and
  uniform averaged weights mid-run to about 0.26. Mid-run the margin is larger
  (epoch 10: 0.66 clean / 0.38, uniform at that accuracy about 0.32).
* Larger target radii buy robustness at larger radii early (target 8/255: 0.17-0.19
  at 8/255 after one epoch of matching, **[strong]** 0.126 under AutoAttack) and
  lose it after about epoch 5.
* Robustness survives strong attacks: for the 2-epoch model, PGD-10 0.315 /
  0.267 / 0.174 at 2 / 4 / 8; PGD-50 worst case 0.299 / 0.244 / 0.134; AutoAttack
  0.299 / 0.242 / 0.126.

### Why large radii collapse

The unconstrained fit concentrates: in the 8/255 run a batch of 128 holds 60
effective distinct samples in epoch 2 and 18 in epoch 20, and by then 10% of the
training set receives 99% of the updates. The rest of the data stops being
trained on, the clean gradient of a random batch ends up pointing *against* the
adversarial gradient (cosine -0.45), and the match falls from 0.93 to 0.31.
Things tried against it, none sufficient: a larger pool, mixing the clean
gradient into the target, drawing the target from held-out data, capping repeats
within a batch (weakens the match and with it the effect). With the 2/255 target
the same drift is slow enough that the run ends well.

### Cost

About 8x a plain step on this model (per-sample gradients of 512 candidates, an
attack on 256 target samples, a 512-variable least-squares fit), against about
2.3x for PGD-3 adversarial training. It is a way of getting robustness out of
the *choice* of clean samples, not a cheaper adversarial training.

## 6. Adversarial training as the reference, and ordering under it

A small model underfits adversarial training: robustness rises throughout and
there is no decay to protect against. There, rules that speed learning up help
both accuracy and robustness: class-balanced batches 0.539 clean / 0.483 at
2/255 and keeping the 75% of each pool with the lowest adversarial margin 0.547
/ 0.482, against uniform 0.510 / 0.458 (AT at 8/255, 20 epochs, 3 seeds; uniform
for 40 epochs: 0.566 / 0.510). Easy-sample rules hurt.

## 7. What is not established

* Three seeds at most per configuration; 10 are needed for small differences.
* One architecture family and one dataset so far for the headline result
  (ResNet-20 and CIFAR-100 runs are in `JOURNAL.md` as they land).
* BatchNorm models (per-sample gradients are taken in eval mode).
* Whether the late collapse can be removed rather than avoided by a small
  target radius.
