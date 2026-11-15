# TODO

Goal: characterise which model updates degrade adversarial robustness (AR), and
turn that into a system that orders / selects training samples to increase AR
without giving up clean accuracy. Small models on CIFAR-10 first; the method
must transfer to other models and datasets and will be assessed by someone else
for consistency and quality.

Where things stand is summarised in `docs/FINDINGS.md`; every experiment is in
`JOURNAL.md`.

## Now

- [ ] Finish the gradient-matching consistency runs (`p7_gm`: target 2/255, seeds
      0-2; capped, larger-pool and half-uniform variants) and pick the default
      configuration from them.
- [ ] Generalisation runs (`p8_general`): ResNet-20 and CIFAR-100, uniform
      against gradient matching, 2 seeds each.
- [ ] Strong-attack evaluation on the untouched **test** split
      (`scripts/final_eval.py`, PGD-50 worst case + AutoAttack) of the headline
      models: uniform, gradient matching, real adversarial training.
- [ ] Write the numbers into `docs/FINDINGS.md` and `README.md`; copy headline
      checkpoints to `/gscratch/amath/sgvtc/ordering-ar/models/`.

## Next

- [ ] Stop the late collapse of gradient matching at large target radii. Open
      leads: half-uniform batches (queued), a usage cap per epoch, an explicit
      penalty on the part of the batch gradient orthogonal to the target.
- [ ] 10 seeds for the headline comparison (3 seeds detect about 0.055 at 2/255
      at 80% power; see the planner's power analysis).
- [ ] BatchNorm models: per-sample gradients are taken in eval mode; check that
      the match and the gain survive.
- [ ] Cost: the sampler is ~8x a plain step on the small CNN. Options measured
      or sketched: coordinate subsets (`n_coords`, measured to work), refreshing
      the fit every k steps, smaller target sets.
- [ ] Fair-frontier baseline from the planner (averaging + early stop +
      interpolation of early and late averaged weights) with 3+ seeds.
- [ ] Order of the early window: the planner's pilot says data order alone moves
      a run's offset from the trade-off curve; sweep G (`p3_offset`) has the
      data, not yet analysed properly.
- [ ] Larger models under adversarial training (ResNet-18), where robust
      overfitting appears and "which updates destroy robustness" has a second
      meaning.

## Done

- [x] 2026-11-15 Klone notes (`docs/KLONE.md`), environment, CIFAR-10 / CIFAR-100.
- [x] 2026-11-15 Core library + unit tests (27): data, models, PGD, probe
      evaluator, steppable trainer, samplers, strong evaluation.
- [x] 2026-11-15 Phase 1: instrumented baseline, one-step replay study at raw
      and averaged weights, characterisation (JOURNAL.md; planner's analysis in
      `docs/plans/2026-11-15_phase2_plan.md`).
- [x] 2026-11-15 Tested and ruled out: difficulty-based selection (easy, hard,
      band), class-balanced batches, top-k by held-out loss gradients, top-k by
      linearised-radius gain, one-epoch lookahead over data orders.
- [x] 2026-11-15 Gradient matching: feasibility, sampler, closed-loop runs,
      collapse diagnosis, AutoAttack sanity check.
- [x] 2026-11-15 Adversarial-training reference runs and selection rules under
      adversarial training.
