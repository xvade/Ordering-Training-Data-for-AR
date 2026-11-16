# TODO

Goal: characterise which model updates degrade adversarial robustness (AR), and
turn that into a system that orders / selects training samples to increase AR
without giving up clean accuracy. Small models on CIFAR-10 first; the method
must transfer to other models and datasets and will be assessed by someone else
for consistency and quality.

Where things stand is summarised in `docs/FINDINGS.md`; every experiment is in
`JOURNAL.md`.

## Open decision (the user's)

- [ ] Is training on attacked inputs in scope? Gradient matching trains on clean
      samples only; at clean accuracy 0.70 it closes under a fifth of the gap
      between uniform training and adversarial training at the same radius, at
      3x the cost of adversarial training. If attacked inputs are allowed, the
      ordering question worth asking is the one in FINDINGS section 6 (which
      samples to attack and train on), where class-balanced batches and keeping
      the low-adversarial-margin samples already help.

## Next

- [ ] Reach clean accuracy above 0.75 with the gain intact (target 1/255 gets to
      0.75; longer schedules, or switching the target radius down late).
- [ ] The late collapse at large target radii, and with it robustness at 8/255.
      Tried without success: pool size, mixed target, held-out target, repeat
      caps, half-uniform batches. Untried: a usage cap per epoch, a penalty on
      the batch-gradient component orthogonal to the target.
- [ ] Cost (8x a plain step). Coordinate subsets work (`n_coords`); untried:
      refreshing the fit every k steps, smaller target sets.
- [ ] Tuned uniform references (learning rate, interpolation) for ResNet-20 and
      CIFAR-100; a deeper BatchNorm network; more than 2 seeds there.
- [ ] AutoAttack on all seeds and the full test set for the headline pair.
- [ ] Order of the early window: sweep G (`p3_offset`) has the data, not yet
      analysed properly.
- [ ] Larger models under adversarial training (ResNet-18), where robust
      overfitting appears and "which updates destroy robustness" has a second
      meaning.

## Done

- [x] 2026-11-15 Klone notes (`docs/KLONE.md`), environment, CIFAR-10 / CIFAR-100.
- [x] 2026-11-15 Core library + unit tests: data, models, PGD, probe evaluator,
      steppable trainer, samplers, strong evaluation.
- [x] 2026-11-15 Phase 1: instrumented baseline, one-step replay study at raw
      and averaged weights, characterisation (JOURNAL.md; planner's analysis in
      `docs/plans/2026-11-15_phase2_plan.md`).
- [x] 2026-11-15 Tested and ruled out: difficulty-based selection (easy, hard,
      band), class-balanced batches, top-k by held-out loss gradients, top-k by
      linearised-radius gain, one-epoch lookahead over data orders.
- [x] 2026-11-15 Gradient matching: feasibility, sampler, closed-loop runs,
      variants, collapse diagnosis, clean-target control.
- [x] 2026-11-15 Fair comparison at matched clean accuracy against tuned uniform
      references with weight averaging and interpolation (5 seeds for the
      headline pair); adversarial-training references at 2/255 and 8/255;
      selection rules under adversarial training.
- [x] 2026-11-15 Generalisation: ResNet-20, CIFAR-100, BatchNorm (after fixing
      the weight average for BatchNorm statistics).
- [x] 2026-11-15 Test-split confirmation with the strong attack and AutoAttack;
      headline checkpoints archived to `/gscratch/amath/sgvtc/ordering-ar/models/`.
