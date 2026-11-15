# TODO

Goal: characterise which model updates degrade adversarial robustness (AR), and
turn that into a system that orders / selects training samples to increase AR
without giving up clean accuracy. Small models on CIFAR-10 first; the method
must transfer to other models and datasets and will be assessed by someone else
for consistency and quality.

## Now

- [ ] Phase 1a: instrumented baseline run (SmallCNN, GroupNorm, 20 epochs) with
      per-step probe robustness and checkpoints.
- [ ] Phase 1b: counterfactual single-update study at ~10 checkpoints
      (`arorder.investigate`): per-sample first-order gains + real finite steps.
- [ ] Phase 1c: characterise the harmful updates. Which sample properties
      predict "helps clean, hurts robust"? How does it change over training?
      Hand the summary tables to a max-effort Fable planning agent (must run
      inside a Slurm allocation, see docs/KLONE.md section 8).

## Next

- [ ] Phase 2: turn the characterisation into a sampler (`arorder.samplers`),
      with a random-order baseline at equal update budget, >= 5 seeds, spread
      reported.
- [ ] Check the sampler on BatchNorm and momentum-free variants, a second
      architecture (ResNet-20) and a second dataset (CIFAR-100 or SVHN).
- [ ] Final evaluation on the untouched test set with an attack stronger than
      the one used for selection (AutoAttack or many-restart margin PGD), to
      rule out gradient masking.
- [ ] Decide and document whether the sampler is also evaluated on top of
      adversarial training (PGD-AT / TRADES), not only standard training.
- [ ] Copy anything an assessor needs out of scrubbed storage
      (`/gscratch/amath/sgvtc/ordering-ar/`) and into the repo (`results/`).

## Done

- [x] 2026-11-15 Klone notes (`docs/KLONE.md`), environment, CIFAR-10 download.
- [x] 2026-11-15 Core library + unit tests: data, models, PGD, probe evaluator,
      instrumented trainer, counterfactual investigation.
