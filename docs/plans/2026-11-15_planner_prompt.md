# Brief for the planning agent (2026-11-15)

You are a research planner for this repository. You are running on a klone
**compute node inside a Slurm allocation with 4 CPUs, 24 GB RAM and no GPU**.
Stay inside that: at most 4 threads, no job submission (`sbatch`/`srun`/`salloc`),
no GPU work, no large downloads. Do not edit code or existing files in the
repository. You may write scratch files under `$TMPDIR` and you must write your
deliverable to `docs/plans/2026-11-15_phase2_plan.md`.

For analysis use the project interpreter: `source scripts/env.sh` then `$PY`
(Python 3.11 with numpy, pandas, scipy, scikit-learn, matplotlib, torch-CPU
usable). Set `OMP_NUM_THREADS=4`.

## The goal (from the project owner)

Characterise which model updates cause a decay in adversarial robustness (AR),
and design a system that chooses the order in which samples are trained on so as
to increase AR. Never training on some samples is allowed, but raw accuracy also
matters. Small CIFAR-10 models first, to iterate quickly; absolute accuracy is
not the point, the training technique is. The technique and the models it
produces will be assessed by someone else, possibly on different models or
datasets, for (a) how consistently it produces adversarially robust models and
(b) the overall quality of the methods.

## What exists

Read these first, in this order:

1. `README.md` (layout and conventions), `TODO.md`.
2. `JOURNAL.md` — every experiment so far with numbers. This is the main input.
3. `results/phase1/trajectory_epochs.csv`, `results/phase1/trajectory.png` —
   per-update probe log of the instrumented baseline.
4. `results/phase1/investigate/summary.md` and the CSVs beside it — the
   single-update replay study at *raw* iterates (epochs 0.33-6 so far; later
   checkpoints may have landed by the time you read this).
5. `results/p2a/summary.csv`, `curves.csv` — first full-training sweeps.

Raw data (under `/gscratch/scrubbed/sgvtc/ordering-ar-runs/`, read-only for you):

* `base_gn_s0/steps.npz` — per-update log (batch indices, probe metrics before
  every update); `base_gn_s0/investigate/step_*.npz` — replay study at raw iterates.
* `base_ema_s0/evals.json` — baseline with weight averaging;
  `base_ema_s0/investigate/step_*.npz` — the same replay study at the
  **weight-averaged** checkpoints, with scale-free robustness targets
  (smoothed robust error, linearised-radius gradient). **These files are being
  written while you work (9 checkpoints, about 18 minutes each on 3 GPUs,
  started 12:13 PST). Use whichever exist; they are the most informative data.**
* `p2a/<variant>/seed*/evals.json` — sweeps.

The `.npz` schema is documented in the docstring of
`arorder/investigate.py::investigate_checkpoint`; the summary script is
`scripts/analyze_investigation.py` (you can run it on a directory of `.npz`
files, writing to a scratch directory, or load the files yourself for anything
it does not compute).

## What has been established so far (details in JOURNAL.md)

* Under standard training, robustness at small radii rises, peaks around epoch
  3-5, then decays steadily while clean accuracy keeps rising. The decay is a
  slow drift, about 100x smaller per update than the update-to-update noise; no
  small set of culprit updates exists.
* At raw iterates, a sample's one-step effect on held-out adversarial loss and
  on held-out clean loss are nearly the same ranking; one-step effects are
  dominated by transient class imbalance and by the iterate sitting on the wall
  of a sharp valley. Weight-averaged checkpoints are both more accurate and more
  robust mid-training, and still show the rise-then-decay.
* Simple rules move along the accuracy-robustness curve, not off it: keeping
  easy samples is *dominated by training the baseline for fewer epochs*; keeping
  hard samples is like training longer. Selecting hard for a first-order
  "robustness" gain measured on held-out data collapses or stalls training.
* Each 20-epoch run of the small model costs 2.5-7 minutes on one GPU, a replay
  study of one checkpoint about 18 minutes, so direct experiments are cheap.
  The trainer can also branch from any checkpoint and run a stretch of the
  schedule under a different sampler (`init_ckpt`, `start_step`, `stop_step`).

## What I need from you

Think hard and be critical; the value you add is synthesis and ideas, not a
restatement of the journal. Use the data: load the `.npz` files and test your
ideas against them where you can, and say which claims you verified and which
are conjecture.

1. **Characterisation.** From the data, what in training makes the model
   adversarially vulnerable? What distinguishes the updates/samples that push the
   (weight-averaged) solution towards lower robustness at matched clean
   accuracy, if anything does: sample properties, class structure, training
   phase, properties of the update direction? Where the current measurements
   cannot answer, name the measurement that would.
2. **Method.** Propose a data-ordering/selection method (or a small family) that
   has a real chance of moving the model *off* the baseline's
   accuracy-robustness curve, with the reasoning, the hyperparameters that
   matter, safeguards against the degenerate outcomes already seen (collapse,
   stalled learning, merely trading accuracy for robustness), and its cost. It
   must work through the sampler seam (`sampler.next_batch(step, model)`); the
   weights are updated only with gradients of the chosen training samples.
   Say plainly if you conclude that selection alone under standard training
   cannot move the curve much, and what the best honest deliverable is then.
   Give a view on whether the method should also be evaluated on top of
   adversarial training, where "which updates destroy robustness" is the
   robust-overfitting question and robust accuracy at 8/255 is not zero.
3. **Evaluation protocol** that an independent assessor would find convincing:
   baselines (including early stopping, learning-rate/weight-decay changes,
   weight averaging), seeds, attacks stronger than the one used for selection,
   a second architecture and dataset, and how to report accuracy-robustness
   trade-offs fairly.
4. **A ranked list of experiments** for the next 6-8 GPU-hours on single
   RTX 6000 cards, each with the question it answers, the expected outcome under
   your hypothesis, and what result would falsify it.

Write the plan to `docs/plans/2026-11-15_phase2_plan.md`. Lead with the
conclusions and the ranked experiment list; put supporting analysis after.
Include the numbers you computed yourself, with enough detail to reproduce
them. Keep it under about 2,500 words excluding tables.
