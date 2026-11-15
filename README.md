# Ordering training data for adversarial robustness

**Question.** During ordinary training, which individual updates make a model
less adversarially robust, and can we choose which samples to train on (and
when, including never) so that the model ends up more robust without giving up
clean accuracy?

**Status.** Work in progress. See `TODO.md` for the plan and `JOURNAL.md` for
every experiment run so far, with hardware and timestamps.

## Layout

| Path | What it is |
|---|---|
| `arorder/data.py` | Whole dataset on GPU, fixed train / probe / val split, GPU augmentation. |
| `arorder/models.py` | Small CNN and small ResNets; input normalisation inside the model. |
| `arorder/attacks.py` | Deterministic L-inf PGD (explicit random starts, for paired comparisons). |
| `arorder/robustness.py` | `ProbeEvaluator`: clean and adversarial metrics on fixed held-out points, and the gradient of held-out adversarial loss. |
| `arorder/train.py` | Step-based trainer; logs the batch and probe robustness for every update; saves replayable checkpoints. |
| `arorder/samplers.py` | The seam where a data-ordering rule plugs in: `next_batch(step, model)`. |
| `arorder/investigate.py` | Counterfactual study of single updates from a checkpoint: per-sample first-order effects on held-out robustness plus real finite steps. |
| `scripts/` | Data preparation, analysis, Slurm job wrapper (`scripts/slurm/run.slurm`), environment (`scripts/env.sh`). |
| `tests/` | Unit tests of the measurement machinery. |
| `docs/KLONE.md` | How to work on the UW klone cluster. |

## Running

Everything that computes runs inside a Slurm job (see `docs/KLONE.md`).

```bash
# one-time: download cifar-10-python.tar.gz into $AR_DATA/raw on a login node, then in a job:
source scripts/env.sh && $PY scripts/prepare_cifar10.py

# tests
srun -A amath -p gpu-rtx6k --gpus=1 -c 4 --mem=16G -t 15 bash -c 'source scripts/env.sh && $PY -m pytest -q tests'

# instrumented baseline, then the per-update study at chosen checkpoints
sbatch --job-name=base scripts/slurm/run.slurm python -m arorder.train --run-dir $AR_RUNS/base --epochs 20 --probe-every 1 --ckpt-every 117
sbatch --job-name=inv  scripts/slurm/run.slurm python -m arorder.investigate --run-dir $AR_RUNS/base --step 351 3510
```

## Conventions

* Attack radii are L-infinity, in units of 1/255 on inputs in `[0, 1]`.
* `train` / `probe` / `val` are a fixed split of the official training set
  (45,000 / 2,000 / 3,000). `probe` drives selection signals, `val` checks
  them, and the official test set is used only for final reporting.
* Sample indices in logs are positions within the `train` split.
