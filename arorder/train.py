"""Step-based trainer with per-update logging and replayable checkpoints.

This is the instrumented baseline used to study individual updates. For every
optimiser step it records which training samples formed the batch and (when
enabled) the clean and adversarial behaviour of the model on a fixed probe set
*before* the step. Row ``t`` of the step log describes the weights after ``t``
updates, so the effect of the batch used at step ``t`` is ``log[t + 1] - log[t]``.

Checkpoint ``ckpt_{t:06d}.pt`` likewise holds the weights and optimiser state
after ``t`` updates, so any step can be replayed with a different batch.

Run as a module::

    python -m arorder.train --run-dir /path/to/run --epochs 20 --probe-every 1

Outputs in ``run_dir``:

* ``config.json``  - the full configuration;
* ``meta.json``    - host, GPU, Slurm job, start/end time, final metrics;
* ``steps.npz``    - per-step arrays (see :class:`StepLog`);
* ``evals.json``   - periodic full evaluations on the ``val`` split;
* ``ckpt_*.pt``    - checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DataBundle, load_cifar10
from .models import build_model
from .robustness import ProbeEvaluator


@dataclass
class TrainConfig:
    """Configuration of one training run. Radii are in units of 1/255."""

    run_dir: str = "runs/debug"
    # model
    model: str = "smallcnn"
    width: int = 32
    norm: str = "gn"
    # optimisation
    epochs: int = 20
    batch_size: int = 128
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: float = 1.0  # linear warm-up, then cosine decay to zero
    augment: bool = True
    seed: int = 0
    # data ordering: "uniform" (baseline shuffling) or a scorer name understood by
    # arorder.samplers.build_scorer (e.g. "loss:low"), applied by PoolFilterSampler
    sampler: str = "uniform"
    pool_mult: float = 2.0          # candidates per step as a multiple of batch_size
    select_start_epoch: float = 0.0  # selection is uniform before this epoch
    sampler_args: str = "{}"        # JSON keyword arguments for the scorer
    # per-step probe measurement (0 disables)
    probe_every: int = 0
    probe_n: int = 1000  # number of probe points used for the per-step measurement
    probe_eps: list[float] = field(default_factory=lambda: [1.0, 2.0])
    probe_steps: int = 5
    # periodic full evaluation on the val split
    eval_every: int = 351
    eval_eps: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    eval_steps: int = 10
    # checkpoints (0 disables); the final state is always saved
    ckpt_every: int = 0


class StepLog:
    """Accumulates per-step scalars and batch indices; saved as one ``.npz``.

    Arrays in the file: ``batch_idx`` (steps, batch) int32 positions within the
    ``train`` split; ``lr``, ``train_loss``, ``train_acc`` (steps,); and, when
    probing is on, ``probe_step`` (the step index of each probe row) and one
    ``probe/<metric>`` array per metric of :meth:`ProbeEvaluator.evaluate`.
    """

    def __init__(self):
        self.batch_idx, self.scalars, self.probe_steps, self.probe = [], {}, [], {}

    def add(self, batch_idx, **scalars):
        self.batch_idx.append(batch_idx.to(torch.int32).cpu().numpy())
        for k, v in scalars.items():
            self.scalars.setdefault(k, []).append(float(v))

    def add_probe(self, step, metrics: dict):
        self.probe_steps.append(step)
        for k, v in metrics.items():
            self.probe.setdefault(k, []).append(float(v))

    def save(self, path: Path):
        arrays = {"batch_idx": np.stack(self.batch_idx) if self.batch_idx else np.zeros((0, 0), np.int32)}
        arrays.update({k: np.asarray(v, np.float32) for k, v in self.scalars.items()})
        arrays["probe_step"] = np.asarray(self.probe_steps, np.int64)
        arrays.update({f"probe/{k}": np.asarray(v, np.float32) for k, v in self.probe.items()})
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)


def lr_at(step: int, cfg: TrainConfig, steps_per_epoch: int, total_steps: int) -> float:
    """Linear warm-up for ``warmup_epochs`` then cosine decay to zero."""
    warm = int(cfg.warmup_epochs * steps_per_epoch)
    if step < warm:
        return cfg.lr * (step + 1) / warm
    frac = (step - warm) / max(1, total_steps - warm)
    # Plain Python float: a numpy scalar would end up inside the optimiser state
    # and make checkpoints unloadable with torch.load(weights_only=True).
    return float(cfg.lr * 0.5 * (1 + math.cos(math.pi * frac)))


def environment_info() -> dict:
    """Host, GPU and Slurm job details, recorded with every run for the journal."""
    return {
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "torch": torch.__version__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def save_checkpoint(path: Path, model, opt, step: int):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)


def load_checkpoint(path: Path, map_location=None) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Checkpoints hold only tensors and plain Python numbers and are loaded with
    ``weights_only=True``. Checkpoints written before 2026-11-15 11:40 contain a
    numpy scalar (the learning rate) in the optimiser state, which the safe
    loader rejects; for those we fall back to full unpickling. Only do that with
    checkpoints produced by this code base.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as err:  # legacy checkpoint with a numpy scalar
        print(f"note: {Path(path).name} is a legacy checkpoint, loading with weights_only=False ({type(err).__name__})")
        return torch.load(path, map_location=map_location, weights_only=False)


def make_optimizer(model, cfg: TrainConfig):
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay, nesterov=False)


def train(cfg: TrainConfig, data: DataBundle | None = None, sampler=None) -> dict:
    """Run one training job and return the final ``val`` metrics.

    ``sampler`` may be any object with ``next_batch(step, model) -> LongTensor``
    of positions within the train split; the default is uniform shuffling
    without replacement (one pass over the data per epoch).
    """
    from .samplers import PoolFilterSampler, UniformSampler, build_scorer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    meta = {"start": environment_info()}

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    data = data or load_cifar10(dev)
    model = build_model(cfg.model, data.num_classes, width=cfg.width, norm=cfg.norm).to(dev)
    model = model.to(memory_format=torch.channels_last)
    opt = make_optimizer(model, cfg)

    n = len(data.train)
    steps_per_epoch = n // cfg.batch_size
    total_steps = cfg.epochs * steps_per_epoch
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + 1)  # augmentation RNG
    if sampler is None and cfg.sampler == "uniform":
        sampler = UniformSampler(n, cfg.batch_size, seed=cfg.seed + 2, device=dev)
    elif sampler is None:
        scorer = build_scorer(cfg.sampler, data=data, **json.loads(cfg.sampler_args))
        sampler = PoolFilterSampler(data.train, cfg.batch_size, cfg.pool_mult, scorer,
                                    start_step=int(cfg.select_start_epoch * steps_per_epoch), seed=cfg.seed + 2)

    probe_eval = None
    if cfg.probe_every:
        px, py = data.probe.batch(torch.arange(min(cfg.probe_n, len(data.probe)), device=dev))
        probe_eval = ProbeEvaluator(px, py, [e / 255 for e in cfg.probe_eps], steps=cfg.probe_steps, seed=1234)
    vx, vy = data.val.batch(torch.arange(len(data.val), device=dev))
    val_eval = ProbeEvaluator(vx, vy, [e / 255 for e in cfg.eval_eps], steps=cfg.eval_steps, seed=4321)

    log, evals = StepLog(), []
    t0 = time.time()
    for step in range(total_steps + 1):
        # --- measurements of the weights after `step` updates ---
        if probe_eval is not None and step % cfg.probe_every == 0:
            log.add_probe(step, probe_eval.evaluate(model))
        if step % cfg.eval_every == 0 or step == total_steps:
            ev = val_eval.evaluate(model)
            ev.update(step=step, epoch=step / steps_per_epoch, elapsed=time.time() - t0)
            evals.append(ev)
            print(f"step {step:6d} ep {ev['epoch']:5.2f} | clean {ev['clean_acc']:.3f} "
                  + " ".join(f"@{e:g}:{ev[f'adv_acc@{e:g}']:.3f}" for e in cfg.eval_eps)
                  + f" | auc {ev['auc']:.3f} linr {ev['lin_radius']:.3f} | {ev['elapsed']:.0f}s", flush=True)
            (run_dir / "evals.json").write_text(json.dumps(evals, indent=1))
            log.save(run_dir / "steps.npz")
        if cfg.ckpt_every and step % cfg.ckpt_every == 0:
            save_checkpoint(run_dir / f"ckpt_{step:06d}.pt", model, opt, step)
        if step == total_steps:
            break

        # --- one update ---
        lr = lr_at(step, cfg, steps_per_epoch, total_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        idx = sampler.next_batch(step, model)
        x, y = data.train.batch(idx, augment=cfg.augment, generator=gen)
        model.train()
        logits = model(x.contiguous(memory_format=torch.channels_last))
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log.add(idx, lr=lr, train_loss=loss.item(), train_acc=(logits.argmax(1) == y).float().mean().item(),
                **(sampler.stats() if hasattr(sampler, "stats") else {}))

    save_checkpoint(run_dir / "ckpt_final.pt", model, opt, total_steps)
    log.save(run_dir / "steps.npz")
    meta["end"] = environment_info()
    meta["train_seconds"] = time.time() - t0
    meta["final_val"] = evals[-1]
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return evals[-1]


def _parse_args() -> TrainConfig:
    """Build an argparse CLI from the dataclass fields (``--probe-eps 1 2`` etc.)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(TrainConfig):
        name = "--" + f.name.replace("_", "-")
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if isinstance(default, bool):
            p.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        elif isinstance(default, list):
            p.add_argument(name, type=float, nargs="*", default=default)
        else:
            p.add_argument(name, type=type(default), default=default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
