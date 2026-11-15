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
* ``lookahead.json`` - for runs with lookahead order selection, every candidate's score;
* ``ckpt_*.pt``    - checkpoints.
"""

from __future__ import annotations

import argparse
import copy
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

from .attacks import pgd_linf
from .data import DataBundle, load_dataset
from .models import build_model
from .robustness import ProbeEvaluator


@dataclass
class TrainConfig:
    """Configuration of one training run. Radii are in units of 1/255."""

    run_dir: str = "runs/debug"
    dataset: str = "cifar10"  # name of a prepared dataset, see arorder.data.load_dataset
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
    # data ordering: "uniform" (baseline shuffling), "class_balanced" (every batch
    # has equal class counts), "grad_match" (clean samples whose combined gradient
    # matches the adversarial-training direction), or a scorer name understood by
    # arorder.samplers.build_scorer (e.g. "loss:low"), applied by PoolFilterSampler
    sampler: str = "uniform"
    pool_mult: float = 2.0          # candidates per step as a multiple of batch_size
    select_start_epoch: float = 0.0  # selection is uniform before this epoch
    sampler_args: str = "{}"        # JSON keyword arguments for the scorer
    # which weights the scorer sees: "raw" (the training iterate) or "ema" (the
    # weight average; needs ema_decay > 0). The average is where single-update
    # effects reflect the slow drift rather than the iterate's oscillation.
    score_weights: str = "raw"
    # exponential moving average of the weights (0 disables). The average sits near
    # the floor of the loss valley that the raw iterate oscillates across, so it
    # is the less noisy place to measure a model; it is evaluated alongside the raw
    # weights (metrics prefixed "ema_") and stored in checkpoints.
    ema_decay: float = 0.0
    # branching: start from a checkpoint and run only steps [start_step, stop_step)
    # of the schedule (stop_step = 0 means "to the end")
    init_ckpt: str = ""
    start_step: int = 0
    stop_step: int = 0
    # adversarial training: radius of the PGD attack applied to each training batch
    # (0 = standard training on clean samples) and its number of steps
    train_eps: float = 0.0
    train_attack_steps: int = 3
    # lookahead order selection (lookahead_k = 1 disables): until
    # lookahead_until_epoch, every lookahead_len steps the run is continued under
    # lookahead_k different data orders and the one whose end state scores best on
    # the probe split (robustness + lookahead_lambda * clean accuracy) is kept
    lookahead_k: int = 1
    lookahead_len: int = 351
    lookahead_until_epoch: float = 6.0
    lookahead_metric: str = "auc"       # "auc", "adv_acc" or "lin_radius"
    lookahead_lambda: float = 1.0
    lookahead_eps: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    lookahead_weights: str = "ema"      # score the weight average ("ema") or the raw iterate ("raw")
    lookahead_diag: bool = False        # also record every candidate's val metrics
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

    def copy(self) -> "StepLog":
        """A shallow copy that can be extended independently (entries are never mutated)."""
        out = StepLog()
        out.batch_idx, out.probe_steps = list(self.batch_idx), list(self.probe_steps)
        out.scalars = {k: list(v) for k, v in self.scalars.items()}
        out.probe = {k: list(v) for k, v in self.probe.items()}
        return out

    def add_probe(self, step, metrics: dict):
        self.probe_steps.append(step)
        for k, v in metrics.items():
            self.probe.setdefault(k, []).append(float(v))

    def save(self, path: Path):
        arrays = {"batch_idx": np.stack(self.batch_idx) if self.batch_idx else np.zeros((0, 0), np.int32)}
        n = len(self.batch_idx)  # scalars logged only on some steps (sampler statistics) are left-padded with NaN
        arrays.update({k: np.asarray([np.nan] * (n - len(v)) + list(v), np.float32) for k, v in self.scalars.items()})
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


def save_checkpoint(path: Path, model, opt, step: int, ema=None):
    state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


@torch.no_grad()
def update_ema(ema, model, decay: float) -> None:
    """``ema <- decay * ema + (1 - decay) * model`` for every parameter."""
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)


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


class Trainer:
    """One training run as an object that can be stepped, snapshotted and restored.

    ``Trainer(cfg).fit()`` is an ordinary run. The pieces are public because the
    studies in this project need them: :meth:`train_step` performs one update,
    :meth:`snapshot` / :meth:`restore` rewind the run to an earlier state (model,
    optimiser, weight average, step counter and step log), and
    :meth:`set_order_seed` replaces the stream of batches. Lookahead order
    selection (see :meth:`lookahead_segment`) is built from exactly these.

    ``sampler`` may be any object with ``next_batch(step, model) -> LongTensor``
    of positions within the train split; by default it is built from
    ``cfg.sampler``.
    """

    def __init__(self, cfg: TrainConfig, data: DataBundle | None = None, sampler=None):
        self.cfg = cfg
        self.dev = dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
        self.meta = {"start": environment_info()}

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cudnn.benchmark = True
        self.data = data = data or load_dataset(cfg.dataset, dev)
        self.model = build_model(cfg.model, data.num_classes, width=cfg.width, norm=cfg.norm).to(dev)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.opt = make_optimizer(self.model, cfg)
        self.ema = None
        state = None
        if cfg.init_ckpt:
            state = load_checkpoint(Path(cfg.init_ckpt), map_location=dev)
            self.model.load_state_dict(state["model"])
            self.opt.load_state_dict(state["opt"])
        if cfg.ema_decay > 0:
            self.ema = copy.deepcopy(self.model).eval()
            if state is not None and "ema" in state:
                self.ema.load_state_dict(state["ema"])

        self.steps_per_epoch = len(data.train) // cfg.batch_size
        self.total_steps = cfg.epochs * self.steps_per_epoch
        self.first_step, self.last_step = cfg.start_step, cfg.stop_step or self.total_steps
        self.step = self.first_step
        self.gen = torch.Generator(device=dev).manual_seed(cfg.seed + 1)  # augmentation / attack-start RNG
        self.sampler = sampler or self._make_sampler(cfg.seed + 2)

        self.probe_eval = None
        if cfg.probe_every:
            px, py = data.probe.batch(torch.arange(min(cfg.probe_n, len(data.probe)), device=dev))
            self.probe_eval = ProbeEvaluator(px, py, [e / 255 for e in cfg.probe_eps], steps=cfg.probe_steps, seed=1234)
        vx, vy = data.val.batch(torch.arange(len(data.val), device=dev))
        self.val_eval = ProbeEvaluator(vx, vy, [e / 255 for e in cfg.eval_eps], steps=cfg.eval_steps, seed=4321)
        self.select_eval = None
        if cfg.lookahead_k > 1:
            assert not cfg.probe_every, "per-step probing and lookahead cannot be combined"
            px, py = data.probe.batch(torch.arange(len(data.probe), device=dev))
            self.select_eval = ProbeEvaluator(px, py, [e / 255 for e in cfg.lookahead_eps], steps=cfg.eval_steps, seed=1234)
        self.log, self.evals, self.lookahead_log = StepLog(), [], []
        self.t0 = time.time()

    # ------------------------------------------------------------------ batches

    def _make_sampler(self, seed: int):
        from .samplers import (ClassBalancedSampler, GradientMatchSampler, PoolFilterSampler, UniformSampler,
                               build_scorer)

        cfg, n = self.cfg, len(self.data.train)
        if cfg.sampler == "uniform":
            return UniformSampler(n, cfg.batch_size, seed=seed, device=self.dev)
        if cfg.sampler == "class_balanced":
            return ClassBalancedSampler(self.data.train.y, cfg.batch_size, seed=seed, device=self.dev)
        start = int(cfg.select_start_epoch * self.steps_per_epoch)
        if cfg.sampler == "grad_match":
            return GradientMatchSampler(self.data.train, self.data.probe, cfg.batch_size, pool_mult=cfg.pool_mult,
                                        start_step=start, seed=seed, **json.loads(cfg.sampler_args))
        scorer = build_scorer(cfg.sampler, data=self.data, **json.loads(cfg.sampler_args))
        return PoolFilterSampler(self.data.train, cfg.batch_size, cfg.pool_mult, scorer,
                                 start_step=int(cfg.select_start_epoch * self.steps_per_epoch), seed=seed)

    def set_order_seed(self, seed: int) -> None:
        """Replace the stream of batches (and augmentation draws) by the one for ``seed``."""
        self.sampler = self._make_sampler(seed)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed + 1)

    # ------------------------------------------------------------------ one update

    def train_step(self) -> None:
        """Perform the update for ``self.step`` and advance the step counter."""
        cfg, model = self.cfg, self.model
        lr = lr_at(self.step, cfg, self.steps_per_epoch, self.total_steps)
        for group in self.opt.param_groups:
            group["lr"] = lr
        scoring_model = self.ema if (cfg.score_weights == "ema" and self.ema is not None) else model
        idx = self.sampler.next_batch(self.step, scoring_model)
        x, y = self.data.train.batch(idx, augment=cfg.augment, generator=self.gen)
        x = x.contiguous(memory_format=torch.channels_last)
        model.train()
        if cfg.train_eps > 0:  # adversarial training: train on PGD versions of the chosen samples
            init = torch.rand(x.shape, device=self.dev, generator=self.gen) * 2 - 1
            x = pgd_linf(model, x, y, cfg.train_eps / 255, steps=cfg.train_attack_steps, init=init, keep_best=False)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.ema is not None:
            update_ema(self.ema, model, cfg.ema_decay)
        self.log.add(idx, lr=lr, train_loss=loss.item(), train_acc=(logits.argmax(1) == y).float().mean().item(),
                     **(self.sampler.stats() if hasattr(self.sampler, "stats") else {}))
        self.step += 1

    # ------------------------------------------------------------------ rewind

    def snapshot(self) -> dict:
        """Everything needed to return to the current point of the run."""
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "opt": copy.deepcopy(self.opt.state_dict()),
            "ema": copy.deepcopy(self.ema.state_dict()) if self.ema is not None else None,
            "step": self.step,
            "log": self.log.copy(),
        }

    def restore(self, snap: dict) -> None:
        self.model.load_state_dict(snap["model"])
        self.opt.load_state_dict(copy.deepcopy(snap["opt"]))
        if self.ema is not None:
            self.ema.load_state_dict(snap["ema"])
        self.step = snap["step"]
        self.log = snap["log"].copy()

    # ------------------------------------------------------------------ measurement

    def evaluate(self) -> dict:
        """Val metrics of the raw weights, plus ``ema_``-prefixed ones for the weight average."""
        ev = self.val_eval.evaluate(self.model, per_class=True)
        if self.ema is not None:
            ev.update({f"ema_{k}": v for k, v in self.val_eval.evaluate(self.ema, per_class=True).items()})
        ev.update(step=self.step, epoch=self.step / self.steps_per_epoch, elapsed=time.time() - self.t0)
        return ev

    def measure(self) -> None:
        """Measurements due for the weights after ``self.step`` updates (probe, val, checkpoint)."""
        cfg, step = self.cfg, self.step
        if self.probe_eval is not None and step % cfg.probe_every == 0:
            self.log.add_probe(step, self.probe_eval.evaluate(self.model))
        if (step - self.first_step) % cfg.eval_every == 0 or step == self.last_step:
            ev = self.evaluate()
            self.evals.append(ev)
            pre = "ema_" if self.ema is not None else ""
            print(f"step {step:6d} ep {ev['epoch']:5.2f} | clean {ev['clean_acc']:.3f} "
                  + " ".join(f"@{e:g}:{ev[f'adv_acc@{e:g}']:.3f}" for e in cfg.eval_eps)
                  + f" | auc {ev['auc']:.3f} linr {ev['lin_radius']:.3f}"
                  + (f" | ema clean {ev['ema_clean_acc']:.3f} @2:{ev.get('ema_adv_acc@2', float('nan')):.3f} "
                     f"auc {ev['ema_auc']:.3f}" if pre else "")
                  + f" | {ev['elapsed']:.0f}s", flush=True)
            (self.run_dir / "evals.json").write_text(json.dumps(self.evals, indent=1))
            self.log.save(self.run_dir / "steps.npz")
        if cfg.ckpt_every and step % cfg.ckpt_every == 0:
            save_checkpoint(self.run_dir / f"ckpt_{step:06d}.pt", self.model, self.opt, step, self.ema)

    # ------------------------------------------------------------------ lookahead order selection

    def selection_score(self) -> tuple[float, dict]:
        """Score of the current state for lookahead: robustness plus ``lambda`` x clean accuracy, on ``probe``.

        Measured on the weight average when there is one (the raw iterate's
        metrics swing from step to step). The robustness term is, by
        ``cfg.lookahead_metric``: ``"auc"`` - robust accuracy averaged over
        radii from 0 to the largest of ``cfg.lookahead_eps``; ``"adv_acc"`` -
        accuracy at that largest radius; ``"lin_radius"`` - mean linearised
        boundary distance divided by its cap (16/255).
        """
        cfg = self.cfg
        model = self.ema if (self.ema is not None and cfg.lookahead_weights == "ema") else self.model
        ev = self.select_eval.evaluate(model)
        top = f"{max(cfg.lookahead_eps):g}"
        robust = {"auc": ev.get("auc", ev[f"adv_acc@{top}"] * max(cfg.lookahead_eps)) / max(cfg.lookahead_eps),
                  "adv_acc": ev[f"adv_acc@{top}"], "lin_radius": ev["lin_radius"] / 16.0}[cfg.lookahead_metric]
        return robust + cfg.lookahead_lambda * ev["clean_acc"], dict(robust=robust, **ev)

    def lookahead_segment(self, seg_end: int) -> None:
        """Advance to ``seg_end`` along the best of ``lookahead_k`` candidate data orders.

        From the current state the run is continued ``lookahead_k`` times, each
        time with a different random order of the training data (candidate 0 is
        the order the run would have used anyway). Each candidate's end state is
        scored on the probe split by :meth:`selection_score`; the best one is
        kept. The kept trajectory is ordinary SGD on training samples: only the
        choice among orders used held-out measurements. The batches of the kept
        candidate are in the step log, so the chosen order can be replayed.
        """
        cfg, start = self.cfg, self.step
        snap = self.snapshot()
        candidates = []
        for c in range(cfg.lookahead_k):
            self.restore(snap)
            self.set_order_seed(cfg.seed + 7919 * (start + 1) + 104729 * c)
            while self.step < seg_end:
                self.train_step()
            score, detail = self.selection_score()
            rec = dict(candidate=c, score=score, **{k: detail[k] for k in detail if not k.startswith("class_")})
            if cfg.lookahead_diag:  # the same candidate on val, to check that selection is not fitting probe noise
                vm = self.ema if self.ema is not None else self.model
                rec.update({f"val_{k}": v for k, v in self.val_eval.evaluate(vm).items()})
            candidates.append((score, rec, self.snapshot()))
        best = max(range(len(candidates)), key=lambda i: candidates[i][0])
        self.restore(candidates[best][2])
        self.lookahead_log.append(dict(start=start, end=seg_end, chosen=best, candidates=[c[1] for c in candidates]))
        (self.run_dir / "lookahead.json").write_text(json.dumps(self.lookahead_log, indent=1))

    # ------------------------------------------------------------------ whole run

    def fit(self) -> dict:
        """Run from ``first_step`` to ``last_step`` and return the final val metrics."""
        cfg = self.cfg
        window_end = int(cfg.lookahead_until_epoch * self.steps_per_epoch)
        while True:
            self.measure()
            if self.step >= self.last_step:
                break
            in_window = cfg.lookahead_k > 1 and self.step < window_end
            if in_window and (self.step - self.first_step) % cfg.lookahead_len == 0:
                self.lookahead_segment(min(self.step + cfg.lookahead_len, self.last_step, window_end))
            else:
                self.train_step()
        save_checkpoint(self.run_dir / "ckpt_final.pt", self.model, self.opt, self.last_step, self.ema)
        self.log.save(self.run_dir / "steps.npz")
        self.meta["end"] = environment_info()
        self.meta["train_seconds"] = time.time() - self.t0
        self.meta["final_val"] = self.evals[-1]
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=1))
        return self.evals[-1]


def train(cfg: TrainConfig, data: DataBundle | None = None, sampler=None) -> dict:
    """Run one training job and return the final ``val`` metrics (see :class:`Trainer`)."""
    return Trainer(cfg, data=data, sampler=sampler).fit()


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
