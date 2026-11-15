"""Batch samplers: objects that decide which training samples form each update.

The trainer calls ``sampler.next_batch(step, model)`` once per optimiser step
and trains on whatever positions (within the train split) come back. This is the
single seam through which a data-ordering method acts, so a selection rule is
independent of the model, the loss and the dataset.

Two samplers are provided:

* :class:`UniformSampler` - ordinary shuffling (the baseline);
* :class:`PoolFilterSampler` - draws ``pool_mult`` times more candidates than
  needed, scores them with the current model, and keeps the best. The number of
  optimiser steps is unchanged, so comparisons against the baseline are at an
  equal update budget (the extra cost is the scoring pass).

Scores are produced by *scorers*: callables ``scorer(model, x, y, step) ->
Tensor[N]`` where a higher score means "prefer to train on this sample".
:func:`build_scorer` constructs the registered ones by name.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from .attacks import margin, pgd_linf


class UniformSampler:
    """Standard shuffling: each epoch is a fresh random permutation, last partial batch dropped."""

    def __init__(self, n: int, batch_size: int, seed: int = 0, device="cpu"):
        self.n, self.batch_size, self.device = n, batch_size, device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._perm, self._pos = None, 0

    def next_batch(self, step: int, model=None) -> torch.Tensor:
        if self._perm is None or self._pos + self.batch_size > self.n:
            self._perm = torch.randperm(self.n, generator=self.gen).to(self.device)
            self._pos = 0
        idx = self._perm[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return idx

    def stats(self) -> dict:
        """Scalars describing the most recent selection (logged per step)."""
        return {}


class ClassBalancedSampler:
    """Shuffling in which every batch holds (almost) the same number of samples per class.

    Each class gets its own random permutation; the stream interleaves the
    classes in a fresh random order per round, so any window of ``batch_size``
    consecutive samples is class-balanced to within one sample. This is a pure
    ordering change: every sample is still used equally often (classes with
    fewer samples wrap around slightly earlier).

    Motivation: at a given moment the raw training iterate over-predicts some
    classes, and the one-step effect of a batch on held-out loss is dominated by
    which classes happen to be in it (JOURNAL.md, 2026-11-15). Balancing every
    batch removes that source of step-to-step noise.
    """

    def __init__(self, labels: torch.Tensor, batch_size: int, seed: int = 0, device="cpu"):
        self.batch_size, self.device = batch_size, device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        labels = labels.cpu()
        self.by_class = [torch.nonzero(labels == c, as_tuple=False).flatten() for c in labels.unique().tolist()]
        self._perms = [self._shuffled(members) for members in self.by_class]
        self._pos = [0] * len(self.by_class)
        self._buffer = torch.empty(0, dtype=torch.long)

    def _shuffled(self, members):
        return members[torch.randperm(len(members), generator=self.gen)]

    def _next_round(self) -> torch.Tensor:
        """One sample from every class, classes in random order."""
        out = []
        for c in torch.randperm(len(self.by_class), generator=self.gen).tolist():
            if self._pos[c] >= len(self._perms[c]):
                self._perms[c], self._pos[c] = self._shuffled(self.by_class[c]), 0
            out.append(self._perms[c][self._pos[c]])
            self._pos[c] += 1
        return torch.stack(out)

    def next_batch(self, step: int, model=None) -> torch.Tensor:
        while len(self._buffer) < self.batch_size:
            self._buffer = torch.cat([self._buffer, self._next_round()])
        idx, self._buffer = self._buffer[: self.batch_size], self._buffer[self.batch_size :]
        return idx.to(self.device)

    def stats(self) -> dict:
        return {}


class PoolFilterSampler:
    """Keep the ``batch_size`` best-scoring samples out of a uniform candidate pool.

    Args:
        data: the train split (a :class:`arorder.data.GPUData`).
        batch_size: number of samples trained on per step.
        pool_mult: candidates drawn per step, as a multiple of ``batch_size``.
            ``keep_frac = 1 / pool_mult`` of candidates are trained on.
        scorer: ``scorer(model, x, y, step) -> Tensor[N]``, higher = preferred.
        start_step: before this step the sampler behaves uniformly (selection
            rules that rely on the model are meaningless at initialisation).
        seed: seed of the candidate stream.

    Candidates are scored on un-augmented images with the model in eval mode;
    the trainer applies augmentation afterwards.
    """

    def __init__(self, data, batch_size: int, pool_mult: float, scorer, start_step: int = 0, seed: int = 0,
                 base=None):
        self.data, self.batch_size, self.scorer, self.start_step = data, batch_size, scorer, start_step
        self.pool_size = int(round(pool_mult * batch_size))
        # `base` supplies the candidate pools (default: uniform shuffling)
        self.base = base or UniformSampler(len(data), self.pool_size, seed=seed, device=data.device)
        self._stats: dict = {}

    def next_batch(self, step: int, model=None) -> torch.Tensor:
        pool = self.base.next_batch(step)
        if step < self.start_step or self.pool_size == self.batch_size:
            self._stats = {}
            return pool[: self.batch_size]
        x, y = self.data.batch(pool)
        was_training = model.training
        model.eval()
        scores = self.scorer(model, x, y, step)
        model.train(was_training)
        keep = scores.topk(self.batch_size).indices
        self._stats = {"sel/score_kept": scores[keep].mean().item(), "sel/score_pool": scores.mean().item()}
        return pool[keep]

    def stats(self) -> dict:
        return self._stats


# --------------------------------------------------------------------------- scorers


def directional_derivatives(model, x, y, vectors: torch.Tensor) -> torch.Tensor:
    """``out[i, k] = <grad_theta loss_i, vectors[k]>`` via forward-mode differentiation.

    One forward-mode pass per direction gives the product for every sample in
    the batch at once, without materialising per-sample gradients. Cost is a few
    forward passes per direction.
    """
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    shapes = [params[n].shape for n in names]
    sizes = [params[n].numel() for n in names]

    def f(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (x,)), y, reduction="none")

    out = []
    for v in vectors:
        tangents = {n: t.view(s) for n, t, s in zip(names, v.split(sizes), shapes)}
        out.append(jvp(f, (params,), (tangents,))[1])
    return torch.stack(out, dim=1)


@torch.no_grad()
def _logits(model, x):
    return model(x)


def score_loss(model, x, y, step):
    """Per-sample cross-entropy loss."""
    return F.cross_entropy(_logits(model, x), y, reduction="none")


def score_margin(model, x, y, step):
    """Per-sample classification margin."""
    return margin(_logits(model, x), y)


def score_lin_radius(model, x, y, step):
    """First-order L-inf distance of each sample to the decision boundary (pixel units)."""
    x = x.clone().requires_grad_(True)
    m = margin(model(x), y)
    (g,) = torch.autograd.grad(m.sum(), x)
    return m.detach() / g.abs().flatten(1).sum(1).clamp_min(1e-12)


class AdvMarginScorer:
    """The sample's own margin after a short PGD attack at radius ``eps`` (in 1/255)."""

    def __init__(self, eps: float = 2.0, steps: int = 3):
        self.eps, self.steps = eps / 255, steps

    def __call__(self, model, x, y, step):
        xa = pgd_linf(model, x, y, self.eps, steps=self.steps, keep_best=False)
        return margin(_logits(model, xa), y)


class Negated:
    """Wrap a scorer so that *low* raw scores are preferred."""

    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, model, x, y, step):
        return -self.scorer(model, x, y, step)


class Band:
    """Prefer samples whose raw score is closest to a target quantile of the pool.

    ``q = 0.5`` keeps the middle of the pool and drops both extremes.
    """

    def __init__(self, scorer, q: float = 0.5):
        self.scorer, self.q = scorer, q

    def __call__(self, model, x, y, step):
        s = self.scorer(model, x, y, step)
        ranks = s.argsort().argsort().float() / max(len(s) - 1, 1)
        return -(ranks - self.q).abs()


class ProbeGradientScorer:
    """Score candidates by their first-order effect on held-out robustness.

    Every ``refresh_every`` steps the gradients of the clean loss and of the
    adversarial loss (PGD at radius ``eps``, in 1/255) are computed on
    ``probe_batch`` held-out probe points. A candidate ``i`` with training
    gradient ``g_i`` then gets (higher = preferred, i.e. training on it lowers
    the quantity in question to first order):

    * ``mode="adv"``:   ``<g_i, g_adv>``, held-out adversarial loss;
    * ``mode="clean"``: ``<g_i, g_clean>``, held-out clean loss;
    * ``mode="gap"``:   ``<g_i, g_adv - g_clean>``, the adversarial-minus-clean gap;
    * ``mode="rob"``:   ``<g_i, r>`` with ``r`` the part of ``g_adv`` orthogonal
      to ``g_clean``: adversarial loss at unchanged clean loss;
    * ``mode="tilt"``:  ``<g_i, unit(g_clean) + lam * unit(r)>``: clean progress
      tilted towards robustness. ``lam = 0`` is ``clean``; large ``lam``
      approaches ``rob``.

    ``source`` says where the two gradients are measured: ``"probe"`` (held-out
    probe points, refreshed every ``refresh_every`` steps) or ``"pool"`` (the
    candidate pool being scored, recomputed at every step, no held-out data).

    The inner products are computed for the whole candidate pool with one
    forward-mode pass per direction (:func:`directional_derivatives`).

    Only the *choice* of samples uses the probe set; the weights are still
    updated with ordinary clean-sample gradients.
    """

    def __init__(self, probe, eps: float = 2.0, pgd_steps: int = 5, probe_batch: int = 1000,
                 refresh_every: int = 10, mode: str = "rob", lam: float = 1.0, source: str = "probe",
                 seed: int = 0):
        assert mode in ("adv", "clean", "gap", "rob", "tilt") and source in ("probe", "pool")
        self.lam, self.source = lam, source
        self.probe, self.eps, self.pgd_steps = probe, eps / 255, pgd_steps
        self.probe_batch, self.refresh_every, self.mode = min(probe_batch, len(probe)), refresh_every, mode
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.direction, self._last_refresh = None, None

    def _flat_grad(self, model, x, y):
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(F.cross_entropy(model(x), y), params)
        return torch.cat([g.reshape(-1) for g in grads])

    def refresh(self, model, x=None, y=None) -> None:
        """Recompute the scoring direction on ``(x, y)``, or on a fresh random probe mini-batch."""
        if x is None:
            idx = torch.randperm(len(self.probe), generator=self.gen)[: self.probe_batch].to(self.probe.device)
            x, y = self.probe.batch(idx)
        g_clean = self._flat_grad(model, x, y)
        if self.mode == "clean":
            self.direction = g_clean
            return
        g_adv = self._flat_grad(model, pgd_linf(model, x, y, self.eps, steps=self.pgd_steps, keep_best=False), y)
        if self.mode == "adv":
            self.direction = g_adv
        elif self.mode == "gap":
            self.direction = g_adv - g_clean
        else:
            r = g_adv - (g_adv @ g_clean) / (g_clean @ g_clean).clamp_min(1e-30) * g_clean
            if self.mode == "rob":
                self.direction = r
            else:
                self.direction = g_clean / g_clean.norm().clamp_min(1e-30) + self.lam * r / r.norm().clamp_min(1e-30)

    def __call__(self, model, x, y, step):
        if self.source == "pool":
            self.refresh(model, x, y)
        elif self.direction is None or step - self._last_refresh >= self.refresh_every:
            self.refresh(model)
            self._last_refresh = step
        return directional_derivatives(model, x, y, self.direction[None])[:, 0]


# Factories take the DataBundle (for scorers that need held-out data) and keyword arguments.
_BASE_SCORERS = {
    "loss": lambda data, **kw: score_loss,
    "margin": lambda data, **kw: score_margin,
    "lin_radius": lambda data, **kw: score_lin_radius,
    "adv_margin": lambda data, **kw: AdvMarginScorer(**kw),
    "probe_grad": lambda data, **kw: ProbeGradientScorer(data.probe, **kw),
}


def build_scorer(name: str, data=None, **kw):
    """Build a scorer from a name such as ``"loss:low"``, ``"margin:high"`` or ``"loss:band"``.

    The part before the colon picks the quantity (``loss``, ``margin``,
    ``lin_radius``, ``adv_margin``, ``probe_grad``); the part after picks which end of the pool
    is kept: ``high``, ``low`` or ``band`` (closest to pool quantile ``q``,
    default the median). ``data`` is the :class:`arorder.data.DataBundle`, needed
    by scorers that use held-out points (``probe_grad``). Extra keyword
    arguments go to the quantity's scorer.
    """
    base_name, _, mode = name.partition(":")
    q = kw.pop("q", 0.5)
    if base_name not in _BASE_SCORERS:
        raise ValueError(f"unknown scorer {base_name!r}; known: {sorted(_BASE_SCORERS)}")
    base = _BASE_SCORERS[base_name](data, **kw)
    if mode in ("", "high"):
        return base
    if mode == "low":
        return Negated(base)
    if mode == "band":
        return Band(base, q)
    raise ValueError(f"unknown mode {mode!r} in scorer {name!r}")


def register_scorer(name: str, factory) -> None:
    """Register an additional base scorer factory under ``name``."""
    _BASE_SCORERS[name] = factory
