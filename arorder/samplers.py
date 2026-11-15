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
from torch.func import functional_call, grad, jvp, vmap

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


# --------------------------------------------------------------------------- gradient matching


def per_sample_gradient_matrix(model, x, y, chunk: int = 128, coords: torch.Tensor | None = None) -> torch.Tensor:
    """``(N, P)`` matrix whose row ``i`` is the flat loss gradient of sample ``i``.

    Uses ``torch.func`` per-sample gradients. With ``coords`` (a LongTensor of
    parameter positions) only those coordinates are kept, chunk by chunk, so
    memory is ``N * len(coords)`` floats plus one chunk of full gradients; a few
    thousand random coordinates are enough for gradient matching (JOURNAL.md,
    2026-11-15). The model must be in eval mode (per-sample gradients of
    train-mode BatchNorm are undefined).
    """
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())

    def loss_one(p, xi, yi):
        return F.cross_entropy(functional_call(model, (p, buffers), (xi[None],)), yi[None])

    per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))
    rows = []
    for s in range(0, x.shape[0], chunk):
        g = per_sample(params, x[s : s + chunk], y[s : s + chunk])
        flat = torch.cat([g[n].flatten(1) for n in names], dim=1)
        rows.append(flat if coords is None else flat[:, coords])
    return torch.cat(rows)


def nonneg_least_squares(K: torch.Tensor, b: torch.Tensor, iters: int = 300) -> torch.Tensor:
    """``argmin_{w >= 0} w^T K w - 2 b^T w`` by accelerated projected gradient.

    With ``K = G G^T`` and ``b = G t`` this is the non-negative combination of
    the rows of ``G`` closest to ``t``.
    """
    lip = torch.linalg.eigvalsh(K)[-1].clamp_min(1e-30)
    w = torch.zeros_like(b)
    z, tk = w.clone(), 1.0
    for _ in range(iters):
        w_new = (z - (K @ z - b) / lip).clamp_min(0)
        tk_new = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        z = w_new + (tk - 1) / tk_new * (w_new - w)
        w, tk = w_new, tk_new
    return w


def project_capped_simplex(v: torch.Tensor, total: float, cap: float, rounds: int = 3, grid: int = 64) -> torch.Tensor:
    """Euclidean projection of the rows of ``v`` onto ``{w : 0 <= w_i <= cap, sum(w) = total}``.

    The projection is ``clip(v - tau, 0, cap)`` for the shift ``tau`` that makes
    the sum right; ``tau`` is found by ``rounds`` of evaluating ``grid`` candidate
    shifts at once and narrowing the bracket (a vectorised bisection: each round
    shrinks the bracket ``grid``-fold). ``v`` may be a vector or a batch of
    vectors ``(batch, n)``.
    """
    squeeze = v.dim() == 1
    v = v[None] if squeeze else v
    lo, hi = v.amin(1) - cap, v.amax(1)                       # sum is n*cap >= total at lo and 0 <= total at hi
    frac = torch.linspace(0, 1, grid, device=v.device, dtype=v.dtype)
    for _ in range(rounds):
        taus = lo[:, None] + (hi - lo)[:, None] * frac[None]   # (batch, grid)
        sums = (v[:, None, :] - taus[:, :, None]).clamp(0, cap).sum(2)
        k = ((sums > total).sum(1) - 1).clamp(0, grid - 2)     # last grid point whose sum is still too large
        lo, hi = taus.gather(1, k[:, None]).squeeze(1), taus.gather(1, k[:, None] + 1).squeeze(1)
    w = (v - ((lo + hi) / 2)[:, None]).clamp(0, cap)
    return w[0] if squeeze else w


def capped_direction_fit(K: torch.Tensor, b: torch.Tensor, t_norm: float, total: float, cap: float,
                         iters: int = 80, scales=(1.0, 2.0, 4.0)) -> torch.Tensor:
    """Weights ``0 <= w_i <= cap`` with ``sum(w) = total`` whose combination best matches a direction.

    ``K = G G^T`` and ``b = G t`` describe candidate gradients ``G`` and the
    target ``t``. For a target length ``alpha`` the problem
    ``min_w || G^T w - alpha * t / ||t|| ||^2`` over the capped simplex is
    solved by projected gradient; the length is unknown (only the direction
    matters), so a few lengths, multiples ``scales`` of the length of the
    uniform combination, are solved side by side and the solution with the
    highest cosine to ``t`` is returned.

    With ``total = batch size`` and ``cap = 1`` the weights are inclusion
    probabilities of a batch *without repeats*; a larger cap allows a sample to
    appear up to ``cap`` times.
    """
    n = len(b)
    step = 1.0 / torch.linalg.eigvalsh(K)[-1].clamp_min(1e-30)
    w_uniform = torch.full_like(b, total / n)
    ref = (w_uniform @ K @ w_uniform).clamp_min(1e-30).sqrt()  # length of the uniform combination
    b_hat = b / max(t_norm, 1e-30)                              # G t / ||t||
    alphas = torch.tensor(scales, device=b.device, dtype=b.dtype) * ref
    W = w_uniform.repeat(len(scales), 1)                        # one row per target length
    for _ in range(iters):
        W = project_capped_simplex(W - step * (W @ K - alphas[:, None] * b_hat[None]), total, cap)
    W = torch.cat([W, w_uniform[None]])                         # the uniform weights are always a candidate
    cosines = (W @ b_hat) / ((W @ K) * W).sum(1).clamp_min(1e-30).sqrt()
    return W[cosines.argmax()]


def systematic_counts(w: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Integer counts summing to ``n`` with expectation ``n * w / w.sum()`` (systematic resampling).

    Each count differs from its expectation by less than one, so the resulting
    multiset reproduces the weighted combination with little sampling noise.
    """
    p = w / w.sum()
    u = torch.rand(1, generator=generator).item() if generator is not None else 0.5
    edges = torch.cumsum(p, 0) * n
    edges[-1] = n  # guard against rounding
    upper = torch.floor(edges - u + 1).clamp(0, n)
    return (upper - torch.cat([upper.new_zeros(1), upper[:-1]])).long()


class GradientMatchSampler:
    """Choose clean samples whose combined gradient points along the adversarial-training gradient.

    Adversarial training updates the weights along ``t``, the gradient of the
    loss at attacked inputs. This sampler never trains on an attacked input.
    Each step it

    1. draws ``pool_mult * batch_size`` candidate training samples;
    2. measures the target direction ``t = (1 - mix) * g_clean + mix * g_adv`` on
       ``target_n`` *other* samples, where ``g_adv`` is the loss gradient at PGD
       inputs of radius ``eps`` (in 1/255). With ``target="train"`` these are
       further training samples drawn for the purpose (no held-out data
       needed); with ``target="probe"`` they come from the held-out probe
       split. Target samples are never candidates: a sample's clean gradient is
       trivially close to its own adversarial gradient, and fitting that
       would match the target's sampling noise instead of its direction;
    3. computes every candidate's clean loss gradient ``g_i`` and solves
       ``min_{w >= 0} || sum_i w_i g_i - t ||``;
    4. returns a batch in which candidate ``i`` appears about
       ``batch_size * w_i / sum(w)`` times.

    The update the trainer then makes is an ordinary step on (repeated) clean
    training samples whose mean gradient approximates the adversarial-training
    direction. Motivation and the measured quality of the approximation are in
    JOURNAL.md (2026-11-15, "gradient matching").

    ``max_repeat`` bounds how often one candidate may appear in a batch
    (``1`` = a batch of distinct samples; ``0`` = unbounded). Unbounded fits put
    most weight on a few dozen candidates; over a run the same samples are then
    trained on again and again, the rest of the data is neglected, and both the
    match and the model degrade (JOURNAL.md, 2026-11-15, run ``p7_gm/gm_e8``).

    Unlike rules that keep the top-scoring candidates by an inner product, the
    fit matches a *direction*: it is not rewarded for picking large gradients,
    which is what made earlier gradient-guided rules collapse or stall.

    Statistics logged per step (``sel/...``): ``cos_fit`` (cosine between the
    weighted combination and ``t``), ``cos_batch`` (the same for the batch
    actually returned), ``cos_uniform_batch`` (for a plain batch of the same
    size: the fair reference), ``cos_uniform`` (for the mean of all
    candidates), ``norm_ratio`` (norm of the batch's mean gradient relative to a plain
    batch's), ``eff_n`` (effective number of distinct samples in the batch).
    """

    def __init__(self, data, probe, batch_size: int, pool_mult: float = 4.0, eps: float = 8.0, pgd_steps: int = 5,
                 mix: float = 1.0, target: str = "train", target_n: int = 256, nnls_iters: int = 200,
                 max_repeat: float = 0.0, n_coords: int = 0, grad_chunk: int = 128, start_step: int = 0,
                 seed: int = 0):
        assert target in ("train", "probe")
        self.data, self.probe, self.batch_size = data, probe, batch_size
        self.n_candidates = int(round(pool_mult * batch_size))
        # with target="train" each draw holds the candidates followed by the target samples
        self.pool_size = self.n_candidates + (target_n if target == "train" else 0)
        self.eps, self.pgd_steps, self.mix = eps / 255, pgd_steps, mix
        self.target, self.target_n, self.nnls_iters, self.start_step = target, target_n, nnls_iters, start_step
        self.max_repeat = max_repeat
        # n_coords > 0: fit on that many randomly chosen gradient coordinates (fixed for the run) instead of
        # all of them, which bounds memory for large models; grad_chunk = samples per per-sample-gradient call
        self.n_coords, self.grad_chunk, self._coords = n_coords, grad_chunk, None
        self.base = UniformSampler(len(data), self.pool_size, seed=seed, device=data.device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        self._stats: dict = {}

    @staticmethod
    def _flat_grad(model, x, y):
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(F.cross_entropy(model(x), y), params)
        return torch.cat([g.reshape(-1) for g in grads])

    def target_direction(self, model, x=None, y=None) -> torch.Tensor:
        """Target direction measured on ``(x, y)``, or on a random probe mini-batch if not given."""
        if x is None:
            idx = torch.randperm(len(self.probe), generator=self.gen)[: self.target_n].to(self.probe.device)
            x, y = self.probe.batch(idx)
        g_adv = self._flat_grad(model, pgd_linf(model, x, y, self.eps, steps=self.pgd_steps, keep_best=False), y)
        if self.mix >= 1:
            return g_adv
        return (1 - self.mix) * self._flat_grad(model, x, y) + self.mix * g_adv

    def next_batch(self, step: int, model=None) -> torch.Tensor:
        draw = self.base.next_batch(step)
        pool, target_idx = draw[: self.n_candidates], draw[self.n_candidates :]
        if step < self.start_step:
            self._stats = {}
            return pool[: self.batch_size]
        x, y = self.data.batch(pool)
        was_training = model.training
        model.eval()
        t = self.target_direction(model, *self.data.batch(target_idx)) if self.target == "train" \
            else self.target_direction(model)
        if self.n_coords and self._coords is None:
            self._coords = torch.randperm(len(t), generator=self.gen)[: self.n_coords].sort().values.to(t.device)
        G = per_sample_gradient_matrix(model, x, y, chunk=self.grad_chunk, coords=self._coords)
        if self._coords is not None:
            t = t[self._coords]  # the fit and the logged cosines then live in the reduced space
        model.train(was_training)
        if self.max_repeat > 0:  # diversity constraint: no candidate more than max_repeat times in the batch
            w = capped_direction_fit(G @ G.T, G @ t, t.norm().item(), float(self.batch_size), self.max_repeat)
        else:
            w = nonneg_least_squares(G @ G.T, G @ t, iters=self.nnls_iters)
        if not torch.isfinite(w).all() or w.sum() <= 0:  # no usable fit: fall back to a uniform batch
            self._stats = {}
            return pool[: self.batch_size]
        counts = systematic_counts(w.cpu(), self.batch_size, self.gen).to(pool.device)
        cf = counts.float()
        mean_pool, fit, batch = G.mean(0), w @ G, cf @ G / self.batch_size
        cos = lambda a: (a @ t / (a.norm() * t.norm()).clamp_min(1e-30)).item()
        uniform_batch = G[: self.batch_size].mean(0)  # what a plain batch of the same size would give
        self._stats = {"sel/cos_fit": cos(fit), "sel/cos_batch": cos(batch), "sel/cos_uniform": cos(mean_pool),
                       "sel/cos_uniform_batch": cos(uniform_batch),
                       "sel/norm_ratio": (batch.norm() / uniform_batch.norm().clamp_min(1e-30)).item(),
                       "sel/eff_n": (cf.sum() ** 2 / (cf ** 2).sum()).item()}
        return torch.repeat_interleave(pool, counts)

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
    * ``mode="linr"``:  ``<g_i, grad(-R)>`` with ``R`` the mean linearised
      boundary distance of the held-out points (see
      :meth:`arorder.robustness.ProbeEvaluator.lin_radius_gradient`). ``R`` is
      unchanged by rescaling the logits, so this target cannot be satisfied by
      becoming less confident. No attack is run.
    * ``mode="linr_rob"``: the same with the clean-loss gradient projected out.
    * ``mode="linr_tilt"``: ``<g_i, unit(g_clean) + lam * unit(grad(-R))>``.

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
        assert mode in ("adv", "clean", "gap", "rob", "tilt", "linr", "linr_rob", "linr_tilt")
        assert source in ("probe", "pool")
        self.lam, self.source = lam, source
        self.probe, self.eps, self.pgd_steps = probe, eps / 255, pgd_steps
        self.probe_batch, self.refresh_every, self.mode = min(probe_batch, len(probe)), refresh_every, mode
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.direction, self._last_refresh = None, None

    def _flat_grad(self, model, x, y):
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(F.cross_entropy(model(x), y), params)
        return torch.cat([g.reshape(-1) for g in grads])

    @staticmethod
    def _linr_grad(model, x, y):
        from .robustness import ProbeEvaluator  # local import avoids a cycle at module load
        return ProbeEvaluator(x, y, [], steps=0, batch_size=len(x)).lin_radius_gradient(model)

    def refresh(self, model, x=None, y=None) -> None:
        """Recompute the scoring direction on ``(x, y)``, or on a fresh random probe mini-batch."""
        if x is None:
            idx = torch.randperm(len(self.probe), generator=self.gen)[: self.probe_batch].to(self.probe.device)
            x, y = self.probe.batch(idx)
        if self.mode == "linr":
            self.direction = self._linr_grad(model, x, y)
            return
        g_clean = self._flat_grad(model, x, y)
        if self.mode == "clean":
            self.direction = g_clean
            return
        if self.mode in ("linr_rob", "linr_tilt"):
            g_lin = self._linr_grad(model, x, y)
            if self.mode == "linr_rob":
                self.direction = g_lin - (g_lin @ g_clean) / (g_clean @ g_clean).clamp_min(1e-30) * g_clean
            else:
                self.direction = (g_clean / g_clean.norm().clamp_min(1e-30)
                                  + self.lam * g_lin / g_lin.norm().clamp_min(1e-30))
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
