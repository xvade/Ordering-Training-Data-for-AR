"""L-infinity projected gradient descent (PGD), written for paired measurements.

The attack is fully deterministic given its inputs: the random start is passed
in explicitly as a tensor (``init``), not drawn inside. Evaluating two sets of
weights with the same ``init`` therefore gives a *paired* comparison in which
the only thing that changed is the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Classification margin ``z_y - max_{j != y} z_j`` per sample (positive iff correct)."""
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), float("-inf")).amax(dim=1)
    return true - other


def pgd_linf(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 10,
    alpha: float | None = None,
    init: torch.Tensor | None = None,
    loss: str = "ce",
    keep_best: bool = True,
) -> torch.Tensor:
    """Return adversarial inputs within an L-inf ball of radius ``eps`` around ``x``.

    Args:
        model: classifier taking inputs in ``[0, 1]``. Its mode (train/eval) is
            left as the caller set it.
        eps: radius in pixel units (e.g. ``2/255``). ``eps == 0`` returns ``x``.
        steps: number of signed-gradient steps.
        alpha: step size; defaults to ``2.5 * eps / steps``.
        init: optional tensor shaped like ``x`` with entries in ``[-1, 1]``; the
            start point is ``x + eps * init``. ``None`` starts at ``x``.
        loss: ``"ce"`` (cross-entropy) or ``"margin"`` (minimise the margin
            directly; stronger when cross-entropy gradients vanish).
        keep_best: if True, return for each sample the iterate with the lowest
            margin seen (including the start), not simply the last iterate.
    """
    if eps <= 0:
        return x
    alpha = 2.5 * eps / steps if alpha is None else alpha
    delta = torch.zeros_like(x) if init is None else init * eps
    delta = (x + delta).clamp_(0, 1) - x
    best_delta = delta.clone()
    best_margin = torch.full((x.shape[0],), float("inf"), device=x.device)
    for it in range(steps + 1):
        delta.requires_grad_(True)
        logits = model(x + delta)
        m = margin(logits, y)
        if keep_best:
            better = m.detach() < best_margin
            best_margin = torch.where(better, m.detach(), best_margin)
            best_delta[better] = delta.detach()[better]
        if it == steps:
            break
        obj = -m.sum() if loss == "margin" else F.cross_entropy(logits, y, reduction="sum")
        (g,) = torch.autograd.grad(obj, delta)
        delta = (delta.detach() + alpha * g.sign()).clamp_(-eps, eps)
        delta = (x + delta).clamp_(0, 1) - x
    return (x + (best_delta if keep_best else delta.detach())).detach()
