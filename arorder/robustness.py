"""Robustness measurement on a fixed set of held-out points.

Why several measures
--------------------
A network trained without adversarial examples has essentially zero accuracy
under the usual 8/255 attack, so "robust accuracy at 8/255" cannot show which
updates make things worse. :class:`ProbeEvaluator` therefore reports measures
that stay informative for weakly robust models:

* accuracy, mean loss and mean margin under PGD at several small radii;
* ``auc``: robust accuracy integrated over the radius grid, i.e. the mean
  (truncated) attack radius the model withstands, in units of 1/255;
* ``lin_radius``: a first-order estimate of each point's distance to the
  decision boundary, ``margin / ||d margin / d x||_1`` (the L-inf radius at
  which the linearised margin reaches zero), in units of 1/255. It is smooth in
  the weights, needs one backward pass, and involves no attack randomness.

Every measure is reported alongside its clean counterpart, because a model that
simply classifies fewer points correctly is not "less robust" in the sense we
care about. ``robust_given_correct`` isolates robustness from accuracy.

Paired evaluation
-----------------
The PGD random starts are drawn once in the constructor and reused for every
call. Two models evaluated by the same ``ProbeEvaluator`` are attacked from the
same start points, so differences between them are not attack-seed noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attacks import margin, pgd_linf


class ProbeEvaluator:
    """Evaluates clean and adversarial behaviour of a model on fixed points.

    Args:
        x: float inputs in ``[0, 1]``, already on the evaluation device.
        y: int64 labels.
        eps_list: attack radii in pixel units (e.g. ``[1/255, 2/255]``).
        steps: PGD steps per radius.
        seed: seed for the fixed random starts.
        batch_size: evaluation batch size.
        loss: PGD objective, ``"ce"`` or ``"margin"``.
    """

    def __init__(self, x, y, eps_list, steps: int = 10, seed: int = 0, batch_size: int = 1000,
                 loss: str = "ce"):
        self.x, self.y = x, y
        self.eps_list = [float(e) for e in eps_list]
        self.steps, self.batch_size, self.loss = steps, batch_size, loss
        g = torch.Generator(device=x.device).manual_seed(seed)
        self.init = torch.rand(x.shape, device=x.device, generator=g) * 2 - 1

    def __len__(self):
        return self.x.shape[0]

    def _batches(self):
        for s in range(0, len(self), self.batch_size):
            sl = slice(s, s + self.batch_size)
            yield self.x[sl], self.y[sl], self.init[sl]

    def adversarial_inputs(self, model, eps: float) -> torch.Tensor:
        """PGD inputs for every probe point at radius ``eps`` (fixed starts)."""
        return torch.cat([
            pgd_linf(model, x, y, eps, steps=self.steps, init=init, loss=self.loss)
            for x, y, init in self._batches()
        ])

    @torch.no_grad()
    def _stats(self, model, x):
        logits = torch.cat([model(x[s : s + self.batch_size]) for s in range(0, len(self), self.batch_size)])
        return F.cross_entropy(logits, self.y, reduction="none"), margin(logits, self.y)

    def linearized_radius(self, model) -> torch.Tensor:
        """Per-point first-order L-inf distance to the boundary, in pixel units.

        Negative for misclassified points (their margin is negative).
        """
        out = []
        for x, y, _ in self._batches():
            x = x.clone().requires_grad_(True)
            m = margin(model(x), y)
            (g,) = torch.autograd.grad(m.sum(), x)
            out.append(m.detach() / g.abs().flatten(1).sum(1).clamp_min(1e-12))
        return torch.cat(out)

    def evaluate(self, model, per_example: bool = False, per_class: bool = False) -> dict:
        """Return a dict of scalar metrics (and per-example tensors if asked).

        Keys: ``clean_loss``, ``clean_acc``, ``clean_margin``, ``lin_radius``
        (mean over points of the linearised radius clipped to [0, 16/255], in
        1/255 units), and for each radius ``e`` (formatted in 1/255 units, e.g.
        ``adv_acc@2``): ``adv_loss@e``, ``adv_acc@e``, ``adv_margin@e``,
        ``robust_given_correct@e``; plus ``auc`` when more than one radius is
        configured.
        """
        was_training = model.training
        model.eval()
        res, ex = {}, {}
        loss, m = self._stats(model, self.x)
        res.update(clean_loss=loss.mean().item(), clean_acc=(m > 0).float().mean().item(),
                   clean_margin=m.mean().item())
        ex["clean_margin"] = m
        r = self.linearized_radius(model) * 255
        res["lin_radius"] = r.clamp(0, 16).mean().item()
        ex["lin_radius"] = r
        accs = []
        for eps in self.eps_list:
            tag = f"{eps * 255:g}"
            la, ma = self._stats(model, self.adversarial_inputs(model, eps))
            acc = (ma > 0).float().mean().item()
            res[f"adv_loss@{tag}"] = la.mean().item()
            res[f"adv_acc@{tag}"] = acc
            res[f"adv_margin@{tag}"] = ma.mean().item()
            res[f"robust_given_correct@{tag}"] = acc / max(res["clean_acc"], 1e-12)
            ex[f"adv_margin@{tag}"] = ma
            accs.append(acc)
        if len(self.eps_list) > 1:
            # Trapezoidal integral of robust accuracy over radius, from 0 (clean
            # accuracy) to the largest radius, in 1/255 units.
            radii = [0.0] + [e * 255 for e in self.eps_list]
            vals = [res["clean_acc"]] + accs
            res["auc"] = sum((radii[i + 1] - radii[i]) * (vals[i + 1] + vals[i]) / 2 for i in range(len(accs)))
        model.train(was_training)
        if per_class:  # lists indexed by class: clean accuracy and accuracy under attack at each radius
            classes = range(int(self.y.max().item()) + 1)
            for key, m_ in ex.items():
                if key == "lin_radius":
                    continue
                name = "class_clean_acc" if key == "clean_margin" else "class_adv_acc@" + key.split("@")[1]
                res[name] = [(m_[self.y == c] > 0).float().mean().item() for c in classes]
        if per_example:
            res["per_example"] = ex
        return res

    def lin_radius_gradient(self, model, cap: float = 16.0) -> torch.Tensor:
        """Gradient w.r.t. the weights of ``-mean(linearised radius)`` (1/255 units).

        The linearised radius ``margin / ||d margin / d x||_1`` does not change
        when all logits are rescaled, so unlike a loss it cannot be improved by
        becoming less confident. Radii are clipped to ``[-cap, cap]``.
        Differentiating it needs a second backward pass through the input
        gradient. Returned as a "loss" gradient: stepping against it raises the
        mean radius.
        """
        was_training = model.training
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        total = [torch.zeros_like(p) for p in params]
        for x, y, _ in self._batches():
            x = x.clone().requires_grad_(True)
            m = margin(model(x), y)
            (g,) = torch.autograd.grad(m.sum(), x, create_graph=True)
            r = 255 * m / g.abs().flatten(1).sum(1).clamp_min(1e-12)
            for t, gr in zip(total, torch.autograd.grad(-r.clamp(-cap, cap).sum(), params)):
                t += gr
        model.train(was_training)
        return torch.cat([t.reshape(-1) for t in total]) / len(self)

    def loss_gradient(self, model, eps: float, kind: str = "ce", temperature: float = 1.0) -> torch.Tensor:
        """Gradient w.r.t. the weights of the mean probe loss at radius ``eps``.

        By Danskin's theorem the gradient of the worst-case loss is the gradient
        of the loss at the (fixed) worst-case inputs, so this runs PGD and then
        differentiates the loss at the adversarial points. ``eps = 0`` gives the
        clean-loss gradient. ``kind`` is ``"ce"``, ``"neg_margin"`` (mean of
        ``-margin`` clipped to at most 10, so a few very confident points do not
        dominate) or ``"serr"``: a smoothed error rate,
        ``mean(sigmoid(-margin / temperature))``. With ``temperature``
        proportional to the typical margin, ``serr`` is insensitive to logit
        scale and is driven by points near the decision boundary, i.e. it tracks
        (robust) accuracy rather than confidence.

        Returns one flat vector in ``named_parameters`` order.
        """
        was_training = model.training
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        total = [torch.zeros_like(p) for p in params]
        for x, y, init in self._batches():
            xa = pgd_linf(model, x, y, eps, steps=self.steps, init=init, loss=self.loss)
            logits = model(xa)
            if kind == "ce":
                obj = F.cross_entropy(logits, y, reduction="sum")
            elif kind == "serr":
                obj = torch.sigmoid(-margin(logits, y) / temperature).sum()
            else:
                obj = (-margin(logits, y).clamp(max=10.0)).sum()
            for t, g in zip(total, torch.autograd.grad(obj, params)):
                t += g
        model.train(was_training)
        return torch.cat([t.reshape(-1) for t in total]) / len(self)
