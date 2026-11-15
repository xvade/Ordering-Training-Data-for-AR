"""Batch samplers: objects that decide which training samples form each update.

The trainer calls ``sampler.next_batch(step, model)`` once per optimiser step
and trains on whatever positions (within the train split) come back. This is the
single seam through which a data-ordering method acts, so a selection rule is
independent of the model, the loss and the dataset.
"""

from __future__ import annotations

import torch


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
