"""Small image classifiers for fast iteration.

All models take float inputs in ``[0, 1]`` and normalise internally, so attack
budgets are in pixel units regardless of the dataset statistics.

Normalisation layers
--------------------
``norm`` selects the normalisation layer:

* ``"gn"`` (default for investigations): GroupNorm. It has no running
  statistics and no train/eval difference, so the effect of one optimiser step
  is exactly the change in the weights. That makes per-update attribution clean.
* ``"bn"``: BatchNorm, the common choice in practice. A training step then also
  moves the running statistics, and each sample's output depends on its batch.
* ``"none"``: no normalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "gn":
        return nn.GroupNorm(min(8, channels), channels)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


class _InputNorm(nn.Module):
    """Fixed per-channel standardisation, stored as buffers (not trained)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class SmallCNN(nn.Module):
    """Five 3x3 conv layers, global average pooling, linear head.

    With ``width=32`` this has about 0.28 M parameters and reaches roughly 85%
    clean CIFAR-10 accuracy with standard augmentation, in a few seconds per
    epoch on one GPU.
    """

    def __init__(self, num_classes: int = 10, width: int = 32, norm: str = "gn",
                 mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        w = width

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=(norm == "none")),
                _norm(norm, cout),
                nn.ReLU(inplace=False),
            )

        self.input_norm = _InputNorm(mean, std)
        self.features = nn.Sequential(
            block(3, w, 1),
            block(w, 2 * w, 2),
            block(2 * w, 2 * w, 1),
            block(2 * w, 4 * w, 2),
            block(4 * w, 4 * w, 1),
        )
        self.head = nn.Linear(4 * w, num_classes)

    def forward(self, x):
        h = self.features(self.input_norm(x))
        return self.head(h.mean(dim=(2, 3)))


class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = _norm(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _norm(norm, cout)
        self.relu = nn.ReLU(inplace=False)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), _norm(norm, cout))

    def forward(self, x):
        out = self.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return self.relu(out + (x if self.shortcut is None else self.shortcut(x)))


class SmallResNet(nn.Module):
    """A CIFAR-style ResNet with ``blocks_per_stage`` basic blocks in each of 3 stages.

    ``blocks_per_stage=1, width=16`` is ResNet-8 (about 0.08 M parameters);
    ``blocks_per_stage=3, width=16`` is ResNet-20 (about 0.27 M).
    """

    def __init__(self, num_classes: int = 10, width: int = 16, blocks_per_stage: int = 2,
                 norm: str = "gn", mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.input_norm = _InputNorm(mean, std)
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False), _norm(norm, width), nn.ReLU())
        layers, cin = [], width
        for stage, cout in enumerate((width, 2 * width, 4 * width)):
            for b in range(blocks_per_stage):
                layers.append(_BasicBlock(cin, cout, 2 if (stage > 0 and b == 0) else 1, norm))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(cin, num_classes)

    def forward(self, x):
        h = self.layers(self.stem(self.input_norm(x)))
        return self.head(h.mean(dim=(2, 3)))


def build_model(name: str, num_classes: int = 10, width: int | None = None, norm: str = "gn", **kw) -> nn.Module:
    """Construct a model by name. Known names: ``smallcnn``, ``resnet8``, ``resnet14``, ``resnet20``."""
    if name == "smallcnn":
        return SmallCNN(num_classes, width=width or 32, norm=norm, **kw)
    if name in ("resnet8", "resnet14", "resnet20"):
        blocks = {"resnet8": 1, "resnet14": 2, "resnet20": 3}[name]
        return SmallResNet(num_classes, width=width or 16, blocks_per_stage=blocks, norm=norm, **kw)
    raise ValueError(f"unknown model {name!r}")


def flat_params(model: nn.Module) -> torch.Tensor:
    """All trainable parameters as one detached vector, in ``named_parameters`` order."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def set_flat_params(model: nn.Module, vec: torch.Tensor) -> None:
    """Inverse of :func:`flat_params`: overwrite trainable parameters from ``vec``."""
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(vec[i : i + n].view_as(p))
            i += n
    assert i == vec.numel()
