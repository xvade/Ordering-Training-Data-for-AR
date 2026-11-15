"""Dataset handling: whole-dataset-on-GPU tensors, fixed splits, GPU augmentation.

Design notes
------------
* Small image datasets (CIFAR-10 is 180 MB as uint8) are kept entirely on the GPU
  as one uint8 tensor. There is no DataLoader and there are no worker processes:
  a batch is a single indexing operation. This is both the fastest option for
  small models and the friendliest to a shared cluster (no CPU workers, no
  per-image files on GPFS).
* Inputs handed to models are float32 in ``[0, 1]``, NCHW. Normalisation lives
  inside the model so that attack budgets are expressed in pixel units.
* The official training set is split once, deterministically, into

  - ``train``: samples that may be trained on,
  - ``probe``: held-out samples that *drive* measurements used for selection
    (e.g. the robustness gradient that candidate updates are scored against),
  - ``val``:   held-out samples used to *check* the effect of a selection rule
    during development.

  The official test set is never touched by any selection or tuning code; it is
  reserved for final reporting. Keeping ``probe`` and ``val`` disjoint means a
  rule that merely overfits the probe set is caught on ``val``.
"""

from __future__ import annotations

import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

#: Default location of prepared dataset files (bulky, reproducible -> scrubbed).
DEFAULT_DATA_DIR = Path("/gscratch/scrubbed/sgvtc/ordering-ar-data")


def prepare_cifar10(raw_tar: Path, out_path: Path) -> None:
    """Convert the official ``cifar-10-python.tar.gz`` into one ``.npz`` file.

    The output holds ``x_train`` (50000, 32, 32, 3) uint8, ``y_train`` (50000,)
    int64, ``x_test`` (10000, 32, 32, 3) uint8 and ``y_test`` (10000,) int64, in
    the official order. One file instead of 60k images keeps GPFS metadata load
    negligible.
    """
    xs, ys, x_test, y_test = [], [], None, None
    with tarfile.open(raw_tar, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not (name.startswith("data_batch_") or name == "test_batch"):
                continue
            # The archive is the official CIFAR-10 pickle distribution; its
            # checksum is verified by the caller before this function runs.
            batch = pickle.load(tar.extractfile(member), encoding="bytes")
            x = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            y = np.asarray(batch[b"labels"], dtype=np.int64)
            if name == "test_batch":
                x_test, y_test = x, y
            else:
                xs.append((name, x, y))
    xs.sort(key=lambda t: t[0])  # data_batch_1 .. data_batch_5, official order
    x_train = np.concatenate([t[1] for t in xs])
    y_train = np.concatenate([t[2] for t in xs])
    assert x_train.shape == (50000, 32, 32, 3) and x_test.shape == (10000, 32, 32, 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def prepare_cifar100(raw_tar: Path, out_path: Path) -> None:
    """Convert the official ``cifar-100-python.tar.gz`` into one ``.npz`` file (fine labels, 100 classes).

    Same layout as :func:`prepare_cifar10`.
    """
    arrays = {}
    with tarfile.open(raw_tar, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in ("train", "test"):
                continue
            batch = pickle.load(tar.extractfile(member), encoding="bytes")  # official archive, checksum verified by caller
            arrays[f"x_{name}"] = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            arrays[f"y_{name}"] = np.asarray(batch[b"fine_labels"], dtype=np.int64)
    assert arrays["x_train"].shape == (50000, 32, 32, 3) and arrays["x_test"].shape == (10000, 32, 32, 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)


@dataclass
class Splits:
    """Index arrays (into the official training set) for the three roles."""

    train: np.ndarray
    probe: np.ndarray
    val: np.ndarray


def make_splits(n: int, n_probe: int = 2000, n_val: int = 3000, seed: int = 0) -> Splits:
    """Deterministically split ``range(n)`` into train / probe / val.

    The split depends only on ``(n, n_probe, n_val, seed)``, so every script
    that uses the same arguments sees the same held-out samples.
    """
    perm = np.random.default_rng(seed).permutation(n)
    probe = np.sort(perm[:n_probe])
    val = np.sort(perm[n_probe : n_probe + n_val])
    train = np.sort(perm[n_probe + n_val :])
    return Splits(train=train, probe=probe, val=val)


class GPUData:
    """An image dataset held on one device as uint8 NCHW plus int64 labels."""

    def __init__(self, x_uint8_nhwc: np.ndarray, y: np.ndarray, device: torch.device | str):
        self.x = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc)).to(device).permute(0, 3, 1, 2).contiguous()
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
        self.device = self.x.device

    def __len__(self) -> int:
        return self.x.shape[0]

    def subset(self, idx: np.ndarray | torch.Tensor) -> "GPUData":
        """A new GPUData restricted to ``idx`` (copies the selected rows)."""
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        out = object.__new__(GPUData)
        out.x, out.y, out.device = self.x[idx], self.y[idx], self.device
        return out

    def batch(
        self,
        idx: torch.Tensor,
        augment: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` for ``idx`` with x float32 in [0, 1].

        With ``augment=True`` applies the standard CIFAR augmentation (4-pixel
        zero-padded random crop and horizontal flip), driven by ``generator`` so
        that a replay with the same generator state sees identical pixels.
        """
        x = self.x[idx].float().div_(255.0)
        if augment:
            x = random_crop_flip(x, generator=generator)
        return x, self.y[idx]


def random_crop_flip(x: torch.Tensor, pad: int = 4, generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-sample random crop (zero padding ``pad``) and horizontal flip, on device."""
    b, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=generator)
    j = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=generator)
    flip = torch.rand(b, device=dev, generator=generator) < 0.5
    rows = i[:, None] + torch.arange(h, device=dev)[None, :]
    cols = torch.arange(w, device=dev)[None, :].expand(b, w)
    cols = torch.where(flip[:, None], w - 1 - cols, cols) + j[:, None]
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(c, device=dev)[None, :, None, None]
    return xp[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]


@dataclass
class DataBundle:
    """Everything an experiment needs: the three training-side roles plus test."""

    train: GPUData
    probe: GPUData
    val: GPUData
    test: GPUData
    splits: Splits
    num_classes: int


def load_dataset(
    name: str,
    device: torch.device | str,
    data_dir: Path = DEFAULT_DATA_DIR,
    n_probe: int = 2000,
    n_val: int = 3000,
    split_seed: int = 0,
) -> DataBundle:
    """Load a prepared dataset (``<name>.npz`` in ``data_dir``) with the fixed train/probe/val split.

    Any dataset stored in the layout written by :func:`prepare_cifar10`
    (``x_train``, ``y_train``, ``x_test``, ``y_test``; uint8 NHWC images) works;
    ``cifar10`` and ``cifar100`` are prepared by ``scripts/prepare_cifar10.py``
    and ``scripts/prepare_cifar100.py``. ``train.x[k]`` is official training
    image ``splits.train[k]``; all logged sample indices in this project are
    positions within ``train`` unless stated otherwise.
    """
    d = np.load(Path(data_dir) / f"{name}.npz")
    full = GPUData(d["x_train"], d["y_train"], device)
    splits = make_splits(len(full), n_probe=n_probe, n_val=n_val, seed=split_seed)
    return DataBundle(
        train=full.subset(splits.train),
        probe=full.subset(splits.probe),
        val=full.subset(splits.val),
        test=GPUData(d["x_test"], d["y_test"], device),
        splits=splits,
        num_classes=int(d["y_train"].max()) + 1,
    )


def load_cifar10(device, data_dir: Path = DEFAULT_DATA_DIR, **kw) -> DataBundle:
    """CIFAR-10 with the fixed split (see :func:`load_dataset`)."""
    return load_dataset("cifar10", device, data_dir, **kw)
