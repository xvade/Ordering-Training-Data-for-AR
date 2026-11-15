"""Verify the downloaded CIFAR-100 archive and convert it to a single .npz file.

Usage: python scripts/prepare_cifar100.py [raw_tar] [out_npz]
"""
import hashlib
import sys
from pathlib import Path

from arorder.data import DEFAULT_DATA_DIR, prepare_cifar100

MD5 = "eb9058c3a382ffc7106e4002c42a8d85"  # published checksum of cifar-100-python.tar.gz

raw = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR / "raw" / "cifar-100-python.tar.gz"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DATA_DIR / "cifar100.npz"
digest = hashlib.md5(raw.read_bytes()).hexdigest()
if digest != MD5:
    sys.exit(f"checksum mismatch for {raw}: got {digest}, expected {MD5}")
prepare_cifar100(raw, out)
print(f"wrote {out}")
