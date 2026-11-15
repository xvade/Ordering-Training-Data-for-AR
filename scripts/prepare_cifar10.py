"""Verify the downloaded CIFAR-10 archive and convert it to a single .npz file.

Usage: python scripts/prepare_cifar10.py [raw_tar] [out_npz]
"""
import hashlib
import sys
from pathlib import Path

from arorder.data import DEFAULT_DATA_DIR, prepare_cifar10

MD5 = "c58f30108f718f92721af3b95e74349a"  # published checksum of cifar-10-python.tar.gz

raw = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR / "raw" / "cifar-10-python.tar.gz"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DATA_DIR / "cifar10.npz"
digest = hashlib.md5(raw.read_bytes()).hexdigest()
if digest != MD5:
    sys.exit(f"checksum mismatch for {raw}: got {digest}, expected {MD5}")
prepare_cifar10(raw, out)
print(f"wrote {out}")
