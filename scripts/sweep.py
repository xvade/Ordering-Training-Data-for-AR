"""Run several training variants x seeds one after another in a single job.

Usage::

    python scripts/sweep.py --out-root ROOT --seeds 0 1 2 \
        --variants base "lossLow:sampler=loss:low;pool_mult=2" "short:epochs=10"

Each variant is ``NAME`` or ``NAME:key=value;key=value``, overriding fields of
:class:`arorder.train.TrainConfig`. Runs land in ``ROOT/NAME/seed<k>``. A run
whose ``meta.json`` already exists is skipped, so an interrupted sweep can be
resubmitted as is. Small-model runs take a few minutes each, so running them
sequentially in one job is kinder to the scheduler than one job per run.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from arorder.data import load_cifar10
from arorder.train import TrainConfig, train


def parse_variant(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    types = {f.name: f.type for f in dataclasses.fields(TrainConfig)}
    defaults = TrainConfig()
    over = {}
    for item in filter(None, rest.split(";")):
        key, _, val = item.partition("=")
        if key not in types:
            raise SystemExit(f"unknown config field {key!r} in variant {spec!r}")
        cur = getattr(defaults, key)
        if isinstance(cur, bool):
            over[key] = val.lower() in ("1", "true", "yes")
        elif isinstance(cur, list):
            over[key] = [float(v) for v in val.split(",")]
        else:
            over[key] = type(cur)(val)
    return name, over


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--common", default="", help="overrides applied to every variant, same syntax")
    a = p.parse_args()
    data = load_cifar10(torch.device("cuda"))
    _, common = parse_variant("common:" + a.common)
    for spec in a.variants:
        name, over = parse_variant(spec)
        for seed in a.seeds:
            run_dir = a.out_root / name / f"seed{seed}"
            if (run_dir / "meta.json").exists():
                print(f"skip {run_dir} (done)", flush=True)
                continue
            cfg = TrainConfig(**{**common, **over, "seed": seed, "run_dir": str(run_dir)})
            print(f"=== {name} seed {seed}: {json.dumps({**common, **over})}", flush=True)
            train(cfg, data=data)
