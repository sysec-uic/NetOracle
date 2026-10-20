#!/usr/bin/env python3
"""Densify the training grid with a broad random sample of novel configs.

Family E extends the original matrix with new rtt/loss/bw values while
avoiding the held-out eval conditions (westwood, rtt=160, bw=400).

  python3 scripts/collect_grid.py --n 400 --seed 3
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flywheel_v2 import knobs_of, run_testbed  # noqa: E402
from config_space import RunConfig  # noqa: E402
from data import load_cells  # noqa: E402

RTTS = [10, 20, 40, 80, 120]
LOSSES = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
BWS = [25, 50, 100, 200]
CCS = ["cubic", "bbr", "reno", "vegas", "illinois"]  # no westwood (heldout)
QDISCS = ["pfifo_deep", "pfifo_bdp", "fq_codel", "cake", "fq"]
BUFS = ["ample", "limited"]
WORKLOADS = ["bulk", "rr", "mixed", "short"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--family", default="E")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    existing = {knobs_of(c) for c in load_cells()}

    space = list(itertools.product(WORKLOADS, CCS, QDISCS, RTTS, LOSSES,
                                   BWS, BUFS))
    rng.shuffle(space)
    selected = []
    for wl, cc, qd, rtt, loss, bw, buf in space:
        cfg = RunConfig(args.family, wl, cc, qd, rtt, loss, bw, buf)
        if knobs_of(cfg.describe()) in existing:
            continue
        selected.append(cfg)
        if len(selected) >= args.n:
            break
    print(f"collecting {len(selected)} novel configs (family {args.family})",
          flush=True)
    results = run_testbed(selected)
    print(f"DONE: {len(results)}/{len(selected)} runs succeeded", flush=True)


if __name__ == "__main__":
    main()
