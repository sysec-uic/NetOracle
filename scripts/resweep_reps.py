#!/usr/bin/env python3
"""Add repetitions 1-4 to the four family-P oracle sweeps so per-config
utilities are 5-run means instead of single draws. Content-addressed run ids
make this resumable; runs land in dataset/runs like the originals (family P
is excluded from training by construction).

Noisiest environments first so partial completion is still useful.

  sudo python3 scripts/resweep_reps.py
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "harness"))

import runner  # noqa: E402
import topology as topo  # noqa: E402
from config_space import RunConfig  # noqa: E402

RUNS = REPO / "dataset" / "runs"

CCS = ["cubic", "bbr", "reno", "vegas", "illinois"]
QDISCS = ["pfifo_deep", "pfifo_bdp", "fq_codel", "cake", "fq"]
BUFS = ["ample", "limited"]

# (workload, rtt, loss, bw), noisiest first
ENVS = [
    ("mixed", 160, 0.1, 100),
    ("mixed", 40, 2.0, 100),
    ("bulk", 160, 0.5, 100),
    ("bulk", 40, 0.5, 50),
]


def main() -> None:
    cfgs = []
    for (wl, rtt, loss, bw) in ENVS:
        for rep in range(1, 5):
            for cc in CCS:
                for qd in QDISCS:
                    for buf in BUFS:
                        cfgs.append(RunConfig("P", wl, cc, qd, rtt, loss, bw, buf, rep))
    todo = [c for c in cfgs if not (RUNS / f"{c.run_id}.json").exists()]
    print(f"{len(todo)} of {len(cfgs)} runs to do", flush=True)
    if not todo:
        return
    topo.create()
    servers = runner.Servers()
    servers.start_rr()
    try:
        for i, cfg in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] {cfg.describe()}", flush=True)
            try:
                result = runner.execute(cfg, servers)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                continue
            (RUNS / f"{cfg.run_id}.json").write_text(json.dumps(result, indent=1))
    finally:
        servers.stop_all()
        topo.destroy()


if __name__ == "__main__":
    main()
