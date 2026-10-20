#!/usr/bin/env python3
"""Runtime daemon prototype: online, gated reconfiguration of a live workload.

A recurring bulk workload (10s epochs, e.g. periodic backups/syncs) runs over
a lossy path starting from the kernel-default stack config. Between epochs
the daemon uses the epoch's measured stats as the anchor, asks the world
model for a confidence-gated one-knob improvement (same gate as
policy_hillclimb), applies it, and keeps monitoring. Output: per-epoch
timeline with wall-clock — time-to-better-config and achieved goodput.

  python3 scripts/daemon_demo.py --epochs 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "harness"))

import runner  # noqa: E402
import topology as topo  # noqa: E402
from config_space import RunConfig  # noqa: E402
from data import PRIMARY_METRIC, _run_metrics  # noqa: E402
from policy_eval import DEFAULT  # noqa: E402
from policy_hillclimb import (  # noqa: E402
    anchor_cell, gated_best, neighbors, predict_neighbors,
)
from flywheel_v2 import knobs_of  # noqa: E402

OUT_DIR = HERE.parent / "results" / "policy"

# demo environment: lossy mid-RTT path where the default collapses
ENV = ("bulk", 40, 0.5, 50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    wl, rtt, loss, bw = ENV
    metric, direction = PRIMARY_METRIC[wl]
    cur = RunConfig("P", wl, DEFAULT[0], DEFAULT[1], rtt, loss, bw, DEFAULT[2])
    visited = {knobs_of(cur.describe())}
    converged = False

    trace = []
    t0 = time.time()
    topo.create()
    servers = runner.Servers()
    servers.start_rr()
    try:
        for epoch in range(args.epochs):
            result = runner.execute(cur, servers)  # one 10s workload epoch
            val = _run_metrics(result).get(metric)
            now = round(time.time() - t0, 1)
            row = {"epoch": epoch, "t_s": now,
                   "config": f"{cur.cc}/{cur.qdisc}/{cur.buf}", metric: val}
            print(f"[t={now:6.1f}s] epoch {epoch}: "
                  f"{row['config']:28s} {metric}={val}", flush=True)

            if not converged:
                anchor = anchor_cell(cur, result, metric)
                nbrs = [(k, nb) for k, nb in neighbors(cur)
                        if knobs_of(nb.describe()) not in visited]
                preds = predict_neighbors(anchor, nbrs, metric,
                                          f"daemon_e{epoch}")
                nxt, gain = gated_best(nbrs, preds, direction)
                if nxt is not None:
                    print(f"          daemon: move -> "
                          f"{nxt.cc}/{nxt.qdisc}/{nxt.buf} "
                          f"(predicted gain {gain:.2f}x)", flush=True)
                    row["move"] = f"{nxt.cc}/{nxt.qdisc}/{nxt.buf}"
                    visited.add(knobs_of(nxt.describe()))
                    cur = nxt
                else:
                    print("          daemon: no confident improvement "
                          "-> converged, monitoring only", flush=True)
                    converged = True
            trace.append(row)
    finally:
        servers.stop_all()
        topo.destroy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "daemon_trace.json").write_text(json.dumps(trace, indent=1))
    base = trace[0][metric]
    final = trace[-1][metric]
    print(f"\nDAEMON DONE: {base} -> {final} {metric} "
          f"({final / base:.1f}x) in {trace[-1]['t_s']}s", flush=True)


if __name__ == "__main__":
    main()
