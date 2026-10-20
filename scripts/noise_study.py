#!/usr/bin/env python3
"""Testbed repeatability study: re-execute representative cells N times and
report per-metric dispersion. Runs land in results/noise/runs (kept out of
dataset/runs so cell aggregation for training is untouched).

  sudo python3 scripts/noise_study.py --reps 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "harness"))

import runner  # noqa: E402
import topology as topo  # noqa: E402
from config_space import RunConfig  # noqa: E402

OUT = REPO / "results" / "noise"
RUNS = OUT / "runs"

# (workload, cc, qdisc, rtt, loss, bw, buf): default + oracle-best of the four
# swept environments, plus one rr and one short cell.
CELLS = [
    ("bulk", "cubic", "pfifo_deep", 40, 0.5, 50, "ample"),    # default, swept env 1
    ("bulk", "bbr", "pfifo_deep", 40, 0.5, 50, "ample"),      # oracle best, env 1
    ("bulk", "cubic", "pfifo_deep", 160, 0.5, 100, "ample"),  # default, env 2 (off-grid)
    ("bulk", "bbr", "pfifo_bdp", 160, 0.5, 100, "ample"),     # oracle best, env 2
    ("mixed", "cubic", "pfifo_deep", 40, 2.0, 100, "ample"),  # default, env 3
    ("mixed", "cubic", "fq_codel", 40, 2.0, 100, "ample"),    # oracle best, env 3
    ("mixed", "cubic", "pfifo_deep", 160, 0.1, 100, "ample"), # default, env 4 (off-grid)
    ("mixed", "bbr", "cake", 160, 0.1, 100, "ample"),         # oracle best, env 4
    ("rr", "cubic", "pfifo_deep", 80, 0.5, 25, "ample"),      # rr default
    ("short", "cubic", "pfifo_deep", 80, 1.0, 50, "ample"),   # short default
]

METRICS = {
    "bulk": [("goodput_mbps", lambda r: r["bulk"]["recv_mbps"])],
    "rr": [("rr_p99_ms", lambda r: r["rr"]["lat_p99_ms"]),
           ("rr_p50_ms", lambda r: r["rr"]["lat_p50_ms"])],
    "mixed": [("rr_p99_ms", lambda r: r["rr"]["lat_p99_ms"]),
              ("goodput_mbps", lambda r: r["bulk"]["recv_mbps"])],
    "short": [("fct_p95_ms", lambda r: r["short"]["fct_p95_ms"]),
              ("fct_p50_ms", lambda r: r["short"]["fct_p50_ms"])],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)

    cfgs = []
    for (wl, cc, qd, rtt, loss, bw, buf) in CELLS:
        for rep in range(args.reps):
            cfgs.append(RunConfig("N", wl, cc, qd, rtt, loss, bw, buf, rep))
    todo = [c for c in cfgs if not (RUNS / f"{c.run_id}.json").exists()]
    print(f"{len(todo)} of {len(cfgs)} runs to do")

    if todo:
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

    # ---- aggregate ----
    by_cell: dict[tuple, list[dict]] = {}
    for f in RUNS.glob("*.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        key = (c["workload"], c["cc"], c["qdisc"], c["rtt_ms"], c["loss_pct"],
               c["bw_mbit"], c["buf"])
        by_cell.setdefault(key, []).append(r)

    rows = []
    for key, rs in sorted(by_cell.items()):
        wl = key[0]
        for mname, getter in METRICS[wl]:
            vals = []
            for r in rs:
                try:
                    vals.append(getter(r))
                except KeyError:
                    pass
            if len(vals) < 3:
                continue
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals)
            rows.append({
                "cell": "/".join(str(k) for k in key), "metric": mname,
                "n": len(vals), "mean": round(mean, 3),
                "cv": round(sd / mean, 4) if mean else None,
                "min": round(min(vals), 3), "max": round(max(vals), 3),
                "spread_ratio": round(max(vals) / min(vals), 3) if min(vals) > 0 else None,
            })
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    cvs = sorted(r["cv"] for r in rows if r["cv"] is not None)
    print(f"cells x metrics: {len(rows)}; CV median={cvs[len(cvs)//2]:.3f} "
          f"p90={cvs[int(len(cvs)*0.9)]:.3f} max={cvs[-1]:.3f}")
    for r in rows:
        print(f"  {r['cell']:55s} {r['metric']:14s} n={r['n']} "
              f"mean={r['mean']:>10} cv={r['cv']}")


if __name__ == "__main__":
    main()
