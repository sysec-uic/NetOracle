#!/usr/bin/env python3
"""One-knob hill-climb reachability on the measured 50-config sweeps.

For each swept environment, treat the 50 configurations (5 cc x 5 qdisc x 2
buf) as a graph with one-knob edges and ask: does greedy steepest-ascent on
the TRUE measured utility, starting from the kernel default, reach the global
optimum? Also: what fraction of configurations reach it, and does a >=15%
improvement margin (the gate's) block any step on the default's path?

  python3 scripts/reachability.py
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "dataset" / "runs"

DEFAULT = ("cubic", "pfifo_deep", "ample")


def load_sweeps(reps: bool = True) -> dict:
    """Per-env {config: (value, better)}. With reps=True the value is the
    mean over all repetitions of that configuration (single-rep sweeps are
    unchanged); load_sweeps_full returns the raw per-rep lists."""
    full = load_sweeps_full()
    out = {}
    for env, d in full.items():
        agg = {}
        for cfg, (vals, better) in d.items():
            v = sum(vals) / len(vals) if reps else vals[0]
            agg[cfg] = (v, better)
        out[env] = agg
    return out


def load_sweeps_full() -> dict:
    envs: dict = defaultdict(dict)
    for f in RUNS.glob("*.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        if c["family"] != "P":
            continue
        env = (c["workload"], c["rtt_ms"], c["loss_pct"], c["bw_mbit"])
        try:
            if c["workload"] == "bulk":
                v, better = r["bulk"]["recv_mbps"], "higher"
            else:
                v, better = r["rr"]["lat_p99_ms"], "lower"
        except KeyError:
            continue
        key = (c["cc"], c["qdisc"], c["buf"])
        if key not in envs[env]:
            envs[env][key] = ([], better)
        envs[env][key][0].append(v)
    return {e: d for e, d in envs.items() if len(d) >= 50}


def neighbors(cfg: tuple, space: set) -> list[tuple]:
    out = []
    for other in space:
        if sum(a != b for a, b in zip(cfg, other)) == 1:
            out.append(other)
    return out


def climb(space: dict, start: tuple, better: str, margin: float = 0.0,
          max_steps: int = 10) -> tuple:
    cur = start
    path = [cur]
    for _ in range(max_steps):
        cv = space[cur][0]
        best, bestv = None, cv
        for nb in neighbors(cur, set(space)):
            nv = space[nb][0]
            improves = nv > bestv * (1 + margin) if better == "higher" \
                else nv < bestv * (1 - margin)
            if improves:
                best, bestv = nb, nv
        if best is None:
            break
        cur = best
        path.append(cur)
    return cur, path


def main() -> None:
    out = []
    for env, space in sorted(load_sweeps().items()):
        better = next(iter(space.values()))[1]
        opt = (max if better == "higher" else min)(space, key=lambda c: space[c][0])
        optv = space[opt][0]

        end, path = climb(space, DEFAULT, better)
        end15, path15 = climb(space, DEFAULT, better, margin=0.15)

        reach = 0
        for start in space:
            e, _ = climb(space, start, better)
            v = space[e][0]
            ok = (v >= 0.99 * optv) if better == "higher" else (v <= 1.01 * optv)
            reach += ok

        def rel(cfg):
            v = space[cfg][0]
            return v / optv if better == "higher" else optv / v

        row = {
            "env": "_".join(str(x) for x in env),
            "better": better,
            "optimum": "/".join(opt), "opt_value": round(optv, 2),
            "greedy_end": "/".join(end), "greedy_frac_of_opt": round(rel(end), 4),
            "greedy_steps": len(path) - 1,
            "gate15_end": "/".join(end15), "gate15_frac_of_opt": round(rel(end15), 4),
            "gate15_steps": len(path15) - 1,
            "starts_reaching_opt": f"{reach}/{len(space)}",
        }
        out.append(row)
        print(json.dumps(row, indent=1))
    (REPO / "results" / "policy" / "reachability.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
