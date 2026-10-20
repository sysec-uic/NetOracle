#!/usr/bin/env python3
"""Re-derive the policy-evaluation numbers from 5-rep oracle sweeps.

After scripts/resweep_reps.py finishes, every configuration in the four
family-P environments has 5 measured repetitions. This script reports, per
environment:

  - per-config mean utility, within-config SD, and the across-config vs
    within-config variance split (is the landscape real or noise?)
  - the mean-based oracle configuration and value
  - each chooser's pick (from results/policy) re-scored on rep-mean values,
    with a statistical-tie verdict vs the oracle (non-overlapping +-2 SE)

  python3 scripts/resweep_analysis.py
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from reachability import DEFAULT, load_sweeps_full  # noqa: E402

PICKS = {  # env -> chooser -> config, from results/policy (live runs)
    "bulk_r40_l0.5_b50": {"hillclimb": ("bbr", "pfifo_deep", "ample")},
    "bulk_r160_l0.5_b100": {"hillclimb": ("bbr", "pfifo_deep", "ample")},
    "mixed_r40_l2.0_b100": {"hillclimb": ("cubic", "pfifo_deep", "ample")},
    "mixed_r160_l0.1_b100": {"hillclimb": ("vegas", "pfifo_deep", "ample")},
}


def env_name(env: tuple) -> str:
    return f"{env[0]}_r{env[1]}_l{env[2]}_b{env[3]}"


def main() -> None:
    argmax_picks = {}
    hc = json.load(open(REPO / "results" / "policy" / "hillclimb.json"))
    for e in hc:
        PICKS.setdefault(e["env"], {})["hillclimb"] = tuple(e["final_config"].split("/"))

    out = []
    for env, d in sorted(load_sweeps_full().items()):
        name = env_name(env)
        if name not in PICKS:
            continue
        better = next(iter(d.values()))[1]
        stats = {}
        for cfg, (vals, _) in d.items():
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            stats[cfg] = (m, sd, len(vals))
        n_reps = statistics.median([s[2] for s in stats.values()])
        opt = (max if better == "higher" else min)(stats, key=lambda c: stats[c][0])
        om, osd, on = stats[opt]

        # variance split
        means = [s[0] for s in stats.values()]
        within = statistics.mean([s[1] ** 2 for s in stats.values()])
        across = statistics.pvariance(means)
        # chooser verdicts
        rows = {}
        for chooser, cfg in PICKS[name].items():
            if cfg not in stats:
                continue
            pm, psd, pn = stats[cfg]
            se = math.sqrt(psd ** 2 / max(pn, 1) + osd ** 2 / max(on, 1))
            diff = abs(pm - om)
            tie = diff <= 2 * se
            frac = pm / om if better == "higher" else om / pm
            rows[chooser] = {
                "config": "/".join(cfg), "mean": round(pm, 2), "sd": round(psd, 2),
                "frac_of_oracle": round(frac, 4), "tie_within_2se": bool(tie),
            }
        dm, dsd, dn = stats[DEFAULT]
        rec = {
            "env": name, "better": better, "n_reps": n_reps,
            "oracle_config": "/".join(opt),
            "oracle_mean": round(om, 2), "oracle_sd": round(osd, 2),
            "default_mean": round(dm, 2), "default_sd": round(dsd, 2),
            "across_config_var": round(across, 2),
            "mean_within_config_var": round(within, 2),
            "landscape_signal_ratio": round(across / within, 2) if within else None,
            "choosers": rows,
        }
        out.append(rec)
        print(json.dumps(rec, indent=1))
    (REPO / "results" / "policy" / "resweep_analysis.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
