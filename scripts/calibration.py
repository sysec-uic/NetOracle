#!/usr/bin/env python3
"""Calibration of sampling-spread uncertainty.

Two independent datasets, no new inference:

A. Gate-replay counterfactuals: 1,800 (anchor, target) pairs over the four
   50-config sweeps, K=10 sampled ratio predictions each
   (results/policy/{abl_prompts,gen_abl}.jsonl), truth from the family-P
   measured runs. In-grid envs (rtt 40) vs off-grid (rtt 160) reported
   separately.

B. Flywheel acquisition log: 192 candidate configs with recorded
   uncertainty (std of K=5 sampled log-values), model prediction, and
   measured outcome (results/flywheel/rounds.jsonl). The random strategy's
   96 rows are an unbiased sample of the candidate pool; the uncertainty
   strategy's rows are top-of-pool by construction.

Statistics: Spearman rho (spread vs log error), AUROC (spread predicting
rel err > 20%), and empirical coverage of the sample [q10,q90] interval
(nominal 80%).

  python3 scripts/calibration.py
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

REPO = pathlib.Path(__file__).resolve().parent.parent
POL = REPO / "results" / "policy"
FLY = REPO / "results" / "flywheel"
RUNS = REPO / "dataset" / "runs"


def auroc(scores: list[float], labels: list[bool]) -> float:
    pairs = [(s, l) for s, l in zip(scores, labels)]
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def load_p_cells() -> dict[str, dict]:
    cells = {}
    for f in RUNS.glob("*.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        if c["family"] != "P":
            continue
        try:
            v = (r["bulk"]["recv_mbps"] if c["workload"] == "bulk"
                 else r["rr"]["lat_p99_ms"])
        except KeyError:
            continue
        cells[c["run_id"]] = {"value": v, "rtt": c["rtt_ms"],
                              "workload": c["workload"]}
    return cells


def part_a() -> dict:
    cells = load_p_cells()
    gens = {}
    for line in open(POL / "gen_abl.jsonl"):
        r = json.loads(line)
        gens[r["task_id"]] = r.get("texts") or [r["text"]]

    groups = defaultdict(list)
    for tid, texts in gens.items():
        a_id, b_id = tid.split("|")
        if a_id not in cells or b_id not in cells:
            continue
        ratios = []
        for t in texts:
            try:
                v = json.loads(t)
                r = float(v["ratio"])
                if r > 0:
                    ratios.append(math.log10(r))
            except Exception:
                pass
        if len(ratios) < 3:
            continue
        truth = math.log10(cells[b_id]["value"] / cells[a_id]["value"])
        med = statistics.median(ratios)
        spread = statistics.stdev(ratios)
        err = abs(med - truth)
        # measured deviates >20% from the (median) prediction
        surprise = abs(10 ** (truth - med) - 1) > 0.2
        q10, q90 = np.percentile(ratios, [10, 90])
        covered = q10 <= truth <= q90
        grid = "in-grid" if cells[a_id]["rtt"] == 40 else "off-grid"
        groups[grid].append((spread, err, surprise, covered))
        groups["all"].append((spread, err, surprise, covered))

    out = {}
    for g, rows in sorted(groups.items()):
        spreads = [r[0] for r in rows]
        errs = [r[1] for r in rows]
        rho, pval = spearmanr(spreads, errs)
        out[f"A/{g}"] = {
            "n": len(rows),
            "spearman_rho": round(float(rho), 3),
            "spearman_p": float(f"{pval:.2g}"),
            "auroc_err_gt20pct": round(auroc(spreads, [r[2] for r in rows]), 3),
            "surprise_rate": round(sum(r[2] for r in rows) / len(rows), 3),
            "coverage_q10_q90": round(sum(r[3] for r in rows) / len(rows), 3),
        }
    return out


def part_b() -> dict:
    rows = [json.loads(l) for l in open(FLY / "rounds.jsonl")]
    out = {}
    for strat in ("random", "uncertainty"):
        rs = [r for r in rows if r["strategy"] == strat
              and isinstance(r.get("uncertainty"), (int, float))
              and isinstance(r.get("rel_err"), (int, float))]
        us = [r["uncertainty"] for r in rs]
        es = [math.log10(1 + r["rel_err"]) for r in rs]
        rho, pval = spearmanr(us, es)
        out[f"B/{strat}"] = {
            "n": len(rs),
            "spearman_rho": round(float(rho), 3),
            "spearman_p": float(f"{pval:.2g}"),
            "auroc_err_gt20pct": round(
                auroc(us, [r["rel_err"] > 0.2 for r in rs]), 3),
            "surprise_rate": round(
                sum(r["rel_err"] > 0.2 for r in rs) / len(rs), 3),
        }
    return out


def main() -> None:
    res = {}
    res.update(part_a())
    res.update(part_b())
    for k, v in res.items():
        print(k, json.dumps(v))
    (REPO / "results" / "eval" / "calibration.json").write_text(
        json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
