#!/usr/bin/env python3
"""Headline-table columns + paired bootstrap CIs on the 218 leakage-clean
tasks.

Columns match the prediction table in the paper: counterfactual direction
accuracy (all / heldout_cond), median quantitative APE (all), within-2x rate
(heldout_cond). Multi-seed conditions (ft-*-s41/s42/s43) are aggregated
per-seed first (mean +- std across seeds), and enter bootstrap comparisons
as the per-task mean across seeds.

  python3 scripts/eval_stats.py
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "results" / "eval"
MANIFEST = json.load(open(REPO / "dataset" / "ft" / "manifest.json"))
IDS = set(MANIFEST["ft_eval_task_ids"])

N_BOOT = 10000
SEED = 7


def load(cond: str) -> dict[str, dict]:
    f = EVAL_DIR / f"{cond}.jsonl"
    rows = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["task_id"] in IDS:
            rows[r["task_id"]] = r
    return rows


def headline(rows: dict[str, dict]) -> dict:
    cf = [r for r in rows.values() if r["type"] == "counterfactual"]
    cfo = [r for r in cf if r["split"] == "heldout_cond"]
    q = [r for r in rows.values() if r["type"] == "quantitative"]
    qo = [r for r in q if r["split"] == "heldout_cond"]

    def frac(rs, key):
        vals = [r["scores"].get(key) for r in rs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    apes = [r["scores"].get("ape") for r in q]
    apes = [a for a in apes if isinstance(a, (int, float))]
    return {
        "dir_all": frac(cf, "dir_correct"),
        "dir_offgrid": frac(cfo, "dir_correct"),
        "med_ape": statistics.median(apes) if apes else None,
        "w2x_offgrid": frac(qo, "within_2x"),
        "n_cf": len(cf), "n_q": len(q),
        "invalid_q": sum(1 for r in q if not r["scores"].get("valid")),
    }


def per_task_scores(cond: str) -> dict[str, dict[str, float]]:
    """task_id -> scores for one condition."""
    out: dict[str, dict[str, float]] = {}
    for tid, r in load(cond).items():
        d = {}
        for k in ("dir_correct", "ape", "within_2x"):
            v = r["scores"].get(k)
            if isinstance(v, (int, float)):
                d[k] = v
        out[tid] = d
    return out


def boot_diff(a_seeds: list[dict], b_seeds: list[dict], tids: list[str],
              stat) -> tuple:
    """Hierarchical bootstrap CI for stat(a) - stat(b) over paired tasks.
    stat is computed per seed over the resampled task set, then averaged
    across seeds (matching the table aggregation exactly)."""
    rng = random.Random(SEED)
    common = [t for t in tids
              if all(t in s for s in a_seeds) and all(t in s for s in b_seeds)]

    def agg(seeds, sample):
        return statistics.mean(stat(s, sample) for s in seeds)

    point = agg(a_seeds, common) - agg(b_seeds, common)
    diffs = []
    for _ in range(N_BOOT):
        sample = [common[rng.randrange(len(common))] for _ in common]
        diffs.append(agg(a_seeds, sample) - agg(b_seeds, sample))
    diffs.sort()
    lo, hi = diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]
    p = sum(1 for d in diffs if d <= 0) / N_BOOT
    return point, lo, hi, min(p, 1 - p) * 2, len(common)


def stat_dir(scores: dict, sample: list[str]) -> float:
    vals = [scores[t]["dir_correct"] for t in sample if "dir_correct" in scores[t]]
    return sum(vals) / len(vals)


def stat_medape(scores: dict, sample: list[str]) -> float:
    vals = [scores[t]["ape"] for t in sample if "ape" in scores[t]]
    return statistics.median(vals)


def main() -> None:
    ft_seeds = ["ft-all-r0-s41", "ft-all-r0-s42", "ft-all-r0-s43"]
    grid_seeds = ["ft-grid-r0-s41", "ft-grid-r0-s42", "ft-grid-r0-s43"]
    singles = ["knn|L2", "ollama-gemma4-31b|L2", "gbt|L0", "gbt600|L0", "rf|L0",
               "base-qwen3.5-08b|L0"]

    print("== headline columns (218 ft-clean tasks) ==")
    print(f"{'condition':28s} dir_all dir_off medAPE w2x_off invalid_q")
    for cond in singles:
        try:
            h = headline(load(cond))
        except FileNotFoundError:
            continue
        fmt = lambda v: f"{v:.3f}" if v is not None else "  --"
        print(f"{cond:28s} {fmt(h['dir_all'])}  {fmt(h['dir_offgrid'])}  "
              f"{fmt(h['med_ape'])}  {fmt(h['w2x_offgrid'])}  {h['invalid_q']}")
    for name, seeds in (("ft-all (3 seeds)", ft_seeds), ("ft-grid (3 seeds)", grid_seeds)):
        hs = [headline(load(f"{s}|L0")) for s in seeds]
        agg = {k: (statistics.mean(h[k] for h in hs),
                   statistics.stdev(h[k] for h in hs)) for k in
               ("dir_all", "dir_offgrid", "med_ape", "w2x_offgrid")}
        print(f"{name:28s} " + "  ".join(
            f"{m:.3f}±{s:.3f}" for m, s in agg.values()))

    print("\n== paired hierarchical bootstrap (ft-all 3 seeds vs baseline) ==")
    ft = [per_task_scores(f"{s}|L0") for s in ft_seeds]
    cf_ids = sorted(t for t in IDS if t.startswith("counterfactual"))
    q_ids = sorted(t for t in IDS if t.startswith("quantitative"))
    cfo_ids = []
    knn = load("knn|L2")
    for t in cf_ids:
        if t in knn and knn[t]["split"] == "heldout_cond":
            cfo_ids.append(t)

    for base_name in ("knn|L2", "ollama-gemma4-31b|L2", "gbt|L0", "rf|L0"):
        try:
            base = [per_task_scores(base_name)]
        except FileNotFoundError:
            continue
        d = boot_diff(ft, base, cf_ids, stat_dir)
        do = boot_diff(ft, base, cfo_ids, stat_dir)
        a = boot_diff(ft, base, q_ids, stat_medape)
        print(f"vs {base_name}:")
        print(f"  dir acc (all, n={d[4]}):     +{d[0]:.3f}  95% CI [{d[1]:+.3f},{d[2]:+.3f}]  p={d[3]:.4f}")
        print(f"  dir acc (off-grid, n={do[4]}): +{do[0]:.3f}  95% CI [{do[1]:+.3f},{do[2]:+.3f}]  p={do[3]:.4f}")
        print(f"  med APE (all, n={a[4]}):     {a[0]:+.3f}  95% CI [{a[1]:+.3f},{a[2]:+.3f}]  p={a[3]:.4f}")

    out = REPO / "results" / "eval" / "stats_summary.txt"


if __name__ == "__main__":
    main()
