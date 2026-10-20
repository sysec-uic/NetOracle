#!/usr/bin/env python3
"""Replay online-search baselines on the measured 50-configuration sweeps.

Every method sees the same decision space and pays one 'measurement' per
configuration it visits; measured values come from the family-P oracle
sweeps, so the comparison is exact and reproducible. Methods:

  random    expected best-of-k (closed form over the empirical distribution)
  gp-ei     Bayesian optimization: GP over one-hot features, expected
            improvement, 3 random inits (CherryPick-style), 200 seeds
  measured  hill-climb steepest ascent measuring all one-knob neighbors
            (no model; prices what the world model's predictions replace)
  netoracle actual gated hill-climb result (from results/policy)

Reports measurements-to-99%-of-oracle and best-so-far at fixed budgets,
plus exposure (worst configuration visited) for the latency environments.

  python3 scripts/bo_replay.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from reachability import load_sweeps, DEFAULT, neighbors, climb  # noqa: E402

N_SEEDS = 200
BUDGETS = (3, 5, 10, 20)

CCS = ["cubic", "bbr", "reno", "vegas", "illinois"]
QDISCS = ["pfifo_deep", "pfifo_bdp", "fq_codel", "cake", "fq"]
BUFS = ["ample", "limited"]


def feat(cfg: tuple) -> list[float]:
    return ([1.0 if cfg[0] == c else 0.0 for c in CCS]
            + [1.0 if cfg[1] == q else 0.0 for q in QDISCS]
            + [1.0 if cfg[2] == b else 0.0 for b in BUFS])


def expected_best_of_k(values: list[float], k: int, better: str) -> float:
    """Expectation of the best of k draws without replacement."""
    vals = sorted(values, reverse=(better == "higher"))
    n = len(vals)
    total = 0.0
    for i, v in enumerate(vals):
        # P(best draw is vals[i]) = C(n-1-i, k-1)/C(n, k)
        total += v * math.comb(n - 1 - i, k - 1) / math.comb(n, k)
    return total


def gp_ei_run(space: dict, better: str, seed: int, budget: int) -> list[tuple]:
    """One BO trajectory; returns list of visited configs."""
    rng = np.random.RandomState(seed)
    configs = sorted(space)
    X = np.array([feat(c) for c in configs])
    sign = 1.0 if better == "higher" else -1.0
    y_all = {c: sign * math.log10(space[c][0]) for c in configs}

    visited = list(rng.choice(len(configs), size=3, replace=False))
    while len(visited) < budget:
        Xv = X[visited]
        yv = np.array([y_all[configs[i]] for i in visited])
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-4, normalize_y=True,
            random_state=seed)
        gp.fit(Xv, yv)
        mu, sd = gp.predict(X, return_std=True)
        best = yv.max()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - best) / sd
            ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        ei[np.array(visited)] = -np.inf
        visited.append(int(np.argmax(ei)))
    return [configs[i] for i in visited]


def measured_climb(space: dict, better: str) -> list[tuple]:
    """Steepest-ascent hill climb where every neighbor is measured."""
    visited = [DEFAULT]
    cur = DEFAULT
    while True:
        cv = space[cur][0]
        nbs = neighbors(cur, set(space))
        for nb in nbs:
            if nb not in visited:
                visited.append(nb)
        best = min(nbs, key=lambda c: space[c][0]) if better == "lower" \
            else max(nbs, key=lambda c: space[c][0])
        bv = space[best][0]
        if (better == "higher" and bv <= cv) or (better == "lower" and bv >= cv):
            break
        cur = best
    return visited


def main() -> None:
    hc = {r["env"]: r for r in json.load(open(REPO / "results" / "policy" / "budget_baseline.json"))}
    out = []
    for env, space in sorted(load_sweeps().items()):
        better = next(iter(space.values()))[1]
        env_name = f"{env[0]}_r{env[1]}_l{env[2]}_b{env[3]}"
        values = [v for v, _ in space.values()]
        opt = max(values) if better == "higher" else min(values)
        thresh = 0.99 * opt if better == "higher" else opt / 0.99

        def frac(v):
            return v / opt if better == "higher" else opt / v

        row = {"env": env_name, "better": better, "oracle": round(opt, 2)}

        # random: expected best-of-k
        for k in BUDGETS:
            row[f"random_k{k}"] = round(frac(expected_best_of_k(values, k, better)), 4)

        # GP-EI over seeds
        budget = max(BUDGETS)
        to99, at_budget, worst_seen = [], {k: [] for k in BUDGETS}, []
        for seed in range(N_SEEDS):
            traj = gp_ei_run(space, better, seed, budget)
            vals = [space[c][0] for c in traj]
            bests = []
            cur = vals[0]
            for v in vals:
                cur = max(cur, v) if better == "higher" else min(cur, v)
                bests.append(cur)
            hit = next((i + 1 for i, b in enumerate(bests)
                        if (b >= thresh if better == "higher" else b <= thresh)),
                       None)
            to99.append(hit if hit is not None else budget + 1)
            for k in BUDGETS:
                at_budget[k].append(frac(bests[k - 1]))
            worst_seen.append(max(vals) if better == "lower" else min(vals))
        to99.sort()
        row["gpei_meas_to_99pct_median"] = to99[N_SEEDS // 2]
        row["gpei_meas_to_99pct_p90"] = to99[int(N_SEEDS * 0.9)]
        for k in BUDGETS:
            row[f"gpei_k{k}"] = round(float(np.mean(at_budget[k])), 4)
        row["gpei_worst_visited_mean"] = round(float(np.mean(worst_seen)), 2)
        row["gpei_worst_visited_p90"] = round(
            float(np.percentile(worst_seen, 90 if better == "lower" else 10)), 2)

        # measured hill-climb (no model)
        mtraj = measured_climb(space, better)
        mvals = [space[c][0] for c in mtraj]
        mbest = max(mvals) if better == "higher" else min(mvals)
        row["measured_climb_meas"] = len(mtraj)
        row["measured_climb_frac"] = round(frac(mbest), 4)

        # NetOracle actual
        h = hc.get(env_name)
        if h:
            row["netoracle_meas"] = h["hillclimb_measurements"]
            row["netoracle_frac"] = round(frac(h["hillclimb"]), 4)
        # exposure of default (for reference)
        row["default_frac"] = round(frac(space[DEFAULT][0]), 4)
        worst_cfg_val = max(values) if better == "lower" else min(values)
        row["space_worst_frac"] = round(frac(worst_cfg_val), 4)

        out.append(row)
        print(json.dumps(row, indent=1))
    (REPO / "results" / "policy" / "bo_replay.json").write_text(
        json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
