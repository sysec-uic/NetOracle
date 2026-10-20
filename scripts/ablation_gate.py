#!/usr/bin/env python3
"""Gate-parameter sensitivity via measured replay.

The four oracle-swept environments have ALL 50 configurations measured, so a
hill-climb under any gate setting can be replayed offline: anchors and
outcomes come from measured runs; only the model's counterfactual samples
are needed, and those are generated once (K=10 per pair) and reused.

  python3 scripts/ablation_gate.py gen      # export prompts, run on GPU box
  python3 scripts/ablation_gate.py replay   # sweep (K, vote, margin)
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from flywheel_v2 import GPU, REMOTE, RPY, BASE_MODEL, sh, ssh, knobs_of  # noqa: E402
from policy_eval import ENVS, env_configs, better_of, env_id, DEFAULT  # noqa: E402
from policy_hillclimb import neighbors, anchor_cell  # noqa: E402
from data import _run_metrics  # noqa: E402
from llm import extract_json  # noqa: E402
from prompts import counterfactual_prompt  # noqa: E402

OUT_DIR = REPO / "results" / "policy"
RUNS_DIR = REPO / "dataset" / "runs"
ADAPTER = "runs/ft-all-r0-s41/adapter"
K_GEN = 10

ORACLE_ENVS = [e for e in ENVS if e[4]]


def measured(cfg) -> dict | None:
    f = RUNS_DIR / f"{cfg.run_id}.json"
    return json.loads(f.read_text()) if f.exists() else None


def pair_id(a, b) -> str:
    return f"{a.run_id}|{b.run_id}"


def do_gen() -> None:
    prompts = OUT_DIR / "abl_prompts.jsonl"
    n = 0
    with open(prompts, "w") as f:
        for e in ORACLE_ENVS:
            metric, _ = better_of(e[0])
            for cfg in env_configs(e):
                run = measured(cfg)
                if run is None:
                    continue
                anchor = anchor_cell(cfg, run, metric)
                for knob, nb in neighbors(cfg):
                    to_cell = nb.describe()
                    to_cell["metrics"] = {}
                    task = {"type": "counterfactual", "split": "abl",
                            "knob": knob, "from_cell": anchor,
                            "to_cell": to_cell, "metric": metric}
                    f.write(json.dumps({
                        "task_id": pair_id(cfg, nb),
                        "prompt": counterfactual_prompt(task, "L0", []),
                    }) + "\n")
                    n += 1
    print(f"{n} pair prompts -> {prompts}")
    sh(f"rsync -az {prompts} {GPU}:{REMOTE}/abl_prompts.jsonl")
    ssh(f"cd {REMOTE} && {RPY} batch_generate.py --model {BASE_MODEL} "
        f"--adapter {ADAPTER} --prompts abl_prompts.jsonl "
        f"--out gen_abl.jsonl --samples {K_GEN} --temperature 0.8",
        timeout=3600)
    sh(f"rsync -az {GPU}:{REMOTE}/gen_abl.jsonl {OUT_DIR}/gen_abl.jsonl")


def load_samples() -> dict[str, list[dict]]:
    out = {}
    for line in (OUT_DIR / "gen_abl.jsonl").read_text().splitlines():
        g = json.loads(line)
        samples = []
        for t in g.get("texts") or [g["text"]]:
            ans = extract_json(t)
            if not ans or "direction" not in ans:
                continue
            try:
                samples.append({"d": str(ans["direction"]).lower(),
                                "r": float(ans.get("ratio"))})
            except (TypeError, ValueError):
                continue
        out[g["task_id"]] = samples
    return out


def replay(e, samples, k, vote_frac, margin, rng, max_steps=4):
    metric, better = better_of(e[0])
    want = "increase" if metric == "goodput_mbps" else "decrease"
    cfgs = {c.run_id: c for c in env_configs(e)}
    vals = {}
    for rid, c in cfgs.items():
        run = measured(c)
        if run:
            v = _run_metrics(run).get(metric)
            if v and v > 0:
                vals[rid] = v
    oracle = better(vals.values())

    cur = next(c for c in cfgs.values()
               if (c.cc, c.qdisc, c.buf) == DEFAULT)
    visited = {knobs_of(cur.describe())}
    import math
    n_meas = 1
    vote_min = math.ceil(vote_frac * k)
    for _ in range(max_steps):
        best_nb, best_gain = None, 1.0
        for knob, nb in neighbors(cur):
            if knobs_of(nb.describe()) in visited or nb.run_id not in vals:
                continue
            all_s = samples.get(pair_id(cur, nb), [])
            if len(all_s) < k:
                continue
            sub = rng.sample(all_s, k)
            votes = [s for s in sub if s["d"] == want]
            if len(votes) < vote_min:
                continue
            ratios = [s["r"] for s in votes if s["r"] and s["r"] > 0]
            if not ratios:
                continue
            med = statistics.median(ratios)
            gain = med if want == "increase" else 1.0 / med
            if gain >= 1.0 + margin and gain > best_gain:
                best_gain, best_nb = gain, nb
        if best_nb is None:
            break
        visited.add(knobs_of(best_nb.describe()))
        cur = best_nb
        n_meas += 1
    final = vals[cur.run_id]
    default_val = vals[next(c.run_id for c in cfgs.values()
                            if (c.cc, c.qdisc, c.buf) == DEFAULT)]
    regret = abs(final - oracle) / oracle
    regressed = (final < default_val * 0.97 if want == "increase"
                 else final > default_val * 1.03)
    return {"regret": regret, "n_meas": n_meas, "regressed": regressed}


def do_replay() -> None:
    samples = load_samples()
    grid_k = [3, 5, 10]
    grid_vote = [0.5, 0.6, 0.8, 1.0]
    grid_margin = [0.0, 0.05, 0.15, 0.30]
    rows = []
    for k, vf, mg in itertools.product(grid_k, grid_vote, grid_margin):
        regs, meas, nregr = [], [], 0
        for e in ORACLE_ENVS:
            # average subsampling noise over draws
            draws = [replay(e, samples, k, vf, mg, random.Random(s))
                     for s in range(5)] if k < K_GEN else \
                    [replay(e, samples, k, vf, mg, random.Random(0))]
            regs.append(statistics.mean(d["regret"] for d in draws))
            meas.append(statistics.mean(d["n_meas"] for d in draws))
            nregr += statistics.mean(d["regressed"] for d in draws)
        rows.append({"K": k, "vote_frac": vf, "margin": mg,
                     "mean_regret": round(statistics.mean(regs), 4),
                     "max_regret": round(max(regs), 4),
                     "mean_meas": round(statistics.mean(meas), 2),
                     "regressions": round(nregr, 2)})
    (OUT_DIR / "gate_sensitivity.json").write_text(json.dumps(rows, indent=1))
    rows.sort(key=lambda r: (r["mean_regret"], r["mean_meas"]))
    print(f"{'K':>3} {'vote':>5} {'margin':>7} {'meanReg':>8} {'maxReg':>7} "
          f"{'meas':>5} {'regress':>7}")
    for r in rows:
        print(f"{r['K']:>3} {r['vote_frac']:>5} {r['margin']:>7} "
              f"{r['mean_regret']:>8} {r['max_regret']:>7} "
              f"{r['mean_meas']:>5} {r['regressions']:>7}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "replay"])
    args = ap.parse_args()
    (do_gen if args.cmd == "gen" else do_replay)()


if __name__ == "__main__":
    main()
