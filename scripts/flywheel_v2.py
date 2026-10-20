#!/usr/bin/env python3
"""Closed-loop data flywheel with the fine-tuned world model in the loop.

Supports isolated A/B acquisition arms and multi-seed retraining:
  arm U: uncertainty acquisition (std of log-value over sampled predictions)
  arm R: random acquisition (same candidate pool, random pick)
Each arm trains only on the original grid (families ABCD) plus its own runs
(family = arm tag), so arms never contaminate each other. Every round the
LoRA is retrained from scratch once per seed and the fixed ft-clean task set
is scored per seed; the trajectory records per-seed and seed-averaged
metrics (median APE / within_2x / dir_correct — mean APE is outlier-fragile).

  sudo modprobe ... is handled by the harness; run as the user with the ssh key.

  python3 scripts/flywheel_v2.py --baseline-only --seeds 41,42,43
  python3 scripts/flywheel_v2.py --arm R --strategy random --rounds 4 \
      --per-round 24 --seeds 41,42,43
  python3 scripts/flywheel_v2.py --arm U --strategy uncertainty --rounds 4 \
      --per-round 24 --seeds 41,42,43

State/outputs: results/flywheel/{rounds,trajectory}.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import math
import pathlib
import random
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "harness"))
# both eval/ and harness/ ship a metrics.py; harness wins on sys.path (runner
# needs it), so load eval's under a private name for the aggregation step
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("eval_metrics", REPO / "eval" / "metrics.py")
eval_metrics = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(eval_metrics)

import runner  # noqa: E402
import topology as topo  # noqa: E402
from config_space import RunConfig, mutate_config  # noqa: E402
from data import PRIMARY_METRIC, _run_metrics, load_cells  # noqa: E402
from llm import extract_json  # noqa: E402
from prompts import quantitative_prompt  # noqa: E402

OUT_DIR = REPO / "results" / "flywheel"
RUNS_DIR = REPO / "dataset" / "runs"

GPU = os.environ.get("NETORACLE_GPU_HOST", "localhost")
REMOTE = "~/netmodel-ft"
RPY = "~/ft-env/bin/python"
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
GRID_FAMILIES = "ABCD"

# heldout values that must never enter training data
FORBIDDEN = {"cc": {"westwood"}, "rtt_ms": {160}, "bw_mbit": {400}}

CFG_KEYS = ["family", "workload", "cc", "qdisc", "rtt_ms",
            "loss_pct", "bw_mbit", "buf"]
KNOB_KEYS = CFG_KEYS[1:]  # identity of a config regardless of family tag


def sh(cmd: str, **kw) -> subprocess.CompletedProcess:
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, **kw)


def ssh(cmd: str, **kw) -> subprocess.CompletedProcess:
    return sh(f"ssh -o BatchMode=yes {GPU} '{cmd}'", **kw)


def allowed(cfg: RunConfig) -> bool:
    return (cfg.cc not in FORBIDDEN["cc"]
            and cfg.rtt_ms not in FORBIDDEN["rtt_ms"]
            and cfg.bw_mbit not in FORBIDDEN["bw_mbit"])


def knobs_of(d: dict) -> tuple:
    return tuple(d[k] for k in KNOB_KEYS)


def propose(cells: list[dict], n: int, arm: str,
            rng: random.Random) -> list[RunConfig]:
    """Novel configs by mutation; novelty is judged on knobs (family-blind)."""
    seen = {knobs_of(c) for c in cells}
    out: list[RunConfig] = []
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        base = rng.choice(cells)
        cand = mutate_config(
            RunConfig(**{k: base[k] for k in CFG_KEYS}),
            rng.choice([1, 2]), family=arm)
        key = knobs_of(cand.describe())
        if allowed(cand) and key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def pseudo_cell(cfg: RunConfig) -> dict:
    d = cfg.describe()
    d["metrics"] = {}
    return d


def predict_candidates(cands: list[RunConfig], adapter: str, tag: str,
                       samples: int) -> dict[str, list[float]]:
    """Returns cell_id -> list of sampled predicted values."""
    prompts_file = OUT_DIR / f"cand_prompts_{tag}.jsonl"
    with open(prompts_file, "w") as f:
        for c in cands:
            pm = PRIMARY_METRIC[c.workload][0]
            task = {"type": "quantitative", "split": "flywheel",
                    "target": pseudo_cell(c), "metric": pm}
            f.write(json.dumps({
                "task_id": f"cand:{c.cell_id}",
                "prompt": quantitative_prompt(task, "L0", []),
            }) + "\n")
    gen_file = f"gen_cand_{tag}.jsonl"
    sh(f"rsync -az {prompts_file} {GPU}:{REMOTE}/cand_prompts_{tag}.jsonl")
    ssh(f"cd {REMOTE} && rm -f {gen_file} && {RPY} batch_generate.py "
        f"--model {BASE_MODEL} --adapter {adapter} "
        f"--prompts cand_prompts_{tag}.jsonl --out {gen_file} "
        f"--samples {samples} --temperature 0.8")
    sh(f"rsync -az {GPU}:{REMOTE}/{gen_file} {OUT_DIR}/{gen_file}")

    preds: dict[str, list[float]] = {}
    for line in (OUT_DIR / gen_file).read_text().splitlines():
        g = json.loads(line)
        cid = g["task_id"].split(":", 1)[1]
        vals = []
        for t in g.get("texts") or [g["text"]]:
            ans = extract_json(t)
            try:
                v = float(ans.get("value"))
                if v > 0:
                    vals.append(v)
            except (AttributeError, TypeError, ValueError):
                pass
        preds[cid] = vals
    return preds


def uncertainty(vals: list[float], samples: int) -> float:
    if len(vals) < max(2, samples // 2):
        return float("inf")  # model can't even answer -> maximally interesting
    return statistics.stdev(math.log(v) for v in vals)


def run_testbed(selected: list[RunConfig]) -> list[dict]:
    results = []
    topo.create()
    servers = runner.Servers()
    servers.start_rr()
    try:
        for cfg in selected:
            print(f"testbed: {cfg.describe()}", flush=True)
            try:
                result = runner.execute(cfg, servers)
            except Exception as e:
                print(f"  execution failed: {e}", flush=True)
                continue
            (RUNS_DIR / f"{cfg.run_id}.json").write_text(
                json.dumps(result, indent=1))
            results.append(result)
    finally:
        servers.stop_all()
        topo.destroy()
    return results


def robust_agg(rows: list[dict]) -> dict:
    """Seed-level headline metrics; median APE instead of outlier-prone mean."""
    cf = [r for r in rows if r["type"] == "counterfactual"]
    cfo = [r for r in cf if r["split"] == "heldout_cond"]
    q = [r for r in rows if r["type"] == "quantitative"
         and r["scores"].get("ape") is not None]
    qo = [r for r in q if r["split"] == "heldout_cond"]

    def frac(rs, k):
        vals = [r["scores"][k] for r in rs if r["scores"].get(k) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def med(rs):
        vals = [r["scores"]["ape"] for r in rs]
        return round(statistics.median(vals), 4) if vals else None

    return {
        "cf_dir": frac(cf, "dir_correct"),
        "cf_dir_offgrid": frac(cfo, "dir_correct"),
        "q_med_ape": med(q),
        "q_within_2x": frac(q, "within_2x"),
        "q_med_ape_offgrid": med(qo),
        "q_within_2x_offgrid": frac(qo, "within_2x"),
    }


def train_and_eval_seed(arm: str, rnd: int, seed: int, ft_name: str,
                        ft_ids: set[str]) -> dict:
    out = f"runs/ft-{arm}-r{rnd}-s{seed}"
    ssh(f"cd {REMOTE} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"{RPY} train_lora.py --model {BASE_MODEL} --data {ft_name} "
        f"--out {out} --batch 4 --accum 4 --grad-ckpt --seed {seed}")
    gen = f"gen_eval_{arm}_r{rnd}_s{seed}.jsonl"
    ssh(f"cd {REMOTE} && rm -f {gen} && {RPY} batch_generate.py "
        f"--model {BASE_MODEL} --adapter {out}/adapter "
        f"--prompts eval_prompts.jsonl --out {gen}")
    sh(f"rsync -az {GPU}:{REMOTE}/{gen} {OUT_DIR}/{gen}")
    cond = f"ft-{arm}-r{rnd}-s{seed}|L0"
    sh(f"cd {REPO} && python3 scripts/ft_eval_io.py score "
       f"--generations {OUT_DIR}/{gen} --cond '{cond}'")
    rows = [json.loads(l) for l in
            (REPO / "results" / "eval" / f"{cond}.jsonl").read_text().splitlines()
            if json.loads(l)["task_id"] in ft_ids]
    return robust_agg(rows)


def retrain_round(arm: str, rnd: int, seeds: list[int],
                  families: str) -> tuple[dict, str]:
    """Rebuild arm dataset, train every seed, return averaged metrics and the
    adapter (first seed) to use for the next round's predictions."""
    ft_dir = REPO / "dataset" / f"ft-{arm}"
    ft_name = f"ft-{arm}"
    sh(f"cd {REPO} && python3 scripts/build_ft_dataset.py "
       f"--families {families} --out-dir {ft_dir}")
    sh(f"rsync -az {ft_dir} {GPU}:{REMOTE}/")
    ft_ids = set(json.loads((ft_dir / "manifest.json").read_text())
                 ["ft_eval_task_ids"])

    per_seed = {}
    for seed in seeds:
        per_seed[seed] = train_and_eval_seed(arm, rnd, seed, ft_name, ft_ids)

    avg = {}
    for k in next(iter(per_seed.values())):
        vals = [m[k] for m in per_seed.values() if m[k] is not None]
        avg[k] = round(sum(vals) / len(vals), 4) if vals else None
    adapter = f"runs/ft-{arm}-r{rnd}-s{seeds[0]}/adapter"
    return {"avg": avg, "per_seed": {str(s): m for s, m in per_seed.items()}}, \
        adapter


def arm_cells(families: str) -> list[dict]:
    return [c for c in load_cells() if c["family"] in set(families)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="U", help="family tag for this arm's runs")
    ap.add_argument("--strategy", choices=["uncertainty", "random"],
                    default="uncertainty")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--per-round", type=int, default=24)
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--seed", type=int, default=1, help="proposal rng seed")
    ap.add_argument("--baseline-only", action="store_true",
                    help="train+eval round-0 adapters (no testbed) and exit")
    ap.add_argument("--baseline-name", default="grid",
                    help="tag for baseline artifacts (adapter/cond names)")
    ap.add_argument("--families", default=None,
                    help="baseline training families (default ABCD)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds_log = OUT_DIR / "rounds.jsonl"
    traj_log = OUT_DIR / "trajectory.jsonl"

    if args.baseline_only:
        fams = args.families or GRID_FAMILIES
        metrics, _ = retrain_round(args.baseline_name, 0, seeds, fams)
        with open(traj_log, "a") as f:
            f.write(json.dumps({"arm": args.baseline_name, "round": 0,
                                "families": fams, **metrics}) + "\n")
        print("baseline:", json.dumps(metrics["avg"]), flush=True)
        return

    arm = args.arm
    families = GRID_FAMILIES + arm
    rng = random.Random(args.seed)
    adapter = f"runs/ft-grid-r0-s{seeds[0]}/adapter"

    for rnd in range(1, args.rounds + 1):
        print(f"\n=== ARM {arm} ROUND {rnd}/{args.rounds} "
              f"(adapter={adapter}) ===", flush=True)
        cells = arm_cells(families)
        cands = propose(cells, args.candidates, arm, rng)
        preds = predict_candidates(cands, adapter, f"{arm}_r{rnd}",
                                   args.samples)

        scored = []
        for c in cands:
            vals = preds.get(c.cell_id, [])
            u = uncertainty(vals, args.samples)
            gmean = (math.exp(statistics.mean(math.log(v) for v in vals))
                     if vals else None)
            scored.append((u, gmean, c))
        if args.strategy == "uncertainty":
            scored.sort(key=lambda t: -t[0] if t[0] != float("inf") else -1e9)
        else:
            rng.shuffle(scored)
        selected = scored[:args.per_round]

        results = run_testbed([c for _, _, c in selected])

        by_runid = {r["config"]["run_id"]: r for r in results}
        n_surprise = 0
        for u, gmean, c in selected:
            r = by_runid.get(c.run_id)
            if r is None:
                continue
            pm = PRIMARY_METRIC[c.workload][0]
            actual = _run_metrics(r).get(pm)
            rel_err = (abs(gmean - actual) / actual
                       if (gmean and actual and actual > 0) else None)
            if rel_err is not None and rel_err > 0.2:
                n_surprise += 1
            with open(rounds_log, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "arm": arm, "round": rnd,
                    "strategy": args.strategy,
                    "config": c.describe(), "metric": pm,
                    "uncertainty": None if u == float("inf") else round(u, 4),
                    "predicted_gmean": gmean, "actual": actual,
                    "rel_err": rel_err,
                }) + "\n")
        print(f"arm {arm} round {rnd}: {len(results)} runs, "
              f"{n_surprise} surprises (rel_err>20%)", flush=True)

        metrics, adapter = retrain_round(arm, rnd, seeds, families)
        with open(traj_log, "a") as f:
            f.write(json.dumps({
                "arm": arm, "round": rnd, "strategy": args.strategy,
                "n_new_runs": len(results), "n_surprise": n_surprise,
                **metrics,
            }) + "\n")
        print(f"arm {arm} round {rnd} eval(avg): "
              f"{json.dumps(metrics['avg'])}", flush=True)


if __name__ == "__main__":
    main()
