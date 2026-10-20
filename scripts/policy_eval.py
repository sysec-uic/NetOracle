#!/usr/bin/env python3
"""End-to-end policy evaluation: can the world model pick good configs?

For each environment (workload, rtt, loss, bw) the decision space is
(cc x qdisc x buf) = 50 configs. Choosers:
  model    argbest over NetOracle's 50 predicted primary-metric values
  knn      argbest over kNN-predicted values (same visible cells)
  default  kernel-ish default: cubic + pfifo_deep + ample

Envs marked oracle=True get a full 50-config testbed sweep so we can report
regret vs the true optimum; other envs run only the choosers' picks.
Runs are tagged family "P".

  python3 scripts/policy_eval.py            # full: predictions + testbed
  python3 scripts/policy_eval.py --report   # summarize from existing runs
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from flywheel_v2 import (  # noqa: E402
    GPU, REMOTE, RPY, BASE_MODEL, OUT_DIR as FLY_DIR, run_testbed, sh, ssh,
    knobs_of, pseudo_cell,
)
from baselines import _knn_value  # noqa: E402
from config_space import RunConfig  # noqa: E402
from data import PRIMARY_METRIC, _run_metrics, load_cells  # noqa: E402
from llm import extract_json  # noqa: E402
from prompts import quantitative_prompt  # noqa: E402

OUT_DIR = REPO / "results" / "policy"
RUNS_DIR = REPO / "dataset" / "runs"
ADAPTER = "runs/ft-all-r0-s41/adapter"

CCS = ["cubic", "bbr", "reno", "vegas", "illinois"]
QDISCS = ["pfifo_deep", "pfifo_bdp", "fq_codel", "cake", "fq"]
BUFS = ["ample", "limited"]

# (workload, rtt_ms, loss_pct, bw_mbit, oracle_sweep)
ENVS = [
    ("bulk", 40, 0.5, 50, True),
    ("mixed", 40, 2.0, 100, True),
    ("bulk", 120, 0.1, 200, False),
    ("rr", 80, 0.5, 25, False),
    ("mixed", 120, 0.5, 50, False),
    ("short", 20, 0.5, 200, False),
    ("short", 80, 1.0, 50, False),
    ("bulk", 10, 2.0, 100, False),
    # off-grid: rtt/bw values the model never trained on
    ("bulk", 160, 0.5, 100, True),
    ("mixed", 160, 0.1, 100, True),
    ("bulk", 20, 0.0, 400, False),
    ("mixed", 40, 0.5, 400, False),
]

DEFAULT = ("cubic", "pfifo_deep", "ample")


def env_id(e) -> str:
    return f"{e[0]}_r{e[1]}_l{e[2]}_b{e[3]}"


def env_configs(e) -> list[RunConfig]:
    wl, rtt, loss, bw, _ = e
    return [RunConfig("P", wl, cc, qd, rtt, loss, bw, buf)
            for cc, qd, buf in itertools.product(CCS, QDISCS, BUFS)]


def better_of(wl: str):
    metric, direction = PRIMARY_METRIC[wl]
    return metric, (max if direction == "higher" else min)


def model_predictions() -> dict[str, float]:
    prompts = OUT_DIR / "policy_prompts.jsonl"
    with open(prompts, "w") as f:
        for e in ENVS:
            metric, _ = PRIMARY_METRIC[e[0]]
            for cfg in env_configs(e):
                task = {"type": "quantitative", "split": "policy",
                        "target": pseudo_cell(cfg), "metric": metric}
                f.write(json.dumps({
                    "task_id": f"{env_id(e)}|{cfg.cc}|{cfg.qdisc}|{cfg.buf}",
                    "prompt": quantitative_prompt(task, "L0", []),
                }) + "\n")
    sh(f"rsync -az {prompts} {GPU}:{REMOTE}/policy_prompts.jsonl")
    ssh(f"cd {REMOTE} && rm -f gen_policy.jsonl && {RPY} batch_generate.py "
        f"--model {BASE_MODEL} --adapter {ADAPTER} "
        f"--prompts policy_prompts.jsonl --out gen_policy.jsonl")
    sh(f"rsync -az {GPU}:{REMOTE}/gen_policy.jsonl {OUT_DIR}/gen_policy.jsonl")
    preds = {}
    for line in (OUT_DIR / "gen_policy.jsonl").read_text().splitlines():
        g = json.loads(line)
        ans = extract_json(g["text"])
        try:
            preds[g["task_id"]] = float(ans.get("value"))
        except (AttributeError, TypeError, ValueError):
            preds[g["task_id"]] = None
    return preds


def choose(preds: dict) -> list[dict]:
    cells = load_cells()
    choices = []
    for e in ENVS:
        wl = e[0]
        metric, best = better_of(wl)
        cfgs = env_configs(e)

        mv = {c: preds.get(f"{env_id(e)}|{c.cc}|{c.qdisc}|{c.buf}")
              for c in cfgs}
        mv = {c: v for c, v in mv.items() if v is not None and v > 0}
        model_pick = best(mv, key=mv.get) if mv else None

        kv = {}
        for c in cfgs:
            v = _knn_value(cells, {**pseudo_cell(c)}, metric)
            if v is not None and v > 0:
                kv[c] = v
        knn_pick = best(kv, key=kv.get) if kv else None

        default_pick = next(c for c in cfgs if
                            (c.cc, c.qdisc, c.buf) == DEFAULT)
        choices.append({
            "env": env_id(e), "workload": wl, "metric": metric,
            "oracle": e[4],
            "model": model_pick.describe() if model_pick else None,
            "model_predicted": mv.get(model_pick),
            "knn": knn_pick.describe() if knn_pick else None,
            "knn_predicted": kv.get(knn_pick),
            "default": default_pick.describe(),
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "choices.jsonl").write_text(
        "\n".join(json.dumps(c) for c in choices) + "\n")
    return choices


def execute(choices: list[dict]) -> None:
    todo: dict[str, RunConfig] = {}
    for e, ch in zip(ENVS, choices):
        cfgs = env_configs(e)
        if e[4]:  # oracle sweep: everything
            for c in cfgs:
                todo[c.run_id] = c
        for k in ("model", "knn", "default"):
            if ch[k]:
                c = RunConfig(**{kk: ch[k][kk] for kk in
                               ["family", "workload", "cc", "qdisc", "rtt_ms",
                                "loss_pct", "bw_mbit", "buf", "rep"]})
                todo[c.run_id] = c
    done = {p.stem for p in RUNS_DIR.glob("*.json")}
    pending = [c for rid, c in sorted(todo.items()) if rid not in done]
    print(f"policy eval: {len(todo)} configs, {len(pending)} to run",
          flush=True)
    run_testbed(pending)


def report() -> None:
    choices = [json.loads(l) for l in
               (OUT_DIR / "choices.jsonl").read_text().splitlines()]
    by_id = {}
    for p in RUNS_DIR.glob("*.json"):
        by_id[p.stem] = None  # lazy
    def actual(cfg_desc, metric):
        rid = cfg_desc["run_id"]
        p = RUNS_DIR / f"{rid}.json"
        if not p.exists():
            return None
        return _run_metrics(json.loads(p.read_text())).get(metric)

    print(f"{'env':22s} {'metric':12s} {'default':>9s} {'knn':>9s} "
          f"{'model':>9s} {'oracle':>9s}  regret(model)")
    summary = []
    for e, ch in zip(ENVS, choices):
        metric, best = better_of(ch["workload"])
        vals = {k: actual(ch[k], metric) if ch[k] else None
                for k in ("model", "knn", "default")}
        oracle_val = None
        if ch["oracle"]:
            all_vals = [actual(c.describe(), metric) for c in env_configs(e)]
            all_vals = [v for v in all_vals if v is not None and v > 0]
            if all_vals:
                oracle_val = best(all_vals)
        regret = None
        if oracle_val and vals["model"]:
            regret = (abs(vals["model"] - oracle_val) / oracle_val)
        row = {"env": ch["env"], "metric": metric, **vals,
               "oracle": oracle_val, "model_regret": regret}
        summary.append(row)
        fmt = lambda v: f"{v:9.2f}" if v is not None else "        -"
        print(f"{ch['env']:22s} {metric:12s} {fmt(vals['default'])} "
              f"{fmt(vals['knn'])} {fmt(vals['model'])} {fmt(oracle_val)}  "
              f"{regret if regret is None else round(regret, 3)}")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.report:
        preds = model_predictions()
        choices = choose(preds)
        execute(choices)
    report()


if __name__ == "__main__":
    main()
