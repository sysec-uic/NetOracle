#!/usr/bin/env python3
"""Confidence-gated hill-climbing policy chooser.

Instead of argbest over 50 absolute-value predictions (weak for p99-under-
load), climb from the kernel default using the model's strongest skill —
counterfactual direction prediction anchored on a *measured* run:

  1. run the current config on the testbed (a runtime daemon gets this
     measurement for free from live traffic)
  2. for every one-knob neighbor (cc/qdisc/buf), sample K counterfactual
     predictions; a move is allowed only if >= vote_min of K agree the
     metric improves AND the median predicted ratio clears a margin
  3. move to the best gated neighbor, re-measure, repeat (max steps)

Worst case: no confident improvement exists and we stay at the default —
the chooser can never pick a catastrophically bad config it was never
forced to visit.

  python3 scripts/policy_hillclimb.py
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flywheel_v2 import (  # noqa: E402
    GPU, REMOTE, RPY, BASE_MODEL, sh, ssh, knobs_of,
)
import runner  # noqa: E402
import topology as topo  # noqa: E402
from config_space import RunConfig  # noqa: E402
from data import PRIMARY_METRIC, _run_metrics  # noqa: E402
from llm import extract_json  # noqa: E402
from prompts import counterfactual_prompt  # noqa: E402
from policy_eval import ENVS, CCS, QDISCS, BUFS, DEFAULT, env_id  # noqa: E402

OUT_DIR = HERE.parent / "results" / "policy"
RUNS_DIR = HERE.parent / "dataset" / "runs"
ADAPTER = "runs/ft-all-r0-s41/adapter"

MAX_STEPS = 4
K_SAMPLES = 5
VOTE_MIN = 4
MARGIN = 0.15  # median predicted improvement must exceed 15%


def neighbors(cfg: RunConfig) -> list[tuple[str, RunConfig]]:
    out = []
    for cc in CCS:
        if cc != cfg.cc:
            out.append(("cc", RunConfig(**{**cfg.__dict__, "cc": cc})))
    for qd in QDISCS:
        if qd != cfg.qdisc:
            out.append(("qdisc", RunConfig(**{**cfg.__dict__, "qdisc": qd})))
    out.append(("buf", RunConfig(**{**cfg.__dict__,
                "buf": "limited" if cfg.buf == "ample" else "ample"})))
    return out


def measure(cfg: RunConfig, servers) -> dict | None:
    """Run on testbed (reusing an existing run file if present)."""
    f = RUNS_DIR / f"{cfg.run_id}.json"
    if f.exists():
        return json.loads(f.read_text())
    try:
        result = runner.execute(cfg, servers)
    except Exception as e:
        print(f"  execution failed: {e}", flush=True)
        return None
    f.write_text(json.dumps(result, indent=1))
    return result


def anchor_cell(cfg: RunConfig, result: dict, metric: str) -> dict:
    d = cfg.describe()
    d["metrics"] = _run_metrics(result)
    d["primary_metric"] = metric
    d["primary_value"] = d["metrics"].get(metric)
    return d


def predict_neighbors(anchor: dict, nbrs: list[tuple[str, RunConfig]],
                      metric: str, tag: str) -> dict[str, list[dict]]:
    """Returns run_id -> list of {direction, ratio} samples."""
    prompts = OUT_DIR / f"hc_prompts_{tag}.jsonl"
    with open(prompts, "w") as f:
        for knob, nb in nbrs:
            to_cell = nb.describe()
            to_cell["metrics"] = {}
            task = {"type": "counterfactual", "split": "policy", "knob": knob,
                    "from_cell": anchor, "to_cell": to_cell, "metric": metric}
            f.write(json.dumps({
                "task_id": nb.run_id,
                "prompt": counterfactual_prompt(task, "L0", []),
            }) + "\n")
    gen = f"gen_hc_{tag}.jsonl"
    sh(f"rsync -az {prompts} {GPU}:{REMOTE}/hc_prompts.jsonl")
    ssh(f"cd {REMOTE} && rm -f {gen} && {RPY} batch_generate.py "
        f"--model {BASE_MODEL} --adapter {ADAPTER} "
        f"--prompts hc_prompts.jsonl --out {gen} "
        f"--samples {K_SAMPLES} --temperature 0.8")
    sh(f"rsync -az {GPU}:{REMOTE}/{gen} {OUT_DIR}/{gen}")
    preds: dict[str, list[dict]] = {}
    for line in (OUT_DIR / gen).read_text().splitlines():
        g = json.loads(line)
        samples = []
        for t in g.get("texts") or [g["text"]]:
            ans = extract_json(t)
            if not ans or "direction" not in ans:
                continue
            try:
                samples.append({"direction": str(ans["direction"]).lower(),
                                "ratio": float(ans.get("ratio"))})
            except (TypeError, ValueError):
                continue
        preds[g["task_id"]] = samples
    return preds


def gated_best(nbrs, preds, better: str):
    """Best neighbor passing the confidence gate, or None."""
    want = "increase" if better == "higher" else "decrease"
    best_nb, best_gain = None, 1.0
    for knob, nb in nbrs:
        samples = preds.get(nb.run_id, [])
        votes = [s for s in samples if s["direction"] == want]
        if len(samples) < K_SAMPLES - 1 or len(votes) < VOTE_MIN:
            continue
        ratios = [s["ratio"] for s in votes if s["ratio"] and s["ratio"] > 0]
        if not ratios:
            continue
        med = statistics.median(ratios)
        gain = med if better == "higher" else 1.0 / med
        if gain >= 1.0 + MARGIN and gain > best_gain:
            best_gain, best_nb = gain, nb
    return best_nb, best_gain


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traces = []
    topo.create()
    servers = runner.Servers()
    servers.start_rr()
    try:
        for e in ENVS:
            wl, rtt, loss, bw, _ = e
            metric, direction = PRIMARY_METRIC[wl]
            cur = RunConfig("P", wl, DEFAULT[0], DEFAULT[1], rtt, loss, bw,
                            DEFAULT[2])
            trace = {"env": env_id(e), "metric": metric, "better": direction,
                     "steps": []}
            print(f"\n== {env_id(e)} ({metric}, {direction} better) ==",
                  flush=True)
            visited = {knobs_of(cur.describe())}
            for step in range(MAX_STEPS):
                result = measure(cur, servers)
                if result is None:
                    break
                anchor = anchor_cell(cur, result, metric)
                cur_val = anchor["primary_value"]
                print(f"step {step}: {cur.cc}/{cur.qdisc}/{cur.buf} "
                      f"{metric}={cur_val}", flush=True)
                nbrs = [(k, nb) for k, nb in neighbors(cur)
                        if knobs_of(nb.describe()) not in visited]
                if not nbrs:
                    break
                preds = predict_neighbors(anchor, nbrs, metric,
                                          f"{env_id(e)}_s{step}")
                nxt, gain = gated_best(nbrs, preds, direction)
                trace["steps"].append({
                    "config": f"{cur.cc}/{cur.qdisc}/{cur.buf}",
                    "value": cur_val,
                    "move": f"{nxt.cc}/{nxt.qdisc}/{nxt.buf}" if nxt else None,
                    "predicted_gain": round(gain, 3) if nxt else None,
                })
                if nxt is None:
                    print("  no confident improvement -> stop", flush=True)
                    break
                visited.add(knobs_of(nxt.describe()))
                cur = nxt
            # measure final config if the loop ended on a move
            result = measure(cur, servers)
            final_val = (_run_metrics(result).get(metric)
                         if result else None)
            trace["final_config"] = f"{cur.cc}/{cur.qdisc}/{cur.buf}"
            trace["final_value"] = final_val
            traces.append(trace)
            print(f"final: {trace['final_config']} {metric}={final_val}",
                  flush=True)
    finally:
        servers.stop_all()
        topo.destroy()

    (OUT_DIR / "hillclimb.json").write_text(json.dumps(traces, indent=1))
    print("\nHILLCLIMB DONE", flush=True)


if __name__ == "__main__":
    main()
