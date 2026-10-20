#!/usr/bin/env python3
"""E4: zero-shot direction accuracy on a knob value never in training.

The discriminating experiment for the world-model claim: introduce
qdisc values that appear in no training cell and in no context
(default: sfq and pie), measure real counterfactual pairs on the
testbed, then ask the fine-tuned model zero-shot. A tabular regressor
cannot even accept the input (the category does not exist in its
feature space); the model can only answer from pretrained mechanism
knowledge of what sfq/pie do. Direction accuracy above chance on these
pairs is evidence for internalized mechanisms rather than dataset
interpolation.

Stages:
  measure  run (base config vs base+new-qdisc) pairs on the testbed
           across a small environment set                      [root]
  query    ask the model zero-shot for each measured pair       [GPU]
  (both stages: --stage all)

Outputs land in results/newknob/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "harness"))

from config_space import RunConfig  # noqa: E402
from data import PRIMARY_METRIC, _run_metrics  # noqa: E402

OUT = HERE.parent / "results" / "newknob"
NEW_QDISCS = ["sfq", "pie"]
BASES = [("cubic", "pfifo_deep", "ample"), ("bbr", "pfifo_deep", "ample")]
# (workload, rtt_ms, loss_pct, bw_mbit) — mix of grid and off-grid
ENVS = [("bulk", 40, 1.0, 50), ("bulk", 20, 0.0, 100),
        ("mixed", 20, 0.0, 25), ("mixed", 60, 0.3, 75)]
DEAD_BAND = 0.10


def cfg(wl, rtt, loss, bw, triple) -> RunConfig:
    return RunConfig("N", wl, triple[0], triple[1], rtt, loss, bw,
                     triple[2])


def stage_measure(reps: int) -> None:
    import runner  # noqa: E402
    import topology as topo  # noqa: E402
    pairs = []
    topo.create()
    servers = runner.Servers()
    servers.start_rr()

    def run_med(c: RunConfig, metric: str):
        vals, anchor = [], None
        for _ in range(reps):
            try:
                r = runner.execute(c, servers)
                m = _run_metrics(r)
                if m.get(metric):
                    vals.append(m[metric])
                    anchor = m
            except Exception as e:  # noqa: BLE001
                print(f"  fail {c.qdisc}: {e}", flush=True)
        return (statistics.median(vals) if vals else None), anchor

    try:
        for wl, rtt, loss, bw in ENVS:
            metric, _ = PRIMARY_METRIC[wl]
            for base in BASES:
                base_cfg = cfg(wl, rtt, loss, bw, base)
                v_base, anchor = run_med(base_cfg, metric)
                if not v_base:
                    continue
                for nq in NEW_QDISCS:
                    v_new, _ = run_med(
                        cfg(wl, rtt, loss, bw, (base[0], nq, base[2])),
                        metric)
                    if not v_new:
                        continue
                    ratio = v_new / v_base
                    if ratio > 1 + DEAD_BAND:
                        truth = "increase"
                    elif ratio < 1 - DEAD_BAND:
                        truth = "decrease"
                    else:
                        truth = "same"
                    rec = {"workload": wl, "rtt_ms": rtt,
                           "loss_pct": loss, "bw_mbit": bw,
                           "base": "/".join(base), "new_qdisc": nq,
                           "metric": metric, "v_base": v_base,
                           "v_new": v_new, "truth_ratio": round(ratio, 3),
                           "truth_direction": truth,
                           "anchor_metrics": anchor}
                    pairs.append(rec)
                    print(f"measured {wl} {rtt}ms/{loss}%/{bw}M "
                          f"{'/'.join(base)} -> {nq}: {ratio:.2f}x "
                          f"({truth})", flush=True)
    finally:
        servers.stop_all()
        topo.destroy()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pairs))
    print(f"measure: {len(pairs)} pairs", flush=True)


def stage_query() -> None:
    from closed_loop_demo import Model, ADAPTER, query_ratio  # noqa: E402
    model = Model(ADAPTER)
    pairs = [json.loads(l)
             for l in (OUT / "pairs.jsonl").read_text().splitlines()]
    results, correct, decided = [], 0, 0
    for p in pairs:
        anchor = {"workload": p["workload"], "cc": p["base"].split("/")[0],
                  "qdisc": p["base"].split("/")[1],
                  "buf": p["base"].split("/")[2],
                  "rtt_ms": p["rtt_ms"], "loss_pct": p["loss_pct"],
                  "bw_mbit": p["bw_mbit"],
                  "metrics": p["anchor_metrics"],
                  "primary_metric": p["metric"],
                  "primary_value": p["v_base"]}
        nb = RunConfig("N", p["workload"], anchor["cc"], p["new_qdisc"],
                       p["rtt_ms"], p["loss_pct"], p["bw_mbit"],
                       anchor["buf"])
        dirs, ratios = query_ratio(model, anchor, nb, p["metric"],
                                   "qdisc")
        if not dirs:
            results.append({**p, "pred": None})
            continue
        pred = statistics.mode(dirs)
        med = statistics.median(ratios) if ratios else None
        ok = pred == p["truth_direction"]
        decided += 1
        correct += int(ok)
        results.append({**p, "pred": pred, "pred_ratio": med,
                        "correct": ok})
        print(f"query {p['workload']} {p['base']} -> {p['new_qdisc']}: "
              f"pred={pred}({med}) truth={p['truth_direction']}"
              f"({p['truth_ratio']}) {'OK' if ok else 'X'}", flush=True)
    acc = correct / decided if decided else None
    (OUT / "zeroshot_results.json").write_text(json.dumps(
        {"n_pairs": len(pairs), "decided": decided,
         "direction_accuracy": acc, "chance": 0.5,
         "results": results}, indent=1))
    print(f"E4: direction accuracy {acc} over {decided} decided pairs "
          f"(chance 0.5); tabular baselines cannot accept the input",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["measure", "query", "all"])
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("measure", "all"):
        stage_measure(args.reps)
    if args.stage in ("query", "all"):
        stage_query()


if __name__ == "__main__":
    main()
