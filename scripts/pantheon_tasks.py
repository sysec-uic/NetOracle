#!/usr/bin/env python3
"""Build an external-validation task set from Pantheon real-world results.

For every downloaded experiment (dataset/pantheon/raw/*.json) with a cubic
run and at least one of {bbr, vegas}, emit measured-anchor counterfactual
tasks in the same JSON-answer format the world model was trained on:

  anchor: cubic's mean throughput (Mbit/s) and 95th-percentile one-way
          delay (ms) on a described real path
  question: switch cubic -> {bbr|vegas}; direction + ratio for
          {throughput|p95 one-way delay}

Truth is the target scheme's measured mean over runs in the same
experiment. Direction uses the same +-10% band as StackBench scoring.

Output: dataset/pantheon/tasks.jsonl and prompts.jsonl.

  python3 scripts/pantheon_tasks.py
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "dataset" / "pantheon" / "raw"
OUT = REPO / "dataset" / "pantheon"

TARGETS = ["bbr", "vegas"]
METRICS = [("tput", "average throughput", "Mbit/s"),
           ("delay", "95th-percentile one-way delay", "ms")]

PREAMBLE = """You are a model of the Linux kernel network stack. Predict execution outcomes without running anything.

Setting: a real Internet path (Pantheon measurement testbed), NOT an
emulated dumbbell. A sender transmits {flows} concurrent bulk flow(s) for
30 seconds to a receiver over the path described below. The kernel
congestion control module is the only knob that changes; qdiscs and
buffers along the path are whatever the real Internet provides.

Path: {path_desc}

A measured run on this path:
- cc=cubic -> average_throughput={tput} Mbit/s, p95_one_way_delay={delay} ms, loss_rate={loss}

Now exactly one knob changes: `cc` goes from cubic to {target}.
Everything else stays the same.

Predict the effect on {metric_desc} ({unit}):
1. direction: "increase" (>+10%), "decrease" (<-10%), or "same" (within ±10%)
2. ratio: your best numeric estimate of new_value/old_value

Answer with a single JSON object and nothing else.
JSON schema: {{"direction": "increase|decrease|same", "ratio": <float>, "reasoning": "<one short sentence>"}}"""


def scheme_mean(exp: dict, scheme: str) -> dict | None:
    runs = exp.get(scheme)
    if not isinstance(runs, dict):
        return None
    tputs, delays, losses = [], [], []
    for run in runs.values():
        agg = run.get("all") if isinstance(run, dict) else None
        if not agg:
            continue
        if agg.get("tput") and agg.get("delay"):
            tputs.append(agg["tput"])
            delays.append(agg["delay"])
            losses.append(agg.get("loss") or 0.0)
    if len(tputs) < 2:
        return None
    return {"tput": statistics.mean(tputs), "delay": statistics.mean(delays),
            "loss": statistics.mean(losses), "n_runs": len(tputs)}


def parse_name(fn: str) -> dict:
    # e.g. Brazil__reports__2018-05-24T03-27-AWS-Brazil-1-to-Brazil-10-runs-pantheon-perf.json
    node = fn.split("__")[0]
    m = re.search(r"T\d\d-\d\d-(.+?)-\d+-runs(?:-(\d+)-flows)?", fn)
    desc = m.group(1).replace("-", " ") if m else node
    flows = int(m.group(2)) if m and m.group(2) else 1
    kind = ("cellular" if "cellular" in fn or "ppp0" in fn else "wired/cloud")
    return {"node": node, "desc": desc, "flows": flows, "kind": kind}


def main() -> None:
    tasks = []
    for f in sorted(RAW.glob("*.json")):
        try:
            exp = json.loads(f.read_text())
        except Exception:
            continue
        meta = parse_name(f.name)
        cubic = scheme_mean(exp, "cubic")
        if not cubic:
            continue
        for target in TARGETS:
            tgt = scheme_mean(exp, target)
            if not tgt:
                continue
            for mkey, mdesc, unit in METRICS:
                truth_ratio = tgt[mkey] / cubic[mkey] if cubic[mkey] else None
                if not truth_ratio or truth_ratio <= 0:
                    continue
                direction = ("same" if 0.9 <= truth_ratio <= 1.1
                             else "increase" if truth_ratio > 1.1 else "decrease")
                prompt = PREAMBLE.format(
                    flows=meta["flows"],
                    path_desc=f"{meta['desc']} ({meta['kind']} path)",
                    tput=round(cubic["tput"], 2), delay=round(cubic["delay"], 1),
                    loss=round(cubic["loss"], 5),
                    target=target, metric_desc=mdesc, unit=unit)
                tasks.append({
                    "task_id": f"pantheon:{f.stem}:{target}:{mkey}",
                    "file": f.name, "node": meta["node"], "kind": meta["kind"],
                    "flows": meta["flows"], "target": target, "metric": mkey,
                    "anchor": cubic, "truth": tgt[mkey],
                    "truth_ratio": truth_ratio, "truth_direction": direction,
                    "n_runs": min(cubic["n_runs"], tgt["n_runs"]),
                    "prompt": prompt,
                })
    OUT.joinpath("tasks.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks) + "\n")
    with open(OUT / "prompts.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps({"task_id": t["task_id"], "prompt": t["prompt"]}) + "\n")
    from collections import Counter
    print(f"{len(tasks)} tasks from {len(set(t['file'] for t in tasks))} experiments")
    print("by kind:", Counter(t["kind"] for t in tasks))
    print("by target:", Counter(t["target"] for t in tasks))
    print("by metric:", Counter(t["metric"] for t in tasks))
    print("direction dist:", Counter(t["truth_direction"] for t in tasks))


if __name__ == "__main__":
    main()
