#!/usr/bin/env python3
"""Score generations on the Pantheon external task set.

Also provides two trivial reference points computed from the tasks alone:
  no-change    always predicts ratio 1.0 / "same"
  prior-dir    predicts the per-(target,metric) majority direction of the
               task set itself (an oracle-ish upper bound for any
               direction-only prior; computed via leave-one-out)

  python3 scripts/pantheon_eval.py --generations results/ft_r2/gen_pantheon_<cond>.jsonl --cond <name>
  python3 scripts/pantheon_eval.py --baselines
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from llm import extract_json  # noqa: E402

TASKS = {json.loads(l)["task_id"]: json.loads(l)
         for l in open(REPO / "dataset" / "pantheon" / "tasks.jsonl")}
OUT = REPO / "results" / "pantheon"


def score_answer(task: dict, ans: dict | None) -> dict:
    if not ans or "direction" not in ans:
        return {"valid": 0}
    truth_dir = task["truth_direction"]
    s = {"valid": 1, "dir_correct": int(ans["direction"] == truth_dir)}
    try:
        r = float(ans.get("ratio"))
        if r > 0 and task["truth_ratio"] > 0:
            s["log_ratio_err"] = abs(math.log10(r / task["truth_ratio"]))
    except (TypeError, ValueError):
        pass
    return s


def aggregate(rows: list[dict], label: str) -> None:
    by = defaultdict(list)
    for r in rows:
        by["all"].append(r)
        by[f"kind/{r['kind']}"].append(r)
        by[f"target/{r['target']}"].append(r)
        by[f"metric/{r['metric']}"].append(r)
    print(f"== {label} ==")
    out = {}
    for k, rs in sorted(by.items()):
        ok = [r for r in rs if r["scores"].get("valid")]
        if not ok:
            continue
        dir_acc = sum(r["scores"]["dir_correct"] for r in ok) / len(ok)
        lres = [r["scores"]["log_ratio_err"] for r in ok
                if "log_ratio_err" in r["scores"]]
        med_lre = statistics.median(lres) if lres else None
        out[k] = {"n": len(rs), "valid": len(ok) / len(rs),
                  "dir_acc": round(dir_acc, 3),
                  "med_log_ratio_err": round(med_lre, 3) if med_lre else None}
        print(f"  {k:22s} n={len(rs):4d} valid={out[k]['valid']:.2f} "
              f"dir={dir_acc:.3f} medLRE={out[k]['med_log_ratio_err']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label}.json").write_text(json.dumps(out, indent=1))


def score_generations(gen_file: pathlib.Path, cond: str) -> None:
    rows = []
    for line in open(gen_file):
        g = json.loads(line)
        task = TASKS.get(g["task_id"])
        if not task:
            continue
        ans = extract_json(g["text"])
        rows.append({**{k: task[k] for k in ("task_id", "kind", "target", "metric")},
                     "scores": score_answer(task, ans)})
    aggregate(rows, cond)


def baselines() -> None:
    tasks = list(TASKS.values())
    # no-change
    rows = [{**{k: t[k] for k in ("task_id", "kind", "target", "metric")},
             "scores": score_answer(t, {"direction": "same", "ratio": 1.0})}
            for t in tasks]
    aggregate(rows, "baseline-nochange")
    # per-(target,metric) majority direction, leave-one-out
    groups = defaultdict(Counter)
    for t in tasks:
        groups[(t["target"], t["metric"])][t["truth_direction"]] += 1
    rows = []
    for t in tasks:
        c = groups[(t["target"], t["metric"])].copy()
        c[t["truth_direction"]] -= 1
        pred = c.most_common(1)[0][0]
        med = statistics.median([x["truth_ratio"] for x in tasks
                                 if (x["target"], x["metric"]) == (t["target"], t["metric"])])
        rows.append({**{k: t[k] for k in ("task_id", "kind", "target", "metric")},
                     "scores": score_answer(t, {"direction": pred, "ratio": med})})
    aggregate(rows, "baseline-priordir")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=pathlib.Path)
    ap.add_argument("--cond")
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    if a.baselines:
        baselines()
    if a.generations:
        score_generations(a.generations, a.cond)
