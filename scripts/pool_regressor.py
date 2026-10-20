#!/usr/bin/env python3
"""In-context-frame regressor baseline for the HotNets (278-task) table.

For every StackBench task, fit a gradient-boosted-tree regressor on
exactly the task's legal context pool (the same cells the kNN baseline and
the LLM L2 condition see), then answer the task from its predictions.
This answers "would a classical regressor beat kNN given identical
evidence?" without introducing any training corpus beyond the task's own
visible measurements.

  python3 scripts/pool_regressor.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "scripts"))

from metrics import SCORERS  # noqa: E402
from tasks import build_all, context_pool  # noqa: E402
from build_ft_dataset import committed_cells  # noqa: E402
from regressor_baseline import featurize  # noqa: E402

OUT = REPO / "results" / "eval" / "gbt-pool|L2.jsonl"


def fit_pool(pool: list[dict], workload: str, metric: str):
    X, y = [], []
    for c in pool:
        if c["workload"] != workload:
            continue
        v = c["metrics"].get(metric)
        if v is None or v <= 0:
            continue
        X.append(featurize(c))
        y.append(np.log10(v))
    if len(y) < 8:
        return None
    m = GradientBoostingRegressor(n_estimators=200, random_state=0)
    m.fit(np.array(X), np.array(y))
    return m


def predict(model, cell: dict) -> float:
    return float(10 ** model.predict(np.array([featurize(cell)]))[0])


def answer(task: dict, cells: list[dict]) -> dict | None:
    pool = context_pool(cells, task)
    if task["type"] == "quantitative":
        m = fit_pool(pool, task["target"]["workload"], task["metric"])
        return {"value": predict(m, task["target"])} if m else None
    if task["type"] == "counterfactual":
        m = fit_pool(pool, task["to_cell"]["workload"], task["metric"])
        if not m:
            return None
        va = task["from_cell"]["primary_value"]
        vb = predict(m, task["to_cell"])
        if va <= 0:
            return None
        ratio = vb / va
        direction = ("same" if 0.9 <= ratio <= 1.1
                     else "increase" if ratio > 1.1 else "decrease")
        return {"direction": direction, "ratio": ratio}
    if task["type"] == "ranking":
        proto0 = next(iter(task["cells"]))
        m = fit_pool(pool, proto0["workload"], task["metric"])
        if not m:
            return None
        preds = {}
        for opt in task["options"]:
            proto = dict(proto0)
            proto[task["vary"]] = opt
            preds[opt] = predict(m, proto)
        reverse = task["better"] == "higher"
        return {"ranking": sorted(preds, key=preds.get, reverse=reverse)}
    return None


def main() -> None:
    cells = committed_cells()
    tasks = build_all(cells)
    rows = []
    for ttype, lst in tasks.items():
        for i, task in enumerate(lst):
            ans = answer(task, cells)
            rows.append({
                "task_id": f"{ttype}:{i}", "type": ttype,
                "split": task["split"],
                "knob": task.get("knob") or task.get("vary"),
                "answer": ans, "scores": SCORERS[ttype](task, ans),
            })
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT.name}")


if __name__ == "__main__":
    main()
