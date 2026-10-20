#!/usr/bin/env python3
"""Classical feature-based regressor baselines (GBT / RF), trained on exactly
the same cells as the released fine-tuned world model (ft-all: families
ABCDERU, leakage filters identical to build_ft_dataset.py), evaluated on the
same 218 leakage-clean StackBench tasks with the same scorers.

Answers the reviewer question: what does the LM buy over a standard
regressor on hand-built features, given identical training data?

  python3 scripts/regressor_baseline.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(HERE))

from data import RUNS_DIR, load_cells  # noqa: E402
from metrics import SCORERS  # noqa: E402
from tasks import _split_of, build_all, cell_key  # noqa: E402
from build_ft_dataset import committed_cells, METRIC_WHITELIST  # noqa: E402

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402

OUT_DIR = REPO / "results" / "eval"
MANIFEST = json.load(open(REPO / "dataset" / "ft-all" / "manifest.json"))
FAMILIES = set(MANIFEST["families"])

CC_ALGOS = ["cubic", "bbr", "reno", "vegas", "westwood", "illinois"]
QDISCS = ["pfifo_deep", "pfifo_bdp", "fq_codel", "cake", "fq"]
WORKLOADS = ["bulk", "rr", "mixed", "short"]
BUFS = ["ample", "limited"]


def featurize(c: dict) -> list[float]:
    f: list[float] = []
    f += [1.0 if c["workload"] == w else 0.0 for w in WORKLOADS]
    f += [1.0 if c["cc"] == x else 0.0 for x in CC_ALGOS]
    f += [1.0 if c["qdisc"] == q else 0.0 for q in QDISCS]
    f += [1.0 if c["buf"] == b else 0.0 for b in BUFS]
    rtt, loss, bw = c["rtt_ms"], c["loss_pct"], c["bw_mbit"]
    f += [rtt, np.log10(rtt), loss, np.log10(loss + 1e-3), bw, np.log10(bw),
          rtt * loss, bw * rtt]
    return f


def train_cells() -> list[dict]:
    """Reconstruct the exact ft-all training cells: interp-split cells of the
    allowed families minus the stratified ft_test holdout. The holdout is
    recovered from the manifest's ft_eval interp task ids."""
    cells_eval = committed_cells()
    tasks = build_all(cells_eval)
    ft_ids = set(MANIFEST["ft_eval_task_ids"])
    ft_test: set[tuple] = set()
    for t, lst in tasks.items():
        for i, task in enumerate(lst):
            if f"{t}:{i}" not in ft_ids or task["split"] != "interp":
                continue
            if t == "quantitative":
                ft_test.add(cell_key(task["target"]))
            elif t == "counterfactual":
                ft_test.add(cell_key(task["to_cell"]))
    cells_all = [c for c in load_cells(RUNS_DIR) if c["family"] in FAMILIES]
    out = [c for c in cells_all
           if _split_of(c) == "interp" and cell_key(c) not in ft_test]
    print(f"train cells: {len(out)} (manifest says {MANIFEST['n_cells_train']})")
    return out


class PerMetricModel:
    def __init__(self, make):
        self.make = make
        self.models: dict[str, object] = {}

    def fit(self, cells: list[dict]) -> None:
        for m in METRIC_WHITELIST:
            X, y = [], []
            for c in cells:
                v = c["metrics"].get(m)
                if v is None or v <= 0:
                    continue
                X.append(featurize(c))
                y.append(np.log10(v))
            if len(y) < 20:
                continue
            mod = self.make()
            mod.fit(np.array(X), np.array(y))
            self.models[m] = mod

    def predict(self, cell: dict, metric: str) -> float | None:
        mod = self.models.get(metric)
        if mod is None:
            return None
        return float(10 ** mod.predict(np.array([featurize(cell)]))[0])


def answer(task: dict, pm: PerMetricModel) -> dict | None:
    if task["type"] == "quantitative":
        v = pm.predict(task["target"], task["metric"])
        return {"value": v} if v is not None else None
    if task["type"] == "counterfactual":
        va = task["from_cell"]["primary_value"]
        vb = pm.predict(task["to_cell"], task["metric"])
        if vb is None or va <= 0:
            return None
        ratio = vb / va
        direction = ("same" if 0.9 <= ratio <= 1.1
                     else "increase" if ratio > 1.1 else "decrease")
        return {"direction": direction, "ratio": ratio}
    if task["type"] == "ranking":
        preds = {}
        for opt in task["options"]:
            proto = dict(next(iter(task["cells"])))
            proto[task["vary"]] = opt
            v = pm.predict(proto, task["metric"])
            if v is None:
                return None
            preds[opt] = v
        reverse = task["better"] == "higher"
        return {"ranking": sorted(preds, key=preds.get, reverse=reverse)}
    return None


def run(name: str, make) -> None:
    cells_train = train_cells()
    pm = PerMetricModel(make)
    pm.fit(cells_train)
    tasks = build_all(committed_cells())
    out_file = OUT_DIR / f"{name}|L0.jsonl"
    rows = []
    for ttype, lst in tasks.items():
        for i, task in enumerate(lst):
            ans = answer(task, pm)
            rows.append({
                "task_id": f"{ttype}:{i}", "type": ttype,
                "split": task["split"],
                "knob": task.get("knob") or task.get("vary"),
                "answer": ans, "scores": SCORERS[ttype](task, ans),
            })
    out_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{name}: wrote {len(rows)} rows to {out_file.name}")


if __name__ == "__main__":
    run("gbt", lambda: GradientBoostingRegressor(n_estimators=300, random_state=0))
    run("gbt600", lambda: GradientBoostingRegressor(n_estimators=600, random_state=0))
    run("rf", lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1))
