#!/usr/bin/env python3
"""kNN hyperparameter sensitivity: k in {1,3,5,7,10} x {distance,uniform}
weighting, scored on the 218 leakage-clean tasks. Defends the kNN baseline
against a cherry-picking objection.

  python3 scripts/knn_sensitivity.py
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "scripts"))

import baselines  # noqa: E402
from data import config_distance, config_of  # noqa: E402
from metrics import SCORERS  # noqa: E402
from tasks import build_all  # noqa: E402
from build_ft_dataset import committed_cells  # noqa: E402

MANIFEST = json.load(open(REPO / "dataset" / "ft" / "manifest.json"))
IDS = set(MANIFEST["ft_eval_task_ids"])


def knn_value_factory(k: int, weighted: bool):
    def _knn_value(pool, target, metric):
        cands = [c for c in pool if c["workload"] == target["workload"]
                 and metric in c["metrics"]]
        if not cands:
            return None
        ranked = sorted(cands, key=lambda c: config_distance(config_of(c), config_of(target)))
        top = ranked[:k]
        if weighted:
            ws = [1.0 / (0.1 + config_distance(config_of(c), config_of(target)))
                  for c in top]
        else:
            ws = [1.0] * len(top)
        vals = [c["metrics"][metric] for c in top]
        return sum(w * v for w, v in zip(ws, vals)) / sum(ws)
    return _knn_value


def main() -> None:
    cells = committed_cells()
    tasks = build_all(cells)
    print(f"{'variant':16s} dir_all dir_off medAPE w2x_off")
    results = {}
    for k in (1, 3, 5, 7, 10):
        for weighted in (True, False):
            baselines._knn_value = knn_value_factory(k, weighted)
            cf, cfo, apes, qo = [], [], [], []
            for ttype, lst in tasks.items():
                for i, task in enumerate(lst):
                    if f"{ttype}:{i}" not in IDS:
                        continue
                    ans = baselines.knn_answer(task, cells)
                    s = SCORERS[ttype](task, ans)
                    if ttype == "counterfactual":
                        if isinstance(s.get("dir_correct"), (int, float)):
                            cf.append(s["dir_correct"])
                            if task["split"] == "heldout_cond":
                                cfo.append(s["dir_correct"])
                    elif ttype == "quantitative":
                        if isinstance(s.get("ape"), (int, float)):
                            apes.append(s["ape"])
                            if task["split"] == "heldout_cond":
                                qo.append(s["within_2x"])
            name = f"k={k} {'dist' if weighted else 'unif'}"
            row = (sum(cf) / len(cf), sum(cfo) / len(cfo),
                   statistics.median(apes), sum(qo) / len(qo))
            results[name] = row
            print(f"{name:16s} {row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}  {row[3]:.3f}")
    out = REPO / "results" / "eval" / "knn_sensitivity.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
