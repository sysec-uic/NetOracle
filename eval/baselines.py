"""Non-LLM baselines: k-nearest-neighbor tabular predictor.

Gets exactly the same legal context pool as the LLM's L2 condition, so the
comparison isolates what text knowledge adds on top of execution data — in
particular on held-out-CC / held-out-condition splits where pure tabular
matching has nothing to interpolate from.
"""

from __future__ import annotations

import statistics

from data import PRIMARY_METRIC, config_distance, config_of
from tasks import context_pool

K = 5


def _knn_value(pool: list[dict], target: dict, metric: str) -> float | None:
    cands = [c for c in pool if c["workload"] == target["workload"]
             and metric in c["metrics"]]
    if not cands:
        return None
    ranked = sorted(cands, key=lambda c: config_distance(config_of(c), config_of(target)))
    top = ranked[:K]
    weights = [1.0 / (0.1 + config_distance(config_of(c), config_of(target))) for c in top]
    vals = [c["metrics"][metric] for c in top]
    return sum(w * v for w, v in zip(weights, vals)) / sum(weights)


def knn_answer(task: dict, cells: list[dict]) -> dict | None:
    pool = context_pool(cells, task)
    if task["type"] == "quantitative":
        v = _knn_value(pool, task["target"], task["metric"])
        return {"value": v} if v is not None else None
    if task["type"] == "counterfactual":
        va = task["from_cell"]["primary_value"]
        vb = _knn_value(pool, task["to_cell"], task["metric"])
        if vb is None or va <= 0:
            return None
        ratio = vb / va
        direction = "same" if 0.9 <= ratio <= 1.1 else ("increase" if ratio > 1.1 else "decrease")
        return {"direction": direction, "ratio": ratio}
    if task["type"] == "ranking":
        preds = {}
        for opt, _ in task["options"].items():
            # synthesize the option's config from any group cell
            proto = dict(next(iter(task["cells"])))
            proto[task["vary"]] = opt
            v = _knn_value(pool, proto, task["metric"])
            if v is None:
                return None
            preds[opt] = v
        reverse = task["better"] == "higher"
        return {"ranking": sorted(preds, key=preds.get, reverse=reverse)}
    return None
