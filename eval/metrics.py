"""Scoring: Kendall tau, top-1, direction accuracy, log-ratio error, MAPE."""

from __future__ import annotations

import math


def kendall_tau(pred: list, truth: list) -> float | None:
    """Tau over items present in both rankings."""
    common = [x for x in truth if x in pred]
    if len(common) < 2:
        return None
    pos_p = {x: i for i, x in enumerate(pred)}
    pos_t = {x: i for i, x in enumerate(common)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            s = (pos_p[a] - pos_p[b]) * (pos_t[a] - pos_t[b])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    n = conc + disc
    return (conc - disc) / n if n else None


def score_ranking(task: dict, answer: dict | None) -> dict:
    reverse = task["better"] == "higher"
    truth = [k for k, _ in sorted(task["options"].items(), key=lambda kv: kv[1],
                                  reverse=reverse)]
    out = {"n_options": len(truth)}
    pred = (answer or {}).get("ranking")
    if not isinstance(pred, list) or not pred:
        return {**out, "valid": 0, "tau": None, "top1": None}
    pred = [str(x).strip().lower() for x in pred]
    out["valid"] = 1
    out["tau"] = kendall_tau(pred, truth)
    out["top1"] = int(pred[0] == truth[0])
    return out


def score_counterfactual(task: dict, answer: dict | None) -> dict:
    out: dict = {}
    if not answer or "direction" not in answer:
        return {"valid": 0, "dir_correct": None, "log_ratio_err": None}
    out["valid"] = 1
    pred_dir = str(answer["direction"]).strip().lower()
    out["dir_correct"] = int(pred_dir == task["truth_direction"])
    try:
        r = float(answer.get("ratio"))
        if r > 0 and task["truth_ratio"] > 0:
            out["log_ratio_err"] = abs(math.log(r / task["truth_ratio"]))
        else:
            out["log_ratio_err"] = None
    except (TypeError, ValueError):
        out["log_ratio_err"] = None
    return out


def score_quantitative(task: dict, answer: dict | None) -> dict:
    if not answer:
        return {"valid": 0, "ape": None, "within_1_5x": None, "within_2x": None}
    try:
        v = float(answer.get("value"))
    except (TypeError, ValueError):
        return {"valid": 0, "ape": None, "within_1_5x": None, "within_2x": None}
    truth = task["truth"]
    if truth <= 0 or v <= 0:
        return {"valid": 1, "ape": None, "within_1_5x": None, "within_2x": None}
    ratio = max(v / truth, truth / v)
    return {
        "valid": 1,
        "ape": abs(v - truth) / truth,
        "within_1_5x": int(ratio <= 1.5),
        "within_2x": int(ratio <= 2.0),
    }


SCORERS = {
    "ranking": score_ranking,
    "counterfactual": score_counterfactual,
    "quantitative": score_quantitative,
}


def aggregate(rows: list[dict]) -> dict:
    """Mean of every numeric score field, plus counts."""
    out: dict = {"n": len(rows)}
    keys = {k for r in rows for k, v in r.get("scores", {}).items()
            if isinstance(v, (int, float))}
    for k in sorted(keys):
        vals = [r["scores"][k] for r in rows
                if isinstance(r.get("scores", {}).get(k), (int, float))]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
            out[f"{k}_n"] = len(vals)
    return out
