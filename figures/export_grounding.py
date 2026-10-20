#!/usr/bin/env python3
"""Regenerate data/grounding.json from the netmodel eval results.

Usage: python3 export_grounding.py [path-to-netmodel]

Pulls only the five in-context study models (plus the kNN L2 reference)
that Figure "grounding" in the paper shows; the stale figure that mixed
in ft-* conditions came from netmodel/eval/analyze.py which plots every
condition it finds. Re-run this only when the eval reruns;
plot_figures.py reads the JSON.
"""

from __future__ import annotations

import json
import pathlib
import sys

NETMODEL = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                        pathlib.Path(__file__).resolve().parent.parent)
EVAL = NETMODEL / "results" / "eval"

MODELS = {  # eval condition stem -> display name used in the figure
    "ollama-gemma4-31b": "Gemma4 31B",
    "ollama-gemma4-e4b": "Gemma4 8B",
    "ollama-glm-4.7-flash-latest": "GLM-4.7-Flash",
    "ollama-qwen3.5-35b": "Qwen3.5 35B",
    "ollama-qwen3.5-latest": "Qwen3.5 9.7B",
}
LEVELS = ["L0", "L1", "L2", "L3"]
METRICS = {"ranking": "tau", "counterfactual": "dir_correct",
           "quantitative": "within_1_5x"}


def mean_of(rows: list[dict], ttype: str, field: str) -> float | None:
    vals = [r["scores"].get(field) for r in rows
            if r["type"] == ttype
            and isinstance(r["scores"].get(field), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def load(stem: str, level: str) -> list[dict]:
    f = EVAL / f"{stem}|{level}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines()]


out: dict = {"levels": LEVELS, "models": {}, "knn_L2": {}}
for stem, name in MODELS.items():
    out["models"][name] = {}
    for lv in LEVELS:
        rows = load(stem, lv)
        if rows:
            out["models"][name][lv] = {
                t: mean_of(rows, t, fld) for t, fld in METRICS.items()}
knn = load("knn", "L2")
out["knn_L2"] = {t: mean_of(knn, t, fld) for t, fld in METRICS.items()}

dst = pathlib.Path(__file__).resolve().parent / "data" / "grounding.json"
dst.write_text(json.dumps(out, indent=1))
print(f"wrote {dst}")
