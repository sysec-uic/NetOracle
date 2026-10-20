#!/usr/bin/env python3
"""Materialize StackBench: the 278 prediction tasks derived from the frozen
evaluation matrix (run families A to D, 456 runs, 228 cells).

  python3 scripts/export_stackbench.py   -> dataset/stackbench/tasks.jsonl

Task ids are "<type>:<index>" and match the task_id field of every file in
results/eval/. dataset/stackbench/clean_task_ids.json lists the 218 tasks whose
target cells are absent from the fine-tuning set (dataset/ft-all).
"""
from __future__ import annotations
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "eval"))
from data import load_cells  # noqa: E402
from tasks import build_all  # noqa: E402

FROZEN_FAMILIES = {"A", "B", "C", "D"}

def main() -> None:
    cells = [c for c in load_cells() if c["family"] in FROZEN_FAMILIES]
    tasks = build_all(cells)
    out = REPO / "dataset" / "stackbench"; out.mkdir(exist_ok=True)
    n = 0
    with open(out / "tasks.jsonl", "w") as f:
        for ttype, lst in tasks.items():
            for i, t in enumerate(lst):
                f.write(json.dumps({"task_id": f"{ttype}:{i}", **t}, sort_keys=True) + "\n"); n += 1
    manifest = json.load(open(REPO / "dataset" / "ft-all" / "manifest.json"))
    ids = manifest["ft_eval_task_ids"]
    json.dump({"n": len(ids), "task_ids": ids,
               "note": "Tasks whose target cells are not in dataset/ft-all; the paper's Table 1 scores every model on exactly these."},
              open(out / "clean_task_ids.json", "w"), indent=1)
    print(f"{len(cells)} frozen cells -> {n} tasks; {len(ids)} leakage-clean ids")

if __name__ == "__main__":
    main()
