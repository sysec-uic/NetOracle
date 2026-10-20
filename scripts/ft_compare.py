#!/usr/bin/env python3
"""Compare conditions on the FT-clean eval subset.

Restricts every results/eval/<condition>.jsonl to the task_ids recorded in
dataset/ft/manifest.json (tasks whose target the fine-tuned model never saw)
and aggregates per type/split, so fine-tuned and in-context baselines are
scored on identical tasks.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "eval"))

from metrics import aggregate  # noqa: E402

EVAL_DIR = REPO / "results" / "eval"
MANIFEST = REPO / "dataset" / "ft" / "manifest.json"

KEY_FIELDS = {
    "counterfactual": ["dir_correct", "log_ratio_err", "valid"],
    "quantitative": ["ape", "within_1_5x", "within_2x", "valid"],
    "ranking": ["tau", "top1", "valid"],
}


def main() -> None:
    ids = set(json.loads(MANIFEST.read_text())["ft_eval_task_ids"])
    conds = sys.argv[1:] or sorted(f.stem for f in EVAL_DIR.glob("*.jsonl"))

    table: dict[str, dict[str, dict]] = {}
    for cond in conds:
        f = EVAL_DIR / f"{cond}.jsonl"
        if not f.exists():
            print(f"skip {cond}: no file")
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines()
                if json.loads(l)["task_id"] in ids]
        by: dict[str, list] = defaultdict(list)
        for r in rows:
            by[r["type"]].append(r)
            by[f"{r['type']}/{r['split']}"].append(r)
        table[cond] = {k: aggregate(v) for k, v in sorted(by.items())}

    for cond, blocks in table.items():
        print(f"\n== {cond} (ft-clean subset) ==")
        for k, agg in blocks.items():
            t = k.split("/")[0]
            vals = " ".join(f"{f}={agg[f]}" for f in KEY_FIELDS[t] if f in agg)
            print(f"  {k:35s} n={agg['n']:3d}  {vals}")


if __name__ == "__main__":
    main()
