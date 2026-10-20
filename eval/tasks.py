"""Build world-model prediction tasks from the execution dataset.

Task types:
  ranking         rank configurations for a fixed workload+condition
  counterfactual  given a *measured* run A, predict the effect of flipping
                  exactly one knob (direction + magnitude bucket)
  quantitative    predict the primary metric value for an unseen config

Generalization splits (which cells the grounding context may draw from):
  interp       context = everything except the target cell(s)
  heldout_cc   all cells using HELDOUT_CC are invisible to context;
               targets use that CC  -> tests text-knowledge transfer
  heldout_cond family-D off-grid conditions (rtt=160ms / bw=400) as targets;
               context contains no cell at those values
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict

from data import PRIMARY_METRIC, config_of, load_cells

HELDOUT_CC = "westwood"
HELDOUT_RTT = 160
HELDOUT_BW = 400

KNOBS = ["cc", "qdisc", "rtt_ms", "loss_pct", "bw_mbit", "buf"]


def _split_of(cell: dict) -> str:
    if cell["cc"] == HELDOUT_CC:
        return "heldout_cc"
    if cell["rtt_ms"] == HELDOUT_RTT or cell["bw_mbit"] == HELDOUT_BW:
        return "heldout_cond"
    return "interp"


def cell_key(c: dict) -> tuple:
    return tuple(c[k] for k in ("workload", "cc", "qdisc", "rtt_ms", "loss_pct", "bw_mbit", "buf"))


def build_ranking_tasks(cells: list[dict]) -> list[dict]:
    """Group cells that differ in exactly one knob dimension; ask for a ranking."""
    tasks = []
    # (a) rank CCs: family A, fixed (workload, rtt, loss)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        if c["family"] in ("A",) and c["workload"] in ("bulk", "mixed"):
            groups[(c["workload"], c["rtt_ms"], c["loss_pct"])].append(c)
    for key, grp in sorted(groups.items()):
        if len(grp) < 4:
            continue
        pm, direction = PRIMARY_METRIC[key[0]]
        tasks.append({
            "type": "ranking",
            "vary": "cc",
            "split": "heldout_cc" if any(c["cc"] == HELDOUT_CC for c in grp) else "interp",
            "context_desc": (f"workload={key[0]}, rtt={key[1]}ms, loss={key[2]}%, "
                             f"bw=100Mbit, qdisc=pfifo_deep(1000pkt), buf=ample"),
            "options": {c["cc"]: c["primary_value"] for c in sorted(grp, key=lambda x: x["cc"])},
            "metric": pm,
            "better": direction,
            "cells": grp,
        })
    # (b) rank qdiscs: family B mixed/short, fixed (workload, cc, bw)
    groups = defaultdict(list)
    for c in cells:
        if c["family"] == "B":
            groups[(c["workload"], c["cc"], c["bw_mbit"])].append(c)
    for key, grp in sorted(groups.items()):
        if len(grp) < 4:
            continue
        pm, direction = PRIMARY_METRIC[key[0]]
        tasks.append({
            "type": "ranking",
            "vary": "qdisc",
            "split": "interp",
            "context_desc": (f"workload={key[0]}, cc={key[1]}, bw={key[2]}Mbit, "
                             f"rtt=20ms, loss=0%, buf=ample"),
            "options": {c["qdisc"]: c["primary_value"] for c in sorted(grp, key=lambda x: x["qdisc"])},
            "metric": pm,
            "better": direction,
            "cells": grp,
        })
    return tasks


def build_counterfactual_tasks(cells: list[dict], per_knob: int = 30, seed: int = 7) -> list[dict]:
    """Pairs differing in exactly one knob, same workload."""
    rng = random.Random(seed)
    index = {cell_key(c): c for c in cells}
    pairs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    seen = set()
    for c in cells:
        for knob in KNOBS:
            for other in cells:
                if other is c or other["workload"] != c["workload"]:
                    continue
                diffs = [k for k in KNOBS if c[k] != other[k]]
                if diffs != [knob]:
                    continue
                pid = (cell_key(c), cell_key(other))
                if pid in seen:
                    continue
                seen.add(pid)
                pairs[knob].append((c, other))
    tasks = []
    for knob, plist in sorted(pairs.items()):
        rng.shuffle(plist)
        # held-out-split pairs are rare and carry the generalization claim:
        # keep them all, fill the remainder with interp pairs
        plist.sort(key=lambda ab: _split_of(ab[1]) == "interp")
        for a, b in plist[:per_knob]:
            pm, direction = PRIMARY_METRIC[a["workload"]]
            va, vb = a["primary_value"], b["primary_value"]
            if va is None or vb is None or va <= 0:
                continue
            ratio = vb / va
            if 0.9 <= ratio <= 1.1:
                truth = "same"
            elif ratio > 1.1:
                truth = "increase"
            else:
                truth = "decrease"
            tasks.append({
                "type": "counterfactual",
                "knob": knob,
                "split": _split_of(b),
                "from_cell": a,
                "to_cell": b,
                "metric": pm,
                "truth_direction": truth,
                "truth_ratio": round(ratio, 3),
            })
    return tasks


def build_quantitative_tasks(cells: list[dict], n: int = 90, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    # stratify: keep all held-out targets, sample the rest
    heldout = [c for c in cells if _split_of(c) != "interp"]
    interp = [c for c in cells if _split_of(c) == "interp"]
    rng.shuffle(interp)
    targets = heldout + interp[: max(0, n - len(heldout))]
    return [{
        "type": "quantitative",
        "split": _split_of(c),
        "target": c,
        "metric": c["primary_metric"],
        "truth": c["primary_value"],
    } for c in targets]


def context_pool(cells: list[dict], task: dict) -> list[dict]:
    """Cells the grounding context may legally draw from for this task."""
    if task["type"] == "ranking":
        forbidden = {cell_key(c) for c in task["cells"]}
    elif task["type"] == "counterfactual":
        forbidden = {cell_key(task["to_cell"])}
    else:
        forbidden = {cell_key(task["target"])}
    pool = [c for c in cells if cell_key(c) not in forbidden]
    split = task["split"]
    if split == "heldout_cc":
        pool = [c for c in pool if c["cc"] != HELDOUT_CC]
    elif split == "heldout_cond":
        pool = [c for c in pool if c["rtt_ms"] != HELDOUT_RTT and c["bw_mbit"] != HELDOUT_BW]
    return pool


def build_all(cells: list[dict] | None = None) -> dict:
    cells = cells or load_cells()
    return {
        "ranking": build_ranking_tasks(cells),
        "counterfactual": build_counterfactual_tasks(cells),
        "quantitative": build_quantitative_tasks(cells),
    }


if __name__ == "__main__":
    cells = load_cells()
    tasks = build_all(cells)
    for t, lst in tasks.items():
        by_split: dict[str, int] = defaultdict(int)
        for x in lst:
            by_split[x["split"]] += 1
        print(f"{t}: {len(lst)} tasks {dict(by_split)}")
