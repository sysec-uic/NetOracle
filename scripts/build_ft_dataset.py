#!/usr/bin/env python3
"""Build a leakage-free fine-tuning dataset from the testbed runs.

Eval tasks are rebuilt from the *committed* runs only (the 456-run matrix the
existing results/eval/*.jsonl were produced from), so task_ids join with the
old baseline rows. Training cells may additionally use newer (flywheel) runs.

Split design:
  never trained  - heldout_cc cells (westwood)
                 - heldout_cond cells (rtt=160 / bw=400), incl. flywheel cells
                   that happen to land there
                 - a stratified half of the interp eval targets ("ft_test")
  trainable      - everything else, incl. the other half of interp targets
                   (their eval tasks are then contaminated and excluded from
                   the FT comparison)

The manifest records ft_eval_task_ids: the eval tasks whose target the FT
model has never seen. Baselines are compared on exactly this restriction.

Output: dataset/ft/{train,val}.jsonl (chat messages) + manifest.json
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import subprocess
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "eval"))

from data import RUNS_DIR, load_cells  # noqa: E402
from tasks import KNOBS, _split_of, build_all, cell_key  # noqa: E402
from prompts import counterfactual_prompt, quantitative_prompt  # noqa: E402

OUT_DIR = REPO / "dataset" / "ft"

METRIC_WHITELIST = [
    "goodput_mbps", "rr_p50_ms", "rr_p99_ms", "fct_p50_ms", "fct_p95_ms",
    "cwnd_mid", "srtt_mid_ms",
]

CF_CAP_PER_KNOB = 600
VAL_FRAC = 0.05
SEED = 42


def sig(x: float, digits: int = 4) -> float:
    if x == 0:
        return 0.0
    return round(x, max(0, digits - 1 - math.floor(math.log10(abs(x)))))


EVAL_RUNS_LIST = REPO / "dataset" / "ft" / "eval_runs.txt"


def committed_cells() -> list[dict]:
    """Cells rebuilt from the pinned original run matrix (the 456-run grid
    the existing results/eval rows were produced from). The list is frozen in
    eval_runs.txt so later commits of flywheel runs cannot shift task_ids."""
    if EVAL_RUNS_LIST.exists():
        tracked = EVAL_RUNS_LIST.read_text().split()
    else:  # regenerate from the last pre-flywheel commit
        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-tree", "-r", "2a95ea5",
             "--name-only", "--", "dataset/runs"],
            capture_output=True, text=True, check=True).stdout.split()
        EVAL_RUNS_LIST.write_text("\n".join(tracked) + "\n")
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="netmodel_eval_runs_"))
    for rel in tracked:
        p = REPO / rel
        (tmp / p.name).write_text(p.read_text())
    return load_cells(tmp)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=None,
                    help="training-cell family whitelist, e.g. ABCDU "
                         "(default: every non-smoke family on disk)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="output directory (default dataset/ft)")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)

    rng = random.Random(SEED)

    cells_eval = committed_cells()
    tasks = build_all(cells_eval)
    print(f"eval basis: {len(cells_eval)} cells; tasks: "
          + " ".join(f"{t}={len(l)}" for t, l in tasks.items()))

    # sanity: task list must match the old result rows
    knn_file = REPO / "results" / "eval" / "knn|L2.jsonl"
    if knn_file.exists():
        old = {json.loads(l)["task_id"]: json.loads(l)
               for l in knn_file.read_text().splitlines()}
        rebuilt = {}
        for t, lst in tasks.items():
            for i, task in enumerate(lst):
                rebuilt[f"{t}:{i}"] = task
        mismatch = [tid for tid, r in old.items()
                    if tid not in rebuilt
                    or rebuilt[tid]["split"] != r["split"]
                    or (rebuilt[tid].get("knob") or rebuilt[tid].get("vary")) != r["knob"]]
        assert not mismatch, f"task mismatch vs old results: {mismatch[:10]}"
        print(f"task_id join verified against {knn_file.name} ({len(old)} rows)")

    # --- stratified holdout of interp eval targets --------------------------
    ft_test_cells: set[tuple] = set()
    cf_interp = [t for t in tasks["counterfactual"] if t["split"] == "interp"]
    by_knob: dict[str, list[dict]] = defaultdict(list)
    for t in cf_interp:
        by_knob[t["knob"]].append(t)
    for knob, tlist in sorted(by_knob.items()):
        targets = sorted({cell_key(t["to_cell"]) for t in tlist})
        rng.shuffle(targets)
        ft_test_cells.update(targets[: math.ceil(len(targets) / 2)])

    q_interp = [t for t in tasks["quantitative"] if t["split"] == "interp"]
    q_targets = sorted({cell_key(t["target"]) for t in q_interp})
    rng.shuffle(q_targets)
    # cells already held out via cf count toward the quant half
    need = max(0, math.ceil(len(q_targets) / 2)
               - sum(k in ft_test_cells for k in q_targets))
    for k in q_targets:
        if need <= 0:
            break
        if k not in ft_test_cells:
            ft_test_cells.add(k)
            need -= 1

    # --- training cells ------------------------------------------------------
    cells_all = load_cells(RUNS_DIR)  # includes flywheel runs
    if args.families:
        cells_all = [c for c in cells_all if c["family"] in set(args.families)]
    train_cells, dropped = [], defaultdict(int)
    for c in cells_all:
        if _split_of(c) != "interp":
            dropped["heldout_split"] += 1
        elif cell_key(c) in ft_test_cells:
            dropped["ft_test"] += 1
        else:
            train_cells.append(c)
    print(f"train cells: {len(train_cells)} of {len(cells_all)} "
          f"(dropped {dict(dropped)}; ft_test={len(ft_test_cells)})")

    # --- ft-clean eval task ids ----------------------------------------------
    ft_eval_task_ids = []
    for t, lst in tasks.items():
        for i, task in enumerate(lst):
            if task["split"] != "interp":
                ft_eval_task_ids.append(f"{t}:{i}")
            elif t == "quantitative" and cell_key(task["target"]) in ft_test_cells:
                ft_eval_task_ids.append(f"{t}:{i}")
            elif t == "counterfactual" and cell_key(task["to_cell"]) in ft_test_cells:
                ft_eval_task_ids.append(f"{t}:{i}")
            # interp ranking stays contaminated -> never in ft eval set

    # --- quantitative training examples --------------------------------------
    examples: list[dict] = []
    n_quant = 0
    for c in train_cells:
        for m in METRIC_WHITELIST:
            v = c["metrics"].get(m)
            if v is None or v <= 0:
                continue
            task = {"type": "quantitative", "split": "train", "target": c,
                    "metric": m, "truth": v}
            examples.append({
                "kind": "quantitative", "metric": m,
                "messages": [
                    {"role": "user",
                     "content": quantitative_prompt(task, "L0", [])},
                    {"role": "assistant",
                     "content": json.dumps({"value": sig(v)})},
                ],
            })
            n_quant += 1

    # --- counterfactual training examples -------------------------------------
    pairs_by_knob: dict[str, list[tuple[dict, dict]]] = {k: [] for k in KNOBS}
    for a in train_cells:
        for b in train_cells:
            if a is b or a["workload"] != b["workload"]:
                continue
            diffs = [k for k in KNOBS if a[k] != b[k]]
            if len(diffs) == 1:
                pairs_by_knob[diffs[0]].append((a, b))

    n_cf = 0
    for knob, plist in sorted(pairs_by_knob.items()):
        rng.shuffle(plist)
        for a, b in plist[:CF_CAP_PER_KNOB]:
            va, vb = a["primary_value"], b["primary_value"]
            if not va or not vb or va <= 0:
                continue
            ratio = vb / va
            direction = ("same" if 0.9 <= ratio <= 1.1
                         else "increase" if ratio > 1.1 else "decrease")
            task = {"type": "counterfactual", "split": "train", "knob": knob,
                    "from_cell": a, "to_cell": b,
                    "metric": a["primary_metric"]}
            examples.append({
                "kind": "counterfactual", "knob": knob,
                "messages": [
                    {"role": "user",
                     "content": counterfactual_prompt(task, "L0", [])},
                    {"role": "assistant",
                     "content": json.dumps(
                         {"direction": direction, "ratio": sig(ratio, 3)})},
                ],
            })
            n_cf += 1

    # --- split & write ----------------------------------------------------------
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * VAL_FRAC))
    val, train = examples[:n_val], examples[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    by_split_counts = defaultdict(int)
    for tid in ft_eval_task_ids:
        by_split_counts[tid.split(":")[0]] += 1
    manifest = {
        "seed": SEED,
        "families": args.families,
        "n_cells_eval_basis": len(cells_eval),
        "n_cells_all": len(cells_all),
        "n_cells_train": len(train_cells),
        "n_ft_test_cells": len(ft_test_cells),
        "n_quantitative": n_quant,
        "n_counterfactual": n_cf,
        "cf_pairs_available": {k: len(v) for k, v in pairs_by_knob.items()},
        "cf_cap_per_knob": CF_CAP_PER_KNOB,
        "n_train": len(train),
        "n_val": len(val),
        "n_ft_eval_tasks": len(ft_eval_task_ids),
        "ft_eval_tasks_by_type": dict(by_split_counts),
        "ft_eval_task_ids": ft_eval_task_ids,
        "ft_test_cell_keys": sorted(map(list, ft_test_cells)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"quantitative={n_quant} counterfactual={n_cf} "
          f"(pairs available { {k: len(v) for k, v in pairs_by_knob.items()} })")
    print(f"ft-clean eval tasks: {len(ft_eval_task_ids)} {dict(by_split_counts)}")
    print(f"train={len(train)} val={len(val)} -> {out_dir}")


if __name__ == "__main__":
    main()
