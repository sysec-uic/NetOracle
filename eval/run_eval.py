"""Run the world-model evaluation.

Conditions = model x grounding-level; each condition answers every task; the
kNN baseline runs alongside. Results land in results/eval/<condition>.jsonl,
summaries in results/eval/summary.json.

  python3 run_eval.py --backends cli:haiku,cli:sonnet --levels L0,L1,L2,L3
  python3 run_eval.py --backends knn --levels L2
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import threading
import time
from collections import defaultdict

from baselines import knn_answer
from data import load_cells
from llm import UsageLimitError, make_backend
from metrics import SCORERS, aggregate
from prompts import BUILDERS
from tasks import build_all

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "results" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def task_id(task: dict, i: int) -> str:
    return f"{task['type']}:{i}"


def run_condition(backend_spec: str, level: str, tasks: dict, cells: list[dict],
                  types: list[str], workers: int, limit: int) -> list[dict]:
    cond = f"{backend_spec}|{level}".replace("/", "_").replace(":", "-")
    out_file = OUT_DIR / f"{cond}.jsonl"
    done_ids = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            done_ids.add(json.loads(line)["task_id"])
    todo = []
    for ttype in types:
        for i, task in enumerate(tasks[ttype]):
            if limit and i >= limit:
                break
            tid = task_id(task, i)
            if tid not in done_ids:
                todo.append((tid, task))
    if not todo:
        print(f"[{cond}] complete ({len(done_ids)} rows)")
        return []
    print(f"[{cond}] {len(todo)} tasks to run", flush=True)

    lock = threading.Lock()
    backend = None if backend_spec == "knn" else make_backend(backend_spec)

    def one(item):
        tid, task = item
        if backend is None:
            answer = knn_answer(task, cells)
        else:
            prompt = BUILDERS[task["type"]](task, level, cells)
            answer = backend.query_json(prompt)
        scores = SCORERS[task["type"]](task, answer)
        row = {
            "task_id": tid,
            "type": task["type"],
            "split": task["split"],
            "knob": task.get("knob") or task.get("vary"),
            "answer": answer,
            "scores": scores,
        }
        with lock:
            with open(out_file, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    rows = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for n, row in enumerate(pool.map(one, todo)):
                rows.append(row)
                if (n + 1) % 20 == 0:
                    rate = (n + 1) / (time.time() - t0)
                    print(f"[{cond}] {n + 1}/{len(todo)} ({rate:.1f}/s)", flush=True)
        except UsageLimitError as e:
            pool.shutdown(cancel_futures=True)
            raise SystemExit(f"[{cond}] ABORT, quota exhausted: {e}")
    print(f"[{cond}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def summarize() -> dict:
    summary: dict = {}
    for f in sorted(OUT_DIR.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        cond = f.stem
        summary[cond] = {}
        by_type: dict[str, list] = defaultdict(list)
        by_type_split: dict[tuple, list] = defaultdict(list)
        for r in rows:
            by_type[r["type"]].append(r)
            by_type_split[(r["type"], r["split"])].append(r)
        for t, rs in sorted(by_type.items()):
            summary[cond][t] = aggregate(rs)
        for (t, s), rs in sorted(by_type_split.items()):
            summary[cond][f"{t}/{s}"] = aggregate(rs)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="knn",
                    help="comma list: knn | cli:<model> | api:<model> | codex:<model> | ollama:<model>")
    ap.add_argument("--levels", default="L2", help="comma list of L0,L1,L2,L3")
    ap.add_argument("--types", default="ranking,counterfactual,quantitative")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="max tasks per type (debug)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    if not args.summarize_only:
        cells = load_cells()
        tasks = build_all(cells)
        types = args.types.split(",")
        for spec in args.backends.split(","):
            levels = ["L2"] if spec == "knn" else args.levels.split(",")
            for level in levels:
                run_condition(spec, level, tasks, cells, types, args.workers, args.limit)

    summary = summarize()
    for cond, blocks in summary.items():
        print(f"\n== {cond} ==")
        for t in ("ranking", "counterfactual", "quantitative"):
            if t in blocks:
                b = blocks[t]
                keep = {k: v for k, v in b.items() if not k.endswith("_n")}
                print(f"  {t}: {keep}")


if __name__ == "__main__":
    main()
